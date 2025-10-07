# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark offline inference throughput."""
import argparse
import dataclasses
import json
import os
import random
import time
import warnings
from typing import Any, Optional, Union
import cProfile

import torch
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)

from benchmark.output import RequestOutput
from benchmark.datasets import (AIMODataset, BurstGPTDataset,
                                      InstructCoderDataset,
                                      PrefixRepetitionRandomDataset,
                                      RandomDataset, SampleRequest,
                                      ShareGPTDataset, SonnetDataset)
from benchmark.lib.utils import (convert_to_pytorch_benchmark_format,
                                       write_to_json)
from src.grouped_sampling import ReturnManyPipeLine

from vllm.engine.arg_utils import EngineArgs


def run_vllm(
        requests: list[SampleRequest],
        n: int,
        engine_args: EngineArgs,
        do_profile: bool,
) -> tuple[float, Optional[list[RequestOutput]]]:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt, TextPrompt
    llm = LLM(**dataclasses.asdict(engine_args))
    assert all(
        llm.llm_engine.model_config.max_model_len >= (
                request.prompt_len + request.expected_output_len)
        for request in requests), (
        "Please ensure that max_model_len is greater than the sum of"
        " prompt_len and expected_output_len for all requests.")
    # Add the requests to the engine.
    prompts: list[Union[TextPrompt, TokensPrompt]] = []
    sampling_params: list[SamplingParams] = []
    for request in requests:
        prompts.append(
            TokensPrompt(prompt_token_ids=request.prompt["prompt_token_ids"])
            if "prompt_token_ids" in request.prompt else \
                TextPrompt(prompt=request.prompt))
        sampling_params.append(
            SamplingParams(
                n=n,
                temperature=1.0,
                top_p=1.0,
                ignore_eos=True,
                max_tokens=request.expected_output_len,
                detokenize=True,
            ))

    start = time.perf_counter()
    if do_profile:
        llm.start_profile()
    outputs = llm.generate(prompts,
                           sampling_params,
                           use_tqdm=True)
    if do_profile:
        llm.stop_profile()
    end = time.perf_counter()
    return end - start, outputs


def run_hf(
        requests: list[SampleRequest],
        model: str,
        tokenizer: PreTrainedTokenizerBase,
        n: int,
        max_batch_size: int,
        trust_remote_code: bool,
) -> float:
    llm = AutoModelForCausalLM.from_pretrained(
        model, dtype=torch.float16, trust_remote_code=trust_remote_code)
    if llm.config.model_type == "llama":
        # To enable padding in the HF backend.
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    llm = llm.cuda()

    pbar = tqdm(total=len(requests))
    start = time.perf_counter()
    batch: list[str] = []
    max_prompt_len = 0
    max_output_len = 0
    for i in range(len(requests)):
        prompt = requests[i].prompt
        prompt_len = requests[i].prompt_len
        output_len = requests[i].expected_output_len
        # Add the prompt to the batch.
        batch.append(prompt)
        max_prompt_len = max(max_prompt_len, prompt_len)
        max_output_len = max(max_output_len, output_len)
        if len(batch) < max_batch_size and i != len(requests) - 1:
            # Check if we can add more requests to the batch.
            next_prompt_len = requests[i + 1].prompt_len
            next_output_len = requests[i + 1].expected_output_len
            if (max(max_prompt_len, next_prompt_len) +
                max(max_output_len, next_output_len)) <= 2048:
                # We can add more requests to the batch.
                continue

        # Generate the sequences.
        input_ids = tokenizer(batch, return_tensors="pt",
                              padding=True, padding_side='left').input_ids
        llm_outputs = llm.generate(
            input_ids=input_ids.cuda(),
            do_sample=True,
            num_return_sequences=n,
            temperature=1.0,
            top_p=1.0,
            use_cache=True,
            max_new_tokens=max_output_len,
        )
        # Include the decoding time.
        tokenizer.batch_decode(llm_outputs, skip_special_tokens=True)
        pbar.update(len(batch))

        # Clear the batch.
        batch = []
        max_prompt_len = 0
        max_output_len = 0
    end = time.perf_counter()
    return end - start


def run_grouped_sampling(
        requests: list[SampleRequest],
        model: str,
        n: int,
        max_batch_size: int,
) -> float:
    pipeline = ReturnManyPipeLine(model_name=model, seed=0, temperature=1.0, top_p=1.0)
    pbar = tqdm(total=len(requests))
    start = time.perf_counter()
    batch: list[str] = []
    max_prompt_len = 0
    max_output_len = 0
    profiler = cProfile.Profile()
    profiler.enable()
    all_outputs: list[torch.Tensor] = []
    for i in range(len(requests)):
        prompt = requests[i].prompt
        prompt_len = requests[i].prompt_len
        output_len = requests[i].expected_output_len
        # Add the prompt to the batch.
        batch.append(prompt)
        max_prompt_len = max(max_prompt_len, prompt_len)
        max_output_len = max(max_output_len, output_len)
        if len(batch) < max_batch_size and i != len(requests) - 1:
            # Check if we can add more requests to the batch.
            next_prompt_len = requests[i + 1].prompt_len
            next_output_len = requests[i + 1].expected_output_len
            if (max(max_prompt_len, next_prompt_len) +
                max(max_output_len, next_output_len)) <= 2048:
                # We can add more requests to the batch.
                continue

        tokenized_batch = pipeline.tokenize_and_pad(
            batch, max_output_len
        )
        output_tokens: torch.Tensor = pipeline.generate_return_many_tokens(
            prompts=tokenized_batch,
            num_return_sequences=n,
            output_length=max_output_len,
        )
        all_outputs.extend(output_tokens.tolist())
        pbar.update(len(batch))

        # Clear the batch.
        batch = []
        max_prompt_len = 0
        max_output_len = 0
    _output_strings = pipeline.tokenizer.batch_decode(all_outputs, skip_special_tokens=True)
    profiler.disable()
    profiler.dump_stats('benchmark/my_profile_data.prof')
    end = time.perf_counter()
    return end - start


def save_to_pytorch_benchmark_format(args: argparse.Namespace,
                                     results: dict[str, Any]) -> None:
    pt_records = convert_to_pytorch_benchmark_format(
        args=args,
        metrics={
            "requests_per_second": [results["requests_per_second"]],
            "tokens_per_second": [results["tokens_per_second"]],
        },
        extra_info={
            k: results[k]
            for k in ["elapsed_time", "num_requests", "total_num_tokens"]
        })
    if pt_records:
        # Don't use JSON suffix here as we don't want CI to pick it up
        pt_file = f"{os.path.splitext(args.output_json)[0]}.pytorch.json"
        write_to_json(pt_file, pt_records)


def get_requests(args, tokenizer):
    # Common parameters for all dataset types.
    common_kwargs = {
        "dataset_path": args.dataset_path,
    }
    sample_kwargs = {
        "tokenizer": tokenizer,
        "num_requests": args.num_prompts,
        "input_len": args.input_len,
        "output_len": args.output_len,
    }

    if args.dataset_path is None or args.dataset_name == "random":
        sample_kwargs["range_ratio"] = args.random_range_ratio
        sample_kwargs["prefix_len"] = args.prefix_len
        dataset_cls = RandomDataset
    elif args.dataset_name == "sharegpt":
        dataset_cls = ShareGPTDataset
    elif args.dataset_name == "sonnet":
        assert tokenizer.chat_template or tokenizer.default_chat_template, (
            "Tokenizer/model must have chat template for sonnet dataset.")
        dataset_cls = SonnetDataset
        sample_kwargs["prefix_len"] = args.prefix_len
        sample_kwargs["return_prompt_formatted"] = True
    elif args.dataset_name == "burstgpt":
        dataset_cls = BurstGPTDataset
    elif args.dataset_name == "hf":
        if args.dataset_path in InstructCoderDataset.SUPPORTED_DATASET_PATHS:
            dataset_cls = InstructCoderDataset
            common_kwargs['dataset_split'] = "train"
        elif args.dataset_path in ConversationDataset.SUPPORTED_DATASET_PATHS:
            dataset_cls = ConversationDataset
            common_kwargs['dataset_subset'] = args.hf_subset
            common_kwargs['dataset_split'] = args.hf_split
        elif args.dataset_path in AIMODataset.SUPPORTED_DATASET_PATHS:
            dataset_cls = AIMODataset
            common_kwargs['dataset_subset'] = None
            common_kwargs['dataset_split'] = "train"
    elif args.dataset_name == "prefix_repetition":
        dataset_cls = PrefixRepetitionRandomDataset
        sample_kwargs["prefix_len"] = args.prefix_repetition_prefix_len
        sample_kwargs["suffix_len"] = args.prefix_repetition_suffix_len
        sample_kwargs["num_prefixes"] = args.prefix_repetition_num_prefixes
        sample_kwargs["output_len"] = args.prefix_repetition_output_len
    else:
        raise ValueError(f"Unknown dataset name: {args.dataset_name}")
    # Remove None values
    sample_kwargs = {k: v for k, v in sample_kwargs.items() if v is not None}
    requests = dataset_cls(**common_kwargs).sample(**sample_kwargs)
    requests = filter_requests_for_dp(requests, args.data_parallel_size)
    return requests


def filter_requests_for_dp(requests, data_parallel_size):
    # Note(zhuohan): The way we get data_parallel_rank is hacky and only
    # works for external launcher mode. Should be cleaned up and deprecated
    # in the future with a better vLLM distributed process design.
    if data_parallel_size == 1:
        return requests

    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    data_parallel_rank = global_rank // (world_size // data_parallel_size)
    return [r for i, r in enumerate(requests)
            if i % data_parallel_size == data_parallel_rank]


def validate_args(args):
    """
    Validate command-line arguments.
    """

    # === Deprecation and Defaulting ===
    if args.dataset is not None:
        warnings.warn(
            "The '--dataset' argument will be deprecated in the next release. "
            "Please use '--dataset-name' and '--dataset-path' instead.",
            stacklevel=2)
        args.dataset_path = args.dataset

    if not getattr(args, "tokenizer", None):
        args.tokenizer = args.model

    # === Backend Validation ===
    valid_backends = {"vllm", "hf", "grouped-sampling"}
    if args.backend not in valid_backends:
        raise ValueError(f"Unsupported backend: {args.backend}")

    # === Dataset Configuration ===
    if (
            not args.dataset
            and not args.dataset_path
            and args.dataset_name not in {"prefix_repetition"}
    ):
        print(
            "When dataset path is not set, it will default to random dataset")
        args.dataset_name = 'random'
        if args.input_len is None:
            raise ValueError("input_len must be provided for a random dataset")

    # === Dataset Name Specific Checks ===
    # --hf-subset and --hf-split: only used
    # when dataset_name is 'hf'
    if args.dataset_name != "hf" and (
            getattr(args, "hf_subset", None) is not None
            or getattr(args, "hf_split", None) is not None):
        warnings.warn("--hf-subset and --hf-split will be ignored \
                since --dataset-name is not 'hf'.",
                      stacklevel=2)
    elif args.dataset_name == "hf":
        if args.dataset_path in (InstructCoderDataset.SUPPORTED_DATASET_PATHS
                                   | AIMODataset.SUPPORTED_DATASET_PATHS):
            assert args.backend == "vllm", f"{args.dataset_path} needs to use vllm as the backend."  # noqa: E501
        else:
            raise ValueError(
                f"{args.dataset_path} is not supported by hf dataset.")

    # --random-range-ratio: only used when dataset_name is 'random'
    if args.dataset_name != 'random' and args.random_range_ratio is not None:
        warnings.warn("--random-range-ratio will be ignored since \
                --dataset-name is not 'random'.",
                      stacklevel=2)

    # --prefix-len: only used when dataset_name is 'random', 'sonnet', or not
    # set.
    if args.dataset_name not in {"random", "sonnet", None
                                 } and args.prefix_len is not None:
        warnings.warn("--prefix-len will be ignored since --dataset-name\
                 is not 'random', 'sonnet', or not set.",
                      stacklevel=2)

    # === Backend-specific Validations ===
    if args.backend == "hf" and args.hf_max_batch_size is None:
        raise ValueError("HF max batch size is required for HF backend")
    if args.backend not in {"hf", "grouped-sampling"} and args.hf_max_batch_size is not None:
        raise ValueError("HF max batch size is only for HF backend.")


def add_cli_args(parser: argparse.ArgumentParser):
    parser.add_argument("--backend",
                        type=str,
                        choices=["vllm", "hf", "grouped-sampling"],
                        default="vllm")
    parser.add_argument(
        "--dataset-name",
        type=str,
        choices=[
            "sharegpt", "random", "sonnet", "burstgpt", "hf",
            "prefix_repetition"
        ],
        help="Name of the dataset to benchmark on.",
        default="sharegpt")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to the ShareGPT dataset, will be deprecated in\
            the next release. The dataset is expected to "
             "be a json in form of list[dict[..., conversations: "
             "list[dict[..., value: <prompt_or_response>]]]]")
    parser.add_argument("--dataset-path",
                        type=str,
                        default=None,
                        help="Path to the dataset")
    parser.add_argument("--input-len",
                        type=int,
                        default=None,
                        help="Input prompt length for each request")
    parser.add_argument("--output-len",
                        type=int,
                        default=None,
                        help="Output length for each request. Overrides the "
                             "output length from the dataset.")
    parser.add_argument("--n",
                        type=int,
                        default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--hf-max-batch-size",
                        type=int,
                        default=None,
                        help="Maximum batch size for HF backend.")
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Path to save the throughput results in JSON format.')
    parser.add_argument("--disable-frontend-multiprocessing",
                        action='store_true',
                        default=False,
                        help="Disable decoupled async engine frontend.")
    parser.add_argument(
        "--disable-detokenize",
        action="store_true",
        help=("Do not detokenize the response (i.e. do not include "
              "detokenization time in the measurement)"))
    parser.add_argument(
        "--prefix-len",
        type=int,
        default=0,
        help="Number of fixed prefix tokens before the random "
             "context in a request (default: 0).",
    )
    # random dataset
    parser.add_argument(
        "--random-range-ratio",
        type=float,
        default=0.0,
        help="Range ratio for sampling input/output length, "
             "used only for RandomDataset. Must be in the range [0, 1) to define "
             "a symmetric sampling range "
             "[length * (1 - range_ratio), length * (1 + range_ratio)].",
    )

    # hf dtaset
    parser.add_argument("--hf-subset",
                        type=str,
                        default=None,
                        help="Subset of the HF dataset.")
    parser.add_argument("--hf-split",
                        type=str,
                        default=None,
                        help="Split of the HF dataset.")
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Use Torch Profiler. The env variable "
             "VLLM_TORCH_PROFILER_DIR must be set to enable profiler.")

    # prefix repetition dataset
    prefix_repetition_group = parser.add_argument_group(
        "prefix repetition dataset options")
    prefix_repetition_group.add_argument(
        "--prefix-repetition-prefix-len",
        type=int,
        default=None,
        help="Number of prefix tokens per request, used only for prefix "
             "repetition dataset.",
    )
    prefix_repetition_group.add_argument(
        "--prefix-repetition-suffix-len",
        type=int,
        default=None,
        help="Number of suffix tokens per request, used only for prefix "
             "repetition dataset. Total input length is prefix_len + suffix_len.",
    )
    prefix_repetition_group.add_argument(
        "--prefix-repetition-num-prefixes",
        type=int,
        default=None,
        help="Number of prefixes to generate, used only for prefix repetition "
             "dataset. Prompts per prefix is num_requests // num_prefixes.",
    )
    prefix_repetition_group.add_argument(
        "--prefix-repetition-output-len",
        type=int,
        default=None,
        help="Number of output tokens per request, used only for prefix "
             "repetition dataset.",
    )
    _parser = EngineArgs.add_cli_args(parser)
    print(_parser.parse_args())


def main(args: argparse.Namespace):
    if args.tokenizer is None:
        args.tokenizer = args.model
    validate_args(args)
    random.seed(0)
    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code, padding_side="left"
    )
    requests = get_requests(args, tokenizer)
    request_outputs: Optional[list[RequestOutput]] = None
    if args.backend == "vllm":
        elapsed_time, request_outputs = run_vllm(
            requests, args.n, EngineArgs.from_cli_args(args),
            do_profile=args.profile)
    elif args.backend == "hf":
        if args.profile:
            raise NotImplementedError(
                "Profiling not implemented yet for backend='hf'.")
        elapsed_time = run_hf(requests, args.model, tokenizer, args.n,
                              args.hf_max_batch_size, args.trust_remote_code,
                              )
    elif args.backend == "grouped-sampling":
        elapsed_time = run_grouped_sampling(requests, args.model, args.n, args.hf_max_batch_size)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")


    total_num_tokens = sum(r.prompt_len + r.expected_output_len
                           for r in requests)
    total_output_tokens = sum(r.expected_output_len for r in requests)
    total_prompt_tokens = total_num_tokens - total_output_tokens

    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
          f"{total_num_tokens / elapsed_time:.2f} total tokens/s, "
          f"{total_output_tokens / elapsed_time:.2f} output tokens/s")
    print(f"Total num prompt tokens:  {total_prompt_tokens}")
    print(f"Total num output tokens:  {total_output_tokens}")

    results = {
        "elapsed_time": elapsed_time,
        "num_requests": len(requests),
        "total_num_tokens": total_num_tokens,
        "requests_per_second": len(requests) / elapsed_time,
        "tokens_per_second": total_num_tokens / elapsed_time,
    }
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=4)
    save_to_pytorch_benchmark_format(args, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_cli_args(parser)
    args = parser.parse_args()
    main(args)

