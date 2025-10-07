# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the latency of processing a single batch of requests."""

import argparse
import dataclasses
import json
import os
import time
from typing import Any, Optional

import numpy as np
from tqdm import tqdm

import vllm.envs as envs
from vllm.benchmarks.lib.utils import (convert_to_pytorch_benchmark_format,
                                       write_to_json)
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import PromptType

from src.grouped_sampling import ReturnM


def save_to_pytorch_benchmark_format(args: argparse.Namespace,
                                     results: dict[str, Any]) -> None:
    pt_records = convert_to_pytorch_benchmark_format(
        args=args,
        metrics={"latency": results["latencies"]},
        extra_info={k: results[k]
                    for k in ["avg_latency", "percentiles"]})
    if pt_records:
        pt_file = f"{os.path.splitext(args.output_json)[0]}.pytorch.json"
        write_to_json(pt_file, pt_records)


def add_cli_args(parser: argparse.ArgumentParser):
    parser.add_argument("--input-len", type=int, default=32)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of generated sequences per prompt.",
    )
    parser.add_argument(
        "--num-iters-warmup",
        type=int,
        default=10,
        help="Number of iterations to run for warmup.",
    )
    parser.add_argument("--num-iters",
                        type=int,
                        default=30,
                        help="Number of iterations to run.")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="profile the generation process of a single batch",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        help="Path to save the latency results in JSON format.",
    )

    parser = EngineArgs.add_cli_args(parser)
    # V1 enables prefix caching by default, which skews the latency
    # numbers. We need to disable prefix caching by default.
    parser.set_defaults(enable_prefix_caching=False)


def main(args: argparse.Namespace):
    if args.profile and not envs.VLLM_TORCH_PROFILER_DIR:
        raise OSError(
            "The environment variable 'VLLM_TORCH_PROFILER_DIR' is not set. "
            "Please set it to a valid path to use torch profiler.")
    engine_args = EngineArgs.from_cli_args(args)

    # NOTE(woosuk): If the request cannot be processed in a single batch,
    # the engine will automatically process the request in multiple batches.
    llm = LLM(**dataclasses.asdict(engine_args))
    assert llm.llm_engine.model_config.max_model_len >= (
            args.input_len +
            args.output_len), ("Please ensure that max_model_len is greater than"
                               " the sum of input_len and output_len.")

    sampling_params = SamplingParams(
        n=args.n,
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=args.output_len,
        detokenize=True,
    )
    rng = np.random.default_rng(seed=0)
    dummy_prompt_token_ids = rng.integers(10000,
                                               size=(args.batch_size,
                                                     args.input_len))
    dummy_prompts: list[PromptType] = [{
        "prompt_token_ids": batch
    } for batch in dummy_prompt_token_ids.tolist()]

    def run_to_completion(curr_profile_dir: Optional[str] = None) -> Optional[float]:
        if curr_profile_dir:
            llm.start_profile()
            llm.generate(dummy_prompts,
                         sampling_params=sampling_params,
                         use_tqdm=False)
            llm.stop_profile()
            return None
        else:
            start_time = time.perf_counter()
            llm.generate(dummy_prompts,
                         sampling_params=sampling_params,
                         use_tqdm=False)
            end_time = time.perf_counter()
            latency = end_time - start_time
            return latency

    print("Warming up...")
    for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
        run_to_completion(curr_profile_dir=None)

    if args.profile:
        profile_dir = envs.VLLM_TORCH_PROFILER_DIR
        print(f"Profiling (results will be saved to '{profile_dir}')...")
        run_to_completion(curr_profile_dir=profile_dir)
        return

    # Benchmark.
    latencies = []
    for _ in tqdm(range(args.num_iters), desc="Profiling iterations"):
        latencies.append(run_to_completion(curr_profile_dir=None))
    latencies = np.array(latencies)
    percentages = [10, 25, 50, 75, 90, 99]
    percentiles = np.percentile(latencies, percentages)
    print(f"Avg latency: {np.mean(latencies)} seconds")
    for percentage, percentile in zip(percentages, percentiles):
        print(f"{percentage}% percentile latency: {percentile} seconds")


    results = {
        "avg_latency": np.mean(latencies),
        "latencies": latencies.tolist(),
        "percentiles": dict(zip(percentages, percentiles.tolist())),
    }
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=4)
    save_to_pytorch_benchmark_format(args, results)
