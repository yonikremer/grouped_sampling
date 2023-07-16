import gc
import os
import time
from os.path import abspath, dirname
from typing import List, Iterable

import torch
from datasets import get_dataset_config_names
from torch import inference_mode
import pandas as pd
# noinspection PyProtectedMember
from torch._C._profiler import ProfilerActivity
from torch.profiler import profile

from evaluation import (
    process_translation_data,
    DATASET_NAME,
    lang_code_to_name,
    create_pipeline,
    get_experiment_parameters,
)
from fix_bitsandbytes import fix_ld_library_path
from src.grouped_sampling import get_tokenizer

curr_dir = dirname(abspath(__file__))


def prompt_engineering(
    prompts: Iterable[str], input_language_code: str, output_language_code: str
):
    input_language = lang_code_to_name(input_language_code)
    output_language = lang_code_to_name(output_language_code)
    return [
        f"Translate {input_language} to {output_language}.\n{input_language}: {prompt} \n{output_language}: "
        for prompt in prompts
    ]


def get_prompts(debug: bool) -> List[str]:
    """Gets a list of all the examples in the dataset with name DATASET_NAME"""
    sub_set_names: List[str] = get_dataset_config_names(DATASET_NAME)
    if debug:
        sub_set_names = sub_set_names[:1]
    prompts: List[str] = []
    language_code1: str
    language_code2: str
    for sub_set_name in sub_set_names:
        subset_part, _, language_code1, language_code2 = process_translation_data(
            sub_set_name, debug)
        prompts.extend(subset_part[language_code1])
        prompts.extend(subset_part[language_code2])
    return prompts


@inference_mode()
def generate(prompts: List[str], max_batch_size: int, max_prompt_length: int):
    pipeline = create_pipeline(max_batch_size)
    output_length = pipeline.max_total_len - max_prompt_length - 1
    pipeline.max_batch_size = max_batch_size
    start_time = time.time()
    with profile(
        activities=[ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as profiler:
        pipeline.genearte_batch_return_one(prompts, output_length)
    print(profiler.key_averages().table(sort_by="self_cuda_memory_usage"))
    end_time = time.time()
    duration_seconds = end_time - start_time
    del pipeline
    torch.cuda.empty_cache()
    gc.collect()
    return duration_seconds


@inference_mode()
def main(debug: bool = False):
    fix_ld_library_path()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model_name = get_experiment_parameters()["model_name"]
    tokenizer = get_tokenizer(model_name)
    prompts: List[str] = get_prompts(debug=debug)
    batch_size_to_duration = {}
    max_prompt_length = max(len(tokenizer.encode(prompt))
                            for prompt in prompts)
    max_batch_size = 2
    batch_size_to_duration[max_batch_size] = generate(
        prompts, max_batch_size, max_prompt_length
    )
    optimal_batch_size = min(
        batch_size_to_duration,
        key=batch_size_to_duration.get)
    print(f"Optimal batch size: {optimal_batch_size}")
    print(
        f"Duration for batch size {optimal_batch_size}: {batch_size_to_duration[optimal_batch_size]}"
    )
    df = pd.DataFrame.from_dict(
        batch_size_to_duration, orient="index", columns=["Duration"]
    )
    df.to_csv(f"{curr_dir}/batch_size_to_duration.csv")
    df.plot()


if __name__ == "__main__":
    main(False)
