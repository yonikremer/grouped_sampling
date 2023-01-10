from __future__ import annotations

import json
import os
from threading import Thread, Event
from pathlib import Path
from typing import Any, Dict, Tuple, List, Callable
from warnings import warn

from evaluate import TranslationEvaluator
from datasets import load_dataset, Dataset, get_dataset_config_names
# noinspection PyUnresolvedReferences
from nvidia_smi import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates

from evaluation.experiment_manager import ExperimentManager
from evaluation import lang_code_to_name
from src.grouped_sampling import GroupedSamplingPipeLine
from src.grouped_sampling import GroupedGenerationPipeLine

from transformers.utils.logging import disable_progress_bar

disable_progress_bar()

from datasets.utils.logging import disable_progress_bar

disable_progress_bar()

DATASET_NAME = "ted_talks_iwslt"


def check_gpu_utilization(func: Callable) -> Callable:
    """A decorator that checks the GPU utilization during the execution of the wrapped function
    And prints a warning if it is zero for interval consecutive seconds"""
    def wrapper(*args, **kwargs):
        # Initialize the NVML library
        nvmlInit()

        # Get the handle to the GPU device
        device_index = 0  # index of the GPU device to use
        handle = nvmlDeviceGetHandleByIndex(device_index)

        # Start the GPU utilization checking thread
        stop_flag = Event()
        utilization_thread = Thread(target=_check_utilization, args=(handle, stop_flag))
        utilization_thread.start()

        # Run the original function
        result = func(*args, **kwargs)

        # Stop the GPU utilization checking thread
        stop_flag.set()
        utilization_thread.join()

        return result

    return wrapper


def _check_utilization(handle: nvmlDeviceGetHandleByIndex, stop_flag: Event) -> None:
    """A thread that checks the GPU utilization every second
     during the execution of the wrapped function
     and sends a warning if the gpu utilization is zero for interval seconds"""
    while not stop_flag.is_set():
        # Get the GPU utilization using nvidia_smi
        utilization = nvmlDeviceGetUtilizationRates(handle)
        gpu_utilization = utilization.gpu
        if gpu_utilization > 0:
            return
    # Print a warning if the GPU utilization is zero
    warn("GPU utilization is zero")


def process_translation_data(sub_set_name: str, debug: bool) -> Tuple[Dataset, Dataset, str, str]:
    spited_sub_set_name = sub_set_name.split("_")
    language_code1, language_code2 = spited_sub_set_name[:2]
    # noinspection PyUnreachableCode
    if debug:
        sub_set = load_dataset(DATASET_NAME, sub_set_name, split="train[:2]")
    else:
        sub_set = load_dataset(DATASET_NAME, sub_set_name, split="train")
    processed_data1_dict: Dataset
    processed_data2_dict: Dataset

    def rename_keys(x: Dict[str, Any], input_lang_name: str, output_lang_name: str) -> Dict[str, str]:
        translation: Dict[str, str] = x["translation"]
        return {input_lang_name: translation[input_lang_name], output_lang_name: translation[output_lang_name]}

    subset_part1 = sub_set.map(rename_keys,
                               fn_kwargs={"input_lang_name": language_code1, "output_lang_name": language_code2})
    subset_part2 = sub_set.map(rename_keys,
                               fn_kwargs={"input_lang_name": language_code2, "output_lang_name": language_code1})
    return subset_part1, subset_part2, language_code1, language_code2


def sub_experiment_half(
        my_evaluator: TranslationEvaluator,
        sub_set_half: Dataset,
        in_lang_code: str, out_lang_code: str,
        generator: GroupedGenerationPipeLine,
        manager: ExperimentManager) -> None:
    input_lang_name, output_lang_name = lang_code_to_name(in_lang_code), lang_code_to_name(out_lang_code)
    prefix = f"Translate {input_lang_name} to {output_lang_name}: \n {input_lang_name}: "
    postfix = f"\n {output_lang_name}: "
    my_evaluator.METRIC_KWARGS = {"lang": out_lang_code}
    my_evaluator.PIPELINE_KWARGS = {"prefix": prefix, "postfix": postfix}
    # noinspection PyTypeChecker
    scores: Dict[str, List[float] | Any] = my_evaluator.compute(
        model_or_pipeline=generator,
        data=sub_set_half,
        input_column=in_lang_code,
        label_column=out_lang_code
    )
    manager.log_sub_experiment(scores, in_lang_code, out_lang_code, sub_set_half)


def run_experiment(
        generator: GroupedGenerationPipeLine,
        my_evaluator: TranslationEvaluator,
        sub_sut_names: List[str],
        debug: bool,
) -> None:
    generator.task = "translation"
    manager = ExperimentManager(generator, debug=debug)
    for i, sub_set_name in enumerate(sub_sut_names):
        print(f"Running sub-experiment {i + 1} out of {len(sub_sut_names)}")
        subset_part1: Dataset
        subset_part2: Dataset
        language_code1: str
        language_code2: str
        subset_part1, subset_part2, language_code1, language_code2 = process_translation_data(sub_set_name, debug)
        sub_experiment_half(my_evaluator, subset_part1, language_code1, language_code2, generator, manager)
        if not debug:
            sub_experiment_half(my_evaluator, subset_part2, language_code2, language_code1, generator, manager)
    manager.end_experiment()


def create_text_generator() -> GroupedGenerationPipeLine:
    """Creates a text generator from the evaluated_text_generator_dict.json file"""
    parent_folder = Path(__file__).parent
    with open(os.path.join(parent_folder, "evaluated_text_generator_dict.json"), "r") as json_file:
        evaluated_text_generator_dict = json.load(json_file)
    my_generator = GroupedSamplingPipeLine.from_dict(evaluated_text_generator_dict)
    my_generator.get_logits_matrix = check_gpu_utilization(my_generator.get_logits_matrix)
    return my_generator


def create_evaluator() -> TranslationEvaluator:
    """Creates a translation evaluator"""
    METRIC_NAME = "bertscore"
    my_evaluator = TranslationEvaluator(default_metric_name=METRIC_NAME)
    my_evaluator.PREDICTION_PREFIX = "generated"
    return my_evaluator


def main(debug: bool = __debug__) -> None:
    if debug:
        # send a warning
        warn("Running in debug mode, only a small subset of the data will be used")
    sub_sut_names = get_dataset_config_names(DATASET_NAME)
    if debug:
        sub_sut_names = sub_sut_names[:1]
    curr_text_generator = create_text_generator()
    curr_evaluator = create_evaluator()
    run_experiment(curr_text_generator, curr_evaluator, sub_sut_names, debug)


if __name__ == "__main__":
    main()
