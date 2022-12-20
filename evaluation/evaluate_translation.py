from __future__ import annotations

import json
import os
from threading import Thread
from time import sleep, strftime, localtime
from pathlib import Path
from typing import Any, Dict, Tuple, List

from evaluate import TranslationEvaluator
from datasets import load_dataset, Dataset, get_dataset_config_names
from nvidia_smi import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates

from evaluation.experiment_manager import ExperimentManager
from evaluation.helpers import lang_code_to_name
from sampling_generator import SamplingGenerator
from text_generator import TextGenerator

from transformers.utils.logging import disable_progress_bar

disable_progress_bar()

from datasets.utils.logging import disable_progress_bar

disable_progress_bar()

DATASET_NAME = "ted_talks_iwslt"
METRIC_NAME = "bertscore"
stop_gpu_check = False


def check_gpu_utilization():
    # Initialize the NVML library
    nvmlInit()
    device_index = 0  # index of the GPU device to use

    while not stop_gpu_check:
        # Get the handle to the GPU device
        handle = nvmlDeviceGetHandleByIndex(device_index)

        # Get the GPU utilization using nvidia_smi
        utilization = nvmlDeviceGetUtilizationRates(handle)
        gpu_utilization = utilization.gpu

        # Print a warning if the GPU utilization is zero
        if gpu_utilization == 0:
            print("Warning: GPU utilization is zero", strftime("%H:%M:%S", localtime()))

        # Sleep for 30 seconds before checking the utilization again
        sleep(30)


def process_translation_data(sub_set_name: str) -> Tuple[Dataset, Dataset, str, str]:
    spited_sub_set_name = sub_set_name.split("_")
    language_code1, language_code2 = spited_sub_set_name[:2]
    # noinspection PyUnreachableCode
    if __debug__:
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
        generator: TextGenerator,
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
        generator: TextGenerator,
        my_evaluator: TranslationEvaluator,
        sub_sut_names: List[str]
) -> None:
    generator.task = "translation"
    manager = ExperimentManager(generator)
    for i, sub_set_name in enumerate(sub_sut_names):
        print(f"Running sub-experiment {i + 1} out of {len(sub_sut_names)}")
        subset_part1: Dataset
        subset_part2: Dataset
        language_code1: str
        language_code2: str
        subset_part1, subset_part2, language_code1, language_code2 = process_translation_data(sub_set_name)
        sub_experiment_half(my_evaluator, subset_part1, language_code1, language_code2, generator, manager)
        # noinspection PyUnreachableCode
        if not __debug__:
            sub_experiment_half(my_evaluator, subset_part2, language_code2, language_code1, generator, manager)
    manager.end_experiment()


def main() -> None:
    if __debug__:
        # send a warning
        print("WARING: debug mode is on, only a small subset of the data will be used")
    my_evaluator = TranslationEvaluator(default_metric_name=METRIC_NAME)
    my_evaluator.PREDICTION_PREFIX = "generated"
    sub_sut_names = get_dataset_config_names(DATASET_NAME)
    if __debug__:
        sub_sut_names = sub_sut_names[:1]
    parent_folder = Path(__file__).parent
    with open(os.path.join(parent_folder, "evaluated_text_generator_dict.json"), "r") as json_file:
        evaluated_text_generator_dict = json.load(json_file)
    curr_text_generator = SamplingGenerator.from_dict(evaluated_text_generator_dict)
    utilization_thread = Thread(target=check_gpu_utilization)
    utilization_thread.start()
    run_experiment(curr_text_generator, my_evaluator, sub_sut_names)
    global stop_gpu_check
    stop_gpu_check = True
    utilization_thread.join()


if __name__ == "__main__":
    main()
