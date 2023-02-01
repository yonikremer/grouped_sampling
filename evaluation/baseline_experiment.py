from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
from warnings import warn

from datasets import Dataset, get_dataset_config_names
from evaluate import EvaluationModule, load
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline

from evaluation import (
    DATASET_NAME,
    disable_progress_bars,
    lang_code_to_name,
    process_translation_data,
)
from evaluation.experiment_manager import ExperimentManager
from src.grouped_sampling import GenerationType

disable_progress_bars()

METRIC_NAME = "bertscore"
metric: EvaluationModule = load(METRIC_NAME,
                                cache_dir=os.path.join(
                                    os.path.dirname(__file__), "metrics",
                                    "cache"))


def process_sub_set_half(sub_set_half: Dataset, in_lang_code: str,
                         out_lang_code: str) -> Tuple[List[str], List[str]]:
    input_lang_name = lang_code_to_name(in_lang_code)
    output_lang_name = lang_code_to_name(out_lang_code)
    prefix = (
        f"Translate {input_lang_name} to {output_lang_name}: \n {input_lang_name}: "
    )
    postfix = f"\n {output_lang_name}: "
    inputs = [
        prefix + x["translation"][in_lang_code] + postfix for x in sub_set_half
    ]
    references: List[str] = [
        x["translation"][out_lang_code] for x in sub_set_half
    ]
    return inputs, references


def sub_experiment_half(
    in_lang_code: str,
    out_lang_code: str,
    pipeline: TextGenerationPipeline,
    manager: ExperimentManager,
    sub_set_half: Dataset,
) -> None:
    inputs: List[str]
    references: List[str]
    inputs, references = process_sub_set_half(sub_set_half, in_lang_code,
                                              out_lang_code)
    raw_predictions: List[List[Dict[str, str]]] = pipeline(
        inputs,
        num_beams=1,
        num_return_sequences=1,
        do_sample=True,
        temperature=1.0,
        top_p=1,
        return_full_text=False,
        return_text=True,
        return_tensors=False,
        repetition_penalty=1.2,
        max_new_tokens=10000000000,
        max_length=None,
    )
    predictions: List[str] = [x[0]["generated_text"] for x in raw_predictions]
    metric.add_batch(
        predictions=predictions,
        references=references,
    )
    scores = metric.compute(lang=out_lang_code, )

    # noinspection PyTypeChecker

    manager.log_sub_experiment(scores, in_lang_code, out_lang_code,
                               sub_set_half)


def run_experiment(
    pipe: TextGenerationPipeline,
    sub_sut_names: List[str],
    debug: bool,
    parameters: Dict[str, Any],
) -> None:
    manager = ExperimentManager(debug=debug, parameters=parameters)
    for i, sub_set_name in enumerate(sub_sut_names):
        print(f"Running sub-experiment {i + 1} out of {len(sub_sut_names)}")
        subset_part1: Dataset
        subset_part2: Dataset
        language_code1: str
        language_code2: str
        (
            subset_part1,
            subset_part2,
            language_code1,
            language_code2,
        ) = process_translation_data(sub_set_name, debug)
        sub_experiment_half(
            in_lang_code=language_code1,
            out_lang_code=language_code2,
            pipeline=pipe,
            manager=manager,
            sub_set_half=subset_part1,
        )
        sub_experiment_half(
            in_lang_code=language_code2,
            out_lang_code=language_code1,
            pipeline=pipe,
            manager=manager,
            sub_set_half=subset_part2,
        )
    manager.end_experiment()


def create_hugging_face_pipeline(
        debug: bool, ) -> Tuple[TextGenerationPipeline, Dict[str, Any]]:
    """Creates a translation pipeline from hugging face"""
    parent_folder = Path(__file__).parent
    with open(
            os.path.join(parent_folder, "evaluated_text_generator_dict.json"),
            "r") as json_file:
        evaluated_text_generator_dict = json.load(json_file)
    model_name = "gpt2" if debug else evaluated_text_generator_dict[
        "model_name"]
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipeline = TextGenerationPipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        framework="pt",
        do_sample=True,
        num_beams=1,
        num_return_sequences=1,
        temperature=evaluated_text_generator_dict["temp"],
        top_k=evaluated_text_generator_dict["top_k"],
        top_p=evaluated_text_generator_dict["top_p"],
        repetition_penalty=1.2,
    )
    parameters = {
        "model_name": model_name,
        "temp": evaluated_text_generator_dict["temp"],
        "top_k": evaluated_text_generator_dict["top_k"],
        "top_p": evaluated_text_generator_dict["top_p"],
        "repetition_penalty": 1.2,
        "group_size": 1,
        "end_of_sentence_stop": True,
        "generation_type": GenerationType.RANDOM,
    }
    return pipeline, parameters


def main(debug: bool = __debug__) -> None:
    if debug:
        # send a warning
        warn(
            "Running in debug mode, only a small subset of the data will be used"
        )
    sub_sut_names = get_dataset_config_names(DATASET_NAME)
    if debug:
        sub_sut_names = sub_sut_names[:1]
    evaluated_pipeline, parameters = create_hugging_face_pipeline(debug)
    run_experiment(
        pipe=evaluated_pipeline,
        sub_sut_names=sub_sut_names,
        debug=debug,
        parameters=parameters,
    )


if __name__ == "__main__":
    main()
