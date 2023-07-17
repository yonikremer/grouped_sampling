import json
import os
from pathlib import Path
from typing import Callable, Tuple, Dict, Any
from warnings import warn

from datasets import Dataset, load_dataset
from transformers import GenerationConfig

from src.grouped_sampling import BatchPipeLine


STAT_NAME_TO_FUNC: Tuple[Tuple[str, Callable], ...] = (
    ("mean", lambda x: x.mean()),
    ("median", lambda x: x.median()),
    ("25_percentile", lambda x: x.quantile(0.25)),
    ("75_percentile", lambda x: x.quantile(0.75)),
    ("min", lambda x: x.min()),
    ("max", lambda x: x.max()),
    ("standard_deviation", lambda x: x.std()),
)

BERT_SCORES = ("BERT_f1", "BERT_precision", "BERT_recall")
WORKSPACE = USERNAME = "yonikremer"
DATASET_NAME = "ted_talks_iwslt"


def lang_code_to_name(language_code: str) -> str:
    """
    Converts language's iso_639_1 code to language name
    Raises KeyError if language code is not one of the languages in the ted_talks_iwslt dataset
    """
    iso_639_1 = {
        "de": "German",
        "en": "English",
        "nl": "Dutch",
        "eu": "Basque",
        "ja": "Japanese",
        "ca": "Catalan",
        "fr-ca": "Canadian French",
        "hi": "Hindi",
    }
    return iso_639_1[language_code]


def get_project_name(debug: bool = __debug__) -> str:
    if debug:
        warn("RUNNING IN DEBUG MODE")
    return "grouped-sampling-debug" if debug else "grouped-sampling-evaluation"


def process_translation_data(
    sub_set_name: str, debug: bool
) -> Tuple[Dataset, Dataset, str, str]:
    spited_sub_set_name = sub_set_name.split("_")
    language_code1, language_code2 = spited_sub_set_name[:2]
    if debug:
        sub_set: Dataset = load_dataset(DATASET_NAME, sub_set_name, split="train[:2]")
    else:
        sub_set: Dataset = load_dataset(DATASET_NAME, sub_set_name, split="train")

    def rename_keys(
        x: Dict[str, Any], input_lang_name: str, output_lang_name: str
    ) -> Dict[str, str]:
        translation: Dict[str, str] = x["translation"]
        return {
            input_lang_name: translation[input_lang_name],
            output_lang_name: translation[output_lang_name],
        }

    subset_part1: Dataset = sub_set.map(
        rename_keys,
        fn_kwargs={
            "input_lang_name": language_code1,
            "output_lang_name": language_code2,
        },
    )
    subset_part2: Dataset = sub_set.map(
        rename_keys,
        fn_kwargs={
            "input_lang_name": language_code2,
            "output_lang_name": language_code1,
        },
    )
    return subset_part1, subset_part2, language_code1, language_code2


def create_pipeline(max_batch_size: int) -> BatchPipeLine:
    """Creates a text pipeline from the experiment_arguments.json file"""
    experiment_parameters = get_experiment_parameters()
    model_name = experiment_parameters.pop("model_name")
    generation_config = GenerationConfig.from_pretrained(model_name)
    for key, value in experiment_parameters.items():
        setattr(generation_config, key, value)
    pipeline = BatchPipeLine(
        model_name=model_name,
        load_in_8bit=True,
        generation_config=generation_config,
        max_batch_size=max_batch_size,
    )
    return pipeline


def get_experiment_parameters():
    parent_folder = Path(__file__).parent
    with open(
        os.path.join(parent_folder, "experiment_arguments.json"), "r"
    ) as json_file:
        experiment_parameters = json.load(json_file)
    return experiment_parameters


def disable_transformers_progress_bar():
    from transformers.utils.logging import disable_progress_bar

    disable_progress_bar()


def disable_datasets_progress_bar():
    from datasets.utils.logging import disable_progress_bar

    disable_progress_bar()


def disable_progress_bars():
    disable_transformers_progress_bar()
    disable_datasets_progress_bar()
