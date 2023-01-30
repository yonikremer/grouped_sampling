import json
import os
from pathlib import Path
from typing import Callable, Tuple, Dict, Any

from datasets import Dataset, load_dataset
from pandas import Series

from src.grouped_sampling import GroupedGenerationPipeLine, GroupedSamplingPipeLine

try:
    # noinspection PyUnresolvedReferences
    from kaggle_secrets import UserSecretsClient
except ImportError:
    using_kaggle = False
else:  # if we are using kaggle, we need to set the api key
    using_kaggle = True


STAT_NAME_TO_FUNC: Tuple[Tuple[str, Callable[[Series], float]]] = (
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
    """Converts language's ISO_639_1 code to language name
    Raises KeyError if language code is not one of the languages in the ted_talks_iwslt dataset"""
    ISO_639_1 = {
        "de": "German",
        "en": "English",
        "nl": "Dutch",
        "eu": "Basque",
        "ja": "Japanese",
        "ca": "Catalan",
        "fr-ca": "Canadian French",
        "hi": "Hindi",
    }
    return ISO_639_1[language_code]


def get_comet_api_key() -> str:
    """Returns the Comet API key from the file "final_project/evaluate/comet_api_key.txt"
    if this file does not exist, asks the user to enter the key manually and saves it to the file"""
    if using_kaggle:
        return UserSecretsClient().get_secret("comet_ml_api_key")
    if os.getcwd() == "/content":
        # if running on colab
        api_key_file = "final_project/evaluation/comet_ml_api_key.txt"
    else:
        # if running locally
        api_key_file = "comet_ml_api_key.txt"
    if os.path.exists(api_key_file):
        with open(api_key_file, "r") as f:
            return f.read().strip()
    api_key = input("Please enter your api_key for comet ml: ")
    with open(api_key_file, "w") as f:
        f.write(api_key)
    return api_key


def get_project_name(debug: bool = __debug__) -> str:
    if debug:
        print("WARING: RUNNING ON DEBUG MODE")
    return "grouped-sampling-debug" if debug else "grouped-sampling-evaluation"


def process_translation_data(sub_set_name: str, debug: bool) -> Tuple[Dataset, Dataset, str, str]:
    spited_sub_set_name = sub_set_name.split("_")
    language_code1, language_code2 = spited_sub_set_name[:2]
    if debug:
        sub_set: Dataset = load_dataset(DATASET_NAME, sub_set_name, split="train[:2]")
    else:
        sub_set: Dataset = load_dataset(DATASET_NAME, sub_set_name, split="train")

    def rename_keys(x: Dict[str, Any], input_lang_name: str, output_lang_name: str) -> Dict[str, str]:
        translation: Dict[str, str] = x["translation"]
        return {input_lang_name: translation[input_lang_name], output_lang_name: translation[output_lang_name]}

    subset_part1: Dataset = sub_set.map(
        rename_keys,
        fn_kwargs={"input_lang_name": language_code1, "output_lang_name": language_code2}
    )
    subset_part2: Dataset = sub_set.map(
        rename_keys,
        fn_kwargs={"input_lang_name": language_code2, "output_lang_name": language_code1}
    )
    return subset_part1, subset_part2, language_code1, language_code2


def create_pipeline() -> GroupedGenerationPipeLine:
    """Creates a text pipeline from the evaluated_text_generator_dict.json file"""
    parent_folder = Path(__file__).parent
    with open(os.path.join(parent_folder, "evaluated_text_generator_dict.json"), "r") as json_file:
        evaluated_text_generator_dict = json.load(json_file)
    pipeline = GroupedSamplingPipeLine(**evaluated_text_generator_dict)
    return pipeline


def disable_transformers_progress_bar():
    from transformers.utils.logging import disable_progress_bar

    disable_progress_bar()


def disable_datasets_progress_bar():
    from datasets.utils.logging import disable_progress_bar

    disable_progress_bar()


def disable_progress_bars():
    disable_transformers_progress_bar()
    disable_datasets_progress_bar()
