import os
from typing import Callable, Tuple

from pandas import Series

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
