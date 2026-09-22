from __future__ import annotations

from pathlib import Path

try:
    # noinspection PyUnresolvedReferences
    from kaggle_secrets import UserSecretsClient
except ImportError:
    using_kaggle = False
else:  # if we are using kaggle, we need to set the api key
    using_kaggle = True


def get_comet_api_key() -> str:
    """
    Returns the Comet API key from the api key file.
    If the file does not exist, asks the user to enter the key manually and saves it.
    """
    if using_kaggle:
        return UserSecretsClient().get_secret("comet_ml_api_key")
    if Path.cwd() == Path("/content"):
        # if running on colab
        api_key_file = Path("final_project/evaluation/comet_ml_api_key.txt")
    else:
        # if running locally
        api_key_file = Path("comet_ml_api_key.txt")
    try:
        return api_key_file.read_text(encoding="utf-8").strip()
    except OSError:
        api_key = input("Please enter your api_key for comet ml: ")
        api_key_file.write_text(api_key, encoding="utf-8")
        return api_key
