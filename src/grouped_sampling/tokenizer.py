import json
import os
from functools import lru_cache
from typing import Optional, Dict

import requests
from huggingface_hub.utils import validate_repo_id, HFValidationError
from transformers import PreTrainedTokenizer, AutoTokenizer, PretrainedConfig


def get_padding_id(tokenizer: PreTrainedTokenizer):
    if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    if hasattr(tokenizer, "pad_token_ids") and tokenizer.pad_token_ids is not None:
        return tokenizer.pad_token_ids[0]
    if hasattr(tokenizer, "mask_token_id") and tokenizer.mask_token_id is not None:
        return tokenizer.mask_token_id
    if hasattr(tokenizer, "mask_token_ids") and tokenizer.mask_token_ids is not None:
        return tokenizer.mask_token_ids[0]
    if hasattr(tokenizer, "pad_token_type_id") and tokenizer.pad_token_type_id is not None:
        return tokenizer.pad_token_type_id
    if hasattr(tokenizer, "_pad_token") and tokenizer.pad_token is not None:
        return int(tokenizer.pad_token)
    raise RuntimeError(
        "Could not find padding id in a tokenizer with the following attributes: "
        f"{tokenizer.__dict__.keys()}"
        "Please make sure that the tokenizer has the following attributes: "
        "pad_token_id, pad_token_ids, mask_token_id, mask_token_ids"
    )


def get_end_of_text_id(tokenizer: PreTrainedTokenizer, config: PretrainedConfig):
    if tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id
    if hasattr(tokenizer, "eos_token_ids") and tokenizer.eos_token_ids is not None:
        return tokenizer.eos_token_ids[0]
    if hasattr(config, "eos_token_id") and config.eos_token_id is not None:
        return config.eos_token_id
    if hasattr(config, "eos_token_ids") and config.eos_token_ids is not None:
        raise RuntimeError("Could not find end of text id")
    if hasattr(tokenizer, "_eos_token_type_id") and tokenizer.eos_token_type_id is not None:
        return tokenizer.eos_token_type_id
    if hasattr(tokenizer, "_eos_token") and tokenizer.eos_token is not None:
        return int(tokenizer.eos_token)
    raise RuntimeError(
        "Could not find end of text id in a tokenizer with the following attributes: "
        f"{tokenizer.__dict__.keys()}"
        "And in a config with the following attributes: "
        f"{config.__dict__.keys()}"
        "Please make sure that the tokenizer or config has the following attributes: "
        "eos_token_id, eos_token_ids"
    )


def get_model_config(model_name: str) -> Optional[dict]:
    """Returns the config.json file of a model from huggingface as a dictionary"""
    base_url = f"https://huggingface.co/{model_name}/raw/main/config.json"
    try:
        config_json_string: str = requests.get(base_url).text
    except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
        return None
    return json.loads(config_json_string)


def get_tokenizer_config(model_name: str) -> Optional[dict]:
    """
    Returns the tokenizer_config.json file of a tokenizer from huggingface as a dictionary
    """
    base_url = f"https://huggingface.co/{model_name}/raw/main/tokenizer_config.json"
    try:
        config_json_string: str = requests.get(base_url).text
    except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
        return None
    return json.loads(config_json_string)


def is_valid_model_name(model_name: str) -> bool:
    """Returns True if the model name is valid"""
    if "output" in model_name:
        return False
    if model_name in {
        ".",
        "model_medium_300_hf",
        "DialoGPT-small-house",
        "emo-bot",
        "sberbank-ai/rugpt3_medium_gpt2_based",
        "Grossmend/rudialogpt3_medium_based_on_gpt2",
        "workflow"
    }:
        return False
    try:
        validate_repo_id(model_name)
    except HFValidationError:
        return False
    if model_name.count("/") > 1:
        return False
    return True


def get_model_name_from_repo(repo_id: str) -> str:
    try:
        model_config = get_model_config(repo_id)
    except Exception:
        return repo_id
    repo_id_no_org = repo_id.split("/")[-1]
    if model_config is not None and "_name_or_path" in model_config.keys():
        config_model_name = model_config["_name_or_path"]
        if is_valid_model_name(config_model_name) and config_model_name != repo_id_no_org:
            return config_model_name
    return repo_id


def get_tokenizer_name_from_repo(repo_id: str) -> str:
    try:
        tokenizer_config = get_tokenizer_config(repo_id)
    except Exception:
        return repo_id
    repo_id_no_org = repo_id.split("/")[-1]
    if tokenizer_config is not None and "name_or_path" in tokenizer_config.keys():
        config_model_name = tokenizer_config["name_or_path"]
        if is_valid_model_name(config_model_name) and config_model_name != repo_id_no_org:
            return config_model_name
    return repo_id


@lru_cache(maxsize=1)
def get_special_cases() -> Dict[str, str]:
    special_cases_file = os.path.join(os.path.dirname(__file__), "model_to_tokenizer.json")
    with open(special_cases_file) as f:
        return json.loads(f.read())


# noinspection PyProtectedMember
def get_tokenizer_name(
        model_name: str,
) -> str:
    """Returns a tokenizer name based on the model name"""
    special_cases = get_special_cases()
    if model_name in special_cases.keys():
        return special_cases[model_name]
    if model_name.startswith("Aleksandar1932/gpt2") \
            or model_name.startswith("Azaghast/GPT2") \
            or model_name.startswith("SteveC/sdc_bot") \
            or model_name.startswith("benjamin/gpt2-wechsel-"):
        return "gpt2"
    tokenizer_name_from_repo = get_tokenizer_name_from_repo(model_name)
    if tokenizer_name_from_repo != model_name:
        return get_tokenizer_name(tokenizer_name_from_repo)
    model_name_from_repo = get_model_name_from_repo(model_name)
    if model_name_from_repo != model_name:
        return get_tokenizer_name(model_name_from_repo)
    return model_name


def get_tokenizer(
        model_name: str,
) -> PreTrainedTokenizer:
    """Returns a tokenizer based on the model name"""
    tokenizer_name = get_tokenizer_name(model_name)
    raw_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
    )
    if not hasattr(raw_tokenizer, "pad_token_id") or raw_tokenizer.pad_token_id is None:
        raw_tokenizer.pad_token_id = get_padding_id(raw_tokenizer)
    return raw_tokenizer
