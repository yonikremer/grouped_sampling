from transformers import PreTrainedTokenizer, AutoTokenizer, AutoConfig


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


def get_end_of_text_id(tokenizer: PreTrainedTokenizer, config: AutoConfig):
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


def get_tokenizer_name(
        model_name: str,
) -> str:
    """Returns a tokenizer name based on the model name"""
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if hasattr(config, "tokenizer_class") and config.tokenizer_class is not None:
        print(f"Using tokenizer class {config.tokenizer_class} for model {model_name}")
        return config.tokenizer_class
    if model_name == "NovelAI/genji-jp":
        return "EleutherAI/gpt-j-6B"
    if model_name in {"NovelAI/genji-python-6B", "luke-thorburn/suggest-conclusion-full-finetune"}:
        return "EleutherAI/gpt-neo-2.7B"
    if model_name in {"mustapha/distilgpt2-finetuned-wikitext2", "Aleksandar1932/distilgpt2-rock", "AntonClaesson/movie-plot-generator"}:
        return "distilgpt2"
    if model_name in {"mymusise/CPM-Generate-distill"}:
        return "TsinghuaAI/CPM-Generate"
    if model_name.startswith("Aleksandar1932/gpt2") or model_name.startswith("Azaghast/GPT2"):
        return "gpt2"
    return model_name


def get_tokenizer(
        model_name: str,
) -> PreTrainedTokenizer:
    """Returns a tokenizer based on the model name"""
    return AutoTokenizer.from_pretrained(
        get_tokenizer_name(model_name),
        trust_remote_code=True,
    )