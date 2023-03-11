from transformers import AutoConfig, PretrainedConfig

from .llama.configuration_llama import LLaMAConfig


def get_config(
    model_name: str,
) -> PretrainedConfig:
    """Returns a config based on the model name"""
    try:
        return AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
    except KeyError as e:
        if "llama" in model_name.lower():
            return LLaMAConfig.from_pretrained(
                model_name,
            )
        raise e
