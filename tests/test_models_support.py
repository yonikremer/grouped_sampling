from typing import Set
import pip

import pytest

from transformers import AutoTokenizer, AutoConfig

from src.grouped_sampling import GroupedSamplingPipeLine
from src.grouped_sampling.base_pipeline import get_padding_id, get_end_of_text_id
from src.grouped_sampling.support_check import get_supported_model_names


def get_tested_model_names() -> Set[str]:
    return {
        "gpt2",
        "Gustavosta/MagicPrompt-Stable-Diffusion",
        "EleutherAI/gpt-neo-125M",
        "distilgpt2",
        "bigscience/bloom-560m",
        "bigscience/bloomz-560m",
        "microsoft/CodeGPT-small-py",
        "microsoft/DialoGPT-small",
    }


@pytest.mark.parametrize("model_name", get_tested_model_names())
def test_pipeline_creation(model_name: str):
    pipeline: GroupedSamplingPipeLine = GroupedSamplingPipeLine(
        model_name=model_name, group_size=5
    )
    assert pipeline is not None
    assert pipeline.model_name == model_name
    assert pipeline.wrapped_model.group_size == 5
    assert pipeline.wrapped_model.end_of_sentence_stop is True


def get_tokenizer_name(
        model_name: str,
) -> str:
    if model_name == "NovelAI/genji-jp":
        return "EleutherAI/gpt-j-6B"
    if model_name == "NovelAI/genji-python-6B":
        return "EleutherAI/gpt-neo-2.7B"
    return model_name


def get_dependency_name(error: ImportError) -> str:
    """Gets the error and returns the name of the missing dependency.
    The error format is usually:
     ModuleNotFoundError/ImportError: You need to install dependency_name to use x or x"""
    string_error = str(error)
    return string_error.split("You need to install ")[1].split(" to use")[0]


@pytest.mark.parametrize(
    "model_name",
    generate_supported_model_names(
        min_number_of_downloads=0,
        min_number_of_likes=0,
    )
)
def test_supported_tokenizers(model_name: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            get_tokenizer_name(model_name),
            trust_remote_code=True,
        )
        assert tokenizer is not None
    except ImportError as e:
        # in case of a missing dependency install it
        missing_dependency = get_dependency_name(e)
        pip.main(["install", missing_dependency])
        return test_supported_tokenizers(model_name)
    assert tokenizer is not None
    padding_id = get_padding_id(tokenizer)
    assert padding_id is not None
    assert isinstance(padding_id, int)
    assert padding_id >= 0
    try:
        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
    except KeyError:
        config = None
    end_of_sentence_id = get_end_of_text_id(tokenizer, config)
    assert end_of_sentence_id is not None
    assert isinstance(end_of_sentence_id, int)
    assert end_of_sentence_id >= 0


if __name__ == "__main__":
    pytest.main()
