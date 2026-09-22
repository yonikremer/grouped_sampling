from __future__ import annotations

import pytest
from huggingface_hub.utils import RepositoryNotFoundError
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from src.grouped_sampling.tokenizer import get_tokenizer


class TestGetTokenizer:
    #  Tests that a tokenizer is returned when given a valid model name
    def test_valid_model_name(self):
        tokenizer = get_tokenizer("bert-base-uncased")
        assert isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast))

    #  Tests that padding_side is set to 'right' when creating the tokenizer
    def test_padding_side(self):
        tokenizer = get_tokenizer("bert-base-uncased")
        assert tokenizer.padding_side == "right"

    # Tests that RepositoryNotFoundError is raised when model is not found in
    # huggingfacehub or local cache
    def test_repo_not_found(self):
        with pytest.raises(RepositoryNotFoundError):
            get_tokenizer("nonexistent-model")

    # Tests that llama tokenizers are supported
    def test_llama_tokenizer(self):
        tokenizer = get_tokenizer("fxmarty/tiny-llama-fast-tokenizer")
        assert isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast))
