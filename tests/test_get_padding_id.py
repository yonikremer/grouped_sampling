from typing import Generator

import pytest
from transformers import PreTrainedTokenizer, AutoTokenizer

from src.grouped_sampling.base_pipeline import get_padding_id


def create_tokenizers() -> Generator[PreTrainedTokenizer, None, None]:

    yield AutoTokenizer.from_pretrained("gpt2-medium")
    yield AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    yield AutoTokenizer.from_pretrained("t5-small")
    yield AutoTokenizer.from_pretrained("sshleifer/tiny-mbart")


@pytest.mark.parametrize("tokenizer", create_tokenizers())
def test_get_padding_id1(tokenizer: PreTrainedTokenizer):
    padding_id = get_padding_id(tokenizer)
    assert isinstance(padding_id, int), f"{padding_id} is not an int"
