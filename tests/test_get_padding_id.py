from typing import Generator

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizer

from src.grouped_sampling.base_pipeline import get_padding_id


def create_tokenizers() -> Generator[PreTrainedTokenizer, None, None]:
    tokenizer_names = [
        "gpt2",
        "gpt2-medium",
        "facebook/bart-large",
        "facebook/opt-125m",
        "EleutherAI/gpt-j-6B",
        "EleutherAI/gpt-neo-125M",
        "distilgpt2",
        "sberbank-ai/rugpt3large_based_on_gpt2",
    ]
    for tokenizer_name in tokenizer_names:
        yield AutoTokenizer.from_pretrained(tokenizer_name)


@pytest.mark.parametrize("tokenizer", create_tokenizers())
def test_get_padding_id1(tokenizer: PreTrainedTokenizer):
    padding_id = get_padding_id(tokenizer)
    assert isinstance(padding_id, int), f"{padding_id} is not an int"
    tokenizer.mask_token_id = None
    tokenizer.unk_token_id = None
    tokenizer.pad_token_id = None
    with pytest.raises(RuntimeError):
        get_padding_id(tokenizer)


if __name__ == "__main__":
    pytest.main()
