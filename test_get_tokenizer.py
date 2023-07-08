
# Generated by CodiumAI

import pytest
from huggingface_hub.utils import RepositoryNotFoundError
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from src.grouped_sampling.tokenizer import get_tokenizer

"""
Code Analysis

Objective:
The objective of the 'get_tokenizer' function is to return a tokenizer based on the provided model name. The tokenizer can be obtained either from the Hugging Face Hub or from the local Hugging Face cache. The function can return either a PreTrainedTokenizer or a PreTrainedTokenizerFast.

Inputs:
- model_name (str): The name of the model for which the tokenizer is required.

Flow:
- The function first calls the 'get_tokenizer_name' function to get the name of the tokenizer based on the model name.
- It then tries to obtain the tokenizer from the Hugging Face Hub using the obtained tokenizer name.
- If the tokenizer is not found in the Hugging Face Hub, the function raises a 'RepositoryNotFoundError' exception.
- If the tokenizer is found, the function checks if it has a 'pad_token_id' attribute. If not, it calls the 'get_padding_id' function to obtain the padding id for the tokenizer.
- The function then returns the obtained tokenizer.

Outputs:
- raw_tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): The obtained tokenizer.

Additional aspects:
- The function uses the 'get_tokenizer_name' function to obtain the name of the tokenizer based on the model name.
- The function raises a 'RepositoryNotFoundError' exception if the tokenizer is not found in the Hugging Face Hub.
- The function calls the 'get_padding_id' function to obtain the padding id for the tokenizer if it does not have a 'pad_token_id' attribute.
- The function sets the 'pad_token_id' attribute of the tokenizer to the obtained padding id.
"""
class TestGetTokenizer:
    #  Tests that a tokenizer is returned when given a valid model name
    def test_valid_model_name(self):
        tokenizer = get_tokenizer('bert-base-uncased')
        assert isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast))

    #  Tests that padding_side is set to 'right' when creating the tokenizer
    def test_padding_side(self):
        tokenizer = get_tokenizer('bert-base-uncased')
        assert tokenizer.padding_side == 'right'

    #  Tests that RepositoryNotFoundError is raised when model is not found in huggingfacehub or local cache
    def test_repo_not_found(self):
        with pytest.raises(RepositoryNotFoundError):
            get_tokenizer('nonexistent-model')
