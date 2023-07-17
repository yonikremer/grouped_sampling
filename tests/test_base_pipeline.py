# Generated by CodiumAI

import pytest
import torch
from torch import inference_mode

from src.grouped_sampling.base_pipeline import BasePipeLine

"""
Code Analysis

Main functionalities:
The BasePipeLine class is a base class for building pipelines for natural language processing tasks. It provides methods for loading a model and tokenizer from the Hugging Face model hub, tokenizing and padding input prompts, and converting batches of prompts to logits matrices. The class also includes methods for validating input arguments and handling errors.

Methods:
- __init__: Initializes the BasePipeLine object with a model name, whether to load the model in 8-bit mode, model arguments, and a maximum batch size.
- tokens_batch_to_logit_matrices: Given a batch of prompts where each prompt is a sequence of tokens, and an output length, returns the logits matrices of shape (batch_size, output_length, vocab_size) where logits[i] is the logits matrix of the i-th prompt.
- _validate_output_length: Validates the output length argument.
- _validate_prompts: Validates the prompts argument.
- tokenize_and_pad: A helper function that converts a list of strings to a padded tensor of tokens.

Fields:
- max_batch_size: The maximum batch size to use.
- tokenizer: The tokenizer used to tokenize input prompts.
- model: The model used to generate logits matrices from input prompts.
- device: The device used to run the model.
- max_total_len: The maximum total length of input prompts.
"""


class TestBasePipeLine:
    #  Tests that the class can be instantiated with valid arguments
    def test_instantiation(self):
        pipeline = BasePipeLine("gpt2")
        assert pipeline.max_batch_size == 128
        assert pipeline.tokenizer is not None
        assert pipeline.model is not None
        assert pipeline.device == torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        assert pipeline.max_total_len == 1024

    #  Tests that the tokens_batch_to_logit_matrices method returns the expected output
    @inference_mode()
    def test_tokens_batch_to_logit_matrices(self):
        pipeline = BasePipeLine("gpt2")
        prompts = ["Hello world!", "How are you?"]
        tokens = pipeline.tokenize_and_pad(prompts, 10)
        logits, last_non_pad_indices = pipeline.tokens_batch_to_logit_matrices(
            tokens, 10
        )
        assert logits.shape == (2, 10, pipeline.tokenizer.vocab_size)
        assert last_non_pad_indices.tolist() == [1, 3]

    #  Tests that the class raises a TypeError if model_name is not a string
    def test_model_name_type_error(self):
        with pytest.raises(TypeError):
            BasePipeLine(123)  # type: ignore

    #  Tests that the class raises a TypeError if load_in_8bit is not a bool
    def test_load_in_8bit_type_error(self):
        with pytest.raises(TypeError):
            BasePipeLine("gpt2", load_in_8bit=123)  # type: ignore

    #  Tests that the class raises a TypeError if model_kwargs is not a dict or None
    def test_model_kwargs_type_error(self):
        with pytest.raises(TypeError):
            BasePipeLine("gpt2", model_kwargs="invalid")  # type: ignore

    #  Tests that the class raises a TypeError if max_batch_size is not an int
    def test_max_batch_size_type_error(self):
        with pytest.raises(TypeError):
            BasePipeLine("gpt2", max_batch_size="invalid")  # type: ignore

    #  Tests that the class raises a ValueError if max_batch_size is less than 1
    def test_max_batch_size_value_error(self):
        with pytest.raises(ValueError):
            BasePipeLine("gpt2", max_batch_size=0)

    #  Tests that the class raises a ValueError if output_length is less than or equal to 0
    def test_output_length_value_error_1(self):
        pipeline = BasePipeLine("gpt2")
        with pytest.raises(ValueError):
            pipeline._validate_output_length(0)

    #  Tests that the class raises a ValueError if output_length is greater than or equal to max_total_len
    def test_output_length_value_error_2(self):
        pipeline = BasePipeLine("gpt2")
        with pytest.raises(ValueError):
            pipeline._validate_output_length(pipeline.max_total_len)

    #  Tests that the tokens_batch_to_logit_matrices method raises a ValueError if padded_tokens is not a 2D tensor
    def test_tokens_batch_to_logit_matrices_value_error_1(self):
        pipeline = BasePipeLine("gpt2")
        with pytest.raises(ValueError):
            pipeline.tokens_batch_to_logit_matrices(torch.tensor([1, 2, 3]), 10)
