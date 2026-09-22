from __future__ import annotations

# pylint: disable=protected-access  # tests probe internal validators on purpose
import pytest
import torch
from torch import inference_mode

from src.grouped_sampling.base_pipeline import BasePipeLine


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
        logits = pipeline.tokens_batch_to_logit_matrices(
            tokens, 10
        )
        assert logits.shape == (2, 10, pipeline.tokenizer.vocab_size)

    #  Tests that the class raises a TypeError if model_name is not a string
    def test_model_name_type_error(self):
        with pytest.raises(TypeError):
            BasePipeLine(123)  # type: ignore

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
