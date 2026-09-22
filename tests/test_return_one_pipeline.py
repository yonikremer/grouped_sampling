from __future__ import annotations

import random
import string

import pytest
import torch
from huggingface_hub.utils import RepositoryNotFoundError
from torch import Tensor, long, no_grad
from transformers import (
    AutoConfig,
    GenerationConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from src.grouped_sampling.logits_vec_to_token import LogitVectorToTokenPipeLine
from src.grouped_sampling.return_one_pipeline import (
    ReturnOnePipeLine,
)


def validate_logits(
    pipeline: ReturnOnePipeLine,
    logits: Tensor,
    output_length: int,
) -> None:
    if not isinstance(logits, Tensor):
        raise TypeError(f"logits should be a Tensor, got {type(logits)}")
    if logits.requires_grad:
        raise ValueError("logits should not require grad")
    if logits.dim() != 3:
        raise ValueError(f"logits should be 3D tensors, got {logits.dim()}D")
    if logits.shape[1] != output_length:
        raise ValueError(
            f"logits should have {output_length} columns, got {logits.shape[1]}"
        )
    if logits.shape[2] != pipeline.tokenizer.vocab_size:
        raise ValueError(
            f"logits should have {pipeline.tokenizer.vocab_size} columns, got {logits.shape[2]}"
        )
    if logits.device != pipeline.device:
        raise ValueError(
            f"logits should be on device {pipeline.device}, got {logits.device}"
        )
    if logits.dtype != pipeline.model.dtype:
        raise ValueError(f"logits should have the same dtype as model, got {logits.dtype} != {pipeline.model.dtype}")


def validate_padded_tokens(pipeline: ReturnOnePipeLine, padded_tokens: Tensor) -> None:
    if padded_tokens.dim() != 2:
        raise ValueError(
            f"tokens should be a 2D tensor, got {padded_tokens.dim()}D tensor"
        )
    if padded_tokens.requires_grad:
        raise ValueError("tokens should not require grad")
    if padded_tokens.shape[1] > pipeline.max_total_len:
        raise ValueError(
            f"tokens should have at most {pipeline.max_total_len} columns, got {padded_tokens.shape[1]}"
        )
    if min(padded_tokens.shape) == 0:
        raise ValueError("tokens should not be empty")
    if padded_tokens.dtype != long:
        raise ValueError(f"tokens should have dtype {long}, got {padded_tokens.dtype}")
    if not all(
        0 <= token < pipeline.tokenizer.vocab_size for token in padded_tokens.flatten()
    ):
        raise ValueError("tokens should be valid token ids")
    if padded_tokens.device != pipeline.device:
        raise ValueError(
            f"tokens should be on device {pipeline.device}, got {padded_tokens.device}"
        )
    if any(
        padded_tokens[i, -1] == padded_tokens[i, 0]
        for i in range(padded_tokens.shape[0])
    ):
        raise ValueError("The first token can never be the padding token")


def validate_output_tokens(
    pipeline: ReturnOnePipeLine,
    output_tokens: Tensor,
    output_length: int,
    batch_size: int,
) -> None:
    if not isinstance(output_tokens, Tensor):
        raise TypeError(f"output_tokens should be a list, got {type(output_tokens)}")
    if output_tokens.dtype != long:
        raise ValueError(
            f"output_tokens should be a tensor of type long. Got {output_tokens.dtype} instead"
        )
    if output_tokens.device != pipeline.device:
        raise ValueError(
            f"output_tokens should be on device {pipeline.device}, got {output_tokens.device} for some tokens"
        )
    if output_tokens.dim() != 2:
        raise ValueError(
            f"output_tokens should be a 2D tensor. Got {output_tokens.dim()}D tensor"
        )
    if output_tokens.shape != (batch_size, output_length):
        raise ValueError(
            f"output_tokens should be of size (batch_size, output_length). Got {output_tokens.shape} instead"
        )
    if not all(
        0 <= token < pipeline.tokenizer.vocab_size for token in output_tokens.flatten()
    ):
        raise ValueError("output_tokens should be valid token ids")


class TestReturnOnePipeLine:
    # Tests that the function returns a list of output strings for a batch of
    # prompts with positive output length
    pipeline = ReturnOnePipeLine("gpt2", max_batch_size=128)

    def test_happy_path(self):
        prompts = ["Hello", "How are you?"]
        output_length = 5
        result = self.pipeline.generate_batch_return_one(prompts, output_length)
        assert isinstance(result, list)
        assert len(result) == len(prompts)
        for output in result:
            assert isinstance(output, str), f"{output} is not a string"
            assert len(output) >= output_length, f"{len(output)} > {output_length}"
            # Each token is at least 1 character long
            output_tokens = self.pipeline.tokenizer.encode(output)
            rebuilt_output = self.pipeline.tokenizer.decode(output_tokens)
            assert rebuilt_output == output, f"{rebuilt_output} != {output}"
            assert (
                len(output_tokens) <= output_length
            ), f"{len(output_tokens)} != {output_length}"

    #  Tests that the function returns an empty list for an empty batch
    def test_empty_prompts(self):

        prompts = []
        output_length = 5
        expected_output = []
        assert (
            self.pipeline.generate_batch_return_one(prompts, output_length)
            == expected_output
        )

    # Tests that the function returns a list of empty strings for a batch of
    # prompts with output length 0
    def test_empty_output_length(self):
        prompts = ["Hello", "How are you?"]
        output_length = 0
        expected_output = ["", ""]
        assert (
            self.pipeline.generate_batch_return_one(prompts, output_length)
            == expected_output
        )

    # Tests that the function returns a list of empty strings for an empty
    # list of prompts
    def test_empty_prompts_list(self):
        prompts = [""]
        output_length = 5
        with pytest.raises(ValueError):
            self.pipeline.generate_batch_return_one(prompts, output_length)

    # Tests that the function raises a ValueError if the prompts contain an
    # empty string
    def test_empty_string_in_prompts(self):
        prompts = ["Hello", ""]
        output_length = 5
        with pytest.raises(ValueError):
            self.pipeline.generate_batch_return_one(prompts, output_length)

    #  Tests that the function raises a ValueError if output_length is negative
    def test_negative_output_length(self):
        prompts = ["Hello", "How are you?"]
        output_length = -1
        with pytest.raises(ValueError):
            self.pipeline.generate_batch_return_one(prompts, output_length)

    # Tests that the function raises a ValueError if output_length is not an
    # integer
    def test_non_integer_output_length(self):
        prompts = ["Hello", "How are you?"]
        output_length = 1.5
        with pytest.raises(TypeError):  # noinspection PyTypeChecker
            self.pipeline.generate_batch_return_one(prompts, output_length)

    @no_grad()
    def test_step_by_step_pipeline(self):
        prompts = ["Hello", "How are you?"]
        output_length = 5
        padded_tokens = self.pipeline.tokenize_and_pad(prompts, output_length)
        validate_padded_tokens(self.pipeline, padded_tokens)
        assert not padded_tokens.requires_grad
        logits = self.pipeline.tokens_batch_to_logit_matrices(
            padded_tokens, output_length
        )
        validate_logits(self.pipeline, logits, output_length)
        output_tokens = self.pipeline.logit_to_token_pipeline.logits_to_tokens_return_one(
            logits=logits,
        )
        validate_output_tokens(self.pipeline, output_tokens, output_length, 2)

    # Tests that the function raises a ValueError if output_length is too
    # large
    def test_huge_output_length(self):
        prompts = ["Hello", "How are you?"]
        output_length = 1000000
        with pytest.raises(ValueError):
            self.pipeline.generate_batch_return_one(prompts, output_length)

    # test that generate_batch works correctly when it gets a string as input
    def test_string_input(self):
        prompt = "Hello"
        output_length = 5
        result = self.pipeline.generate_batch_return_one(prompt, output_length)
        assert isinstance(result, list)
        assert len(result) == 1
        for output in result:
            assert isinstance(output, str), f"{output} is not a string"
            assert len(output) >= output_length, f"{len(output)} > {output_length}"
            # Each token is at least 1 character long
            output_tokens = self.pipeline.tokenizer.encode(output)
            rebuilt_output = self.pipeline.tokenizer.decode(output_tokens)
            assert rebuilt_output == output, f"{rebuilt_output} != {output}"
            assert (
                len(output_tokens) <= output_length
            ), f"{len(output_tokens)} != {output_length}"

    def test_init(self):
        self.validate_pipeline(self.pipeline)

    def test_init_non_existing_model(self):
        with pytest.raises(RepositoryNotFoundError):
            ReturnOnePipeLine("non_existing_model", max_batch_size=128)

    @no_grad()
    def test_init_8bits_model(self):
        pipeline = ReturnOnePipeLine("fxmarty/tiny-llama-fast-tokenizer", max_batch_size=128)
        self.validate_pipeline(pipeline)
        prompts = ["Hello", "How are you?"]
        output_length = 5
        padded_tokens = pipeline.tokenize_and_pad(prompts, output_length)
        validate_padded_tokens(pipeline, padded_tokens)
        logits = pipeline.tokens_batch_to_logit_matrices(
            padded_tokens, output_length
        )
        # assert that the output is on cuda
        assert (
            logits.device.type == "cuda"
        ), f"device is not cuda: {logits.device.type}"
        validate_logits(pipeline, logits, output_length)
        output_tokens = pipeline.logit_to_token_pipeline.logits_to_tokens_return_one(
            logits=logits,
        )
        validate_output_tokens(pipeline, output_tokens, output_length, 2)
        # self.validate_pipeline(pipeline)

    def test_init_model_kwargs(self):
        config = AutoConfig.from_pretrained("gpt2")
        config.output_hidden_states = True
        pipeline = ReturnOnePipeLine("gpt2", model_kwargs={"config": config}, max_batch_size=128)
        self.validate_pipeline(pipeline)

    def test_init_generation_config(self):
        config = GenerationConfig.from_pretrained("gpt2")
        config.top_k = 10
        config.top_p = 0.9
        pipeline = ReturnOnePipeLine("gpt2", generation_config=config, max_batch_size=128)
        self.validate_pipeline(pipeline)

    @staticmethod
    @no_grad()
    def validate_pipeline(pipeline):
        assert pipeline.tokenizer is not None, "tokenizer is None"
        assert isinstance(pipeline.tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)), \
            "tokenizer is not PreTrainedTokenizer or PreTrainedTokenizerFast"
        assert pipeline.model is not None, "model is None"
        assert (
            pipeline.logit_to_token_pipeline is not None
        ), "logit_to_token_pipeline is None"
        assert isinstance(
            pipeline.logit_to_token_pipeline, LogitVectorToTokenPipeLine
        ), "logit_to_token_pipeline is not LogitVectorToTokenPipeLine"
        assert pipeline.max_total_len is not None, "max_total_len is None"
        assert isinstance(pipeline.max_total_len, int), "max_total_len is not int"
        assert pipeline.max_total_len > 0, "max_total_len <= 0"
        assert pipeline.device is not None, "device is None"
        assert isinstance(pipeline.device, torch.device), "device is not torch.device"
        # assert that the device is cuda
        assert (
            pipeline.device.type == "cuda"
        ), f"device is not cuda: {pipeline.device.type}"
        prompt = "Hello"
        pipeline.generate_batch_return_one(prompt, 5)

    # noinspection PyTypeChecker
    def test_init_wrong_types(self):
        with pytest.raises(TypeError):
            ReturnOnePipeLine(1, 128)
        with pytest.raises(TypeError):
            ReturnOnePipeLine("gpt2", 128, model_kwargs=1)
        with pytest.raises(TypeError):
            ReturnOnePipeLine("gpt2", 128, generation_config=1)

    def test_generate_huge_batch(self):
        number_of_prompts = 1024
        prompts = ["Hello"] * number_of_prompts
        output_length = 5
        result = self.pipeline.generate_batch_return_one(prompts, output_length)
        assert isinstance(result, list)
        assert len(result) == number_of_prompts
        for output in result:
            assert isinstance(output, str), f"{output} is not a string"
            assert len(output) >= output_length, f"{len(output)} > {output_length}"
            # Each token is at least 1 character long
            output_tokens = self.pipeline.tokenizer.encode(output)
            rebuilt_output = self.pipeline.tokenizer.decode(output_tokens)
            assert rebuilt_output == output, f"{rebuilt_output} != {output}"
            assert (
                len(output_tokens) <= output_length
            ), f"{len(output_tokens)} != {output_length}"

    @staticmethod
    def random_prompt(length):
        letters = string.ascii_lowercase + string.ascii_uppercase + string.digits + " "
        return "".join(random.choice(letters) for _ in range(length))

    @no_grad()
    def test_gpu_memory_is_freed(self):
        random.seed(0)
        my_pipeline = ReturnOnePipeLine("fxmarty/tiny-llama-fast-tokenizer", max_batch_size=128)
        prompts = [
            self.random_prompt(64) for _ in range(my_pipeline.max_batch_size * 2)
        ]
        my_pipeline.generate_batch_return_one(prompts, 10)
        mid_memory = torch.cuda.memory_allocated()
        prompts = [
            self.random_prompt(64) for _ in range(my_pipeline.max_batch_size * 2)
        ]
        my_pipeline.generate_batch_return_one(prompts, 10)
        end_memory = torch.cuda.memory_allocated()
        assert end_memory == mid_memory, f"{end_memory} != {mid_memory}"
