import os
import random
import string

import pytest
import torch
from huggingface_hub.utils import RepositoryNotFoundError
from torch import Tensor, float32, inference_mode, long
from transformers import (
    AutoConfig,
    GenerationConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from fix_bitsandbytes import fix_ld_library_path
from src.grouped_sampling.batch_pipeline import BatchPipeLine
from src.grouped_sampling.logits_vec_to_token import LogitVectorToTokenPipeLine

# Generated by CodiumAI
"""
Code Analysis

Main functionalities:
BatchEndToEndSingleSequencePipeLine is a class that provides an end-to-end pipeline for generating sequences from a given prompt using a pre-trained language model. It takes a list of prompts and an output length as input, and returns a list of generated sequences of the specified length. The class uses a tokenizer and a language model from the Hugging Face Transformers library, and provides methods for converting tokens to logits and logits to tokens.

Methods:
- tokens_batch_to_logit_matrices: Given a batch of prompts where each prompt is a sequence of tokens, and an output_length, returns the logits matrices of shape (batch_size, output_length, vocab_size) where logits[i] is the logits matrix of the i-th prompt.
- __call__: Given a batch of prompts and output length, generates a list of output strings.
- get_padding_id: Returns the padding id of a given tokenizer.
- get_tokenizer: Returns a tokenizer based on the model name.
- get_model: Load a model from the huggingface model hub, and compile it for faster inference.

Fields:
- tokenizer: A tokenizer object from the Hugging Face Transformers library.
- model: A language model object from the Hugging Face Transformers library.
- device: The device on which the model is loaded.
- max_total_len: The maximum length of the input sequence that the model can handle.
- logit_to_token_pipeline: An object of the LogitVectorToTokenPipeLine class, which provides methods for converting logits to tokens.
"""


def validate_logits(
    pipeleine: BatchPipeLine,
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
    if logits.shape[2] != pipeleine.tokenizer.vocab_size:
        raise ValueError(
            f"logits should have {pipeleine.tokenizer.vocab_size} columns, got {logits.shape[2]}"
        )
    if logits.device != pipeleine.device:
        raise ValueError(
            f"logits should be on device {pipeleine.device}, got {logits.device}"
        )
    if logits.dtype != float32:
        raise ValueError(
            f"logits should have dtype {float32}, got {logits[0].dtype}")


def validate_padded_tokens(pipeline: BatchPipeLine, padded_tokens: Tensor) -> None:
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
        raise ValueError(
            f"tokens should have dtype {long}, got {padded_tokens.dtype}")
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
    pipeline: BatchPipeLine,
    output_tokens: Tensor,
    output_length: int,
    batch_size: int,
) -> None:
    if not isinstance(output_tokens, Tensor):
        raise TypeError(
            f"output_tokens should be a list, got {type(output_tokens)}")
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


class TestBatchPipeLine:
    @staticmethod
    def setup_method():
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        fix_ld_library_path()
        # noinspection PyUnresolvedReferences
        torch._dynamo.config.verbose = True

    #  Tests that the function returns a list of output strings for a batch of prompts with positive output length
    def test_happy_path(self):
        pipeline = BatchPipeLine("gpt2")
        prompts = ["Hello", "How are you?"]
        output_length = 5
        result = pipeline.genearte_batch(prompts, output_length)
        assert isinstance(result, list)
        assert len(result) == len(prompts)
        for output in result:
            assert isinstance(output, str), f"{output} is not a string"
            assert len(
                output) >= output_length, f"{len(output)} > {output_length}"
            # Each token is at least 1 character long
            output_tokens = pipeline.tokenizer.encode(output)
            rebuilt_output = pipeline.tokenizer.decode(output_tokens)
            assert rebuilt_output == output, f"{rebuilt_output} != {output}"
            assert (
                len(output_tokens) <= output_length
            ), f"{len(output_tokens)} != {output_length}"

    #  Tests that the function returns an empty list for an empty batch
    def test_empty_prompts(self):
        pipeline = BatchPipeLine("gpt2")
        prompts = []
        output_length = 5
        expected_output = []
        assert pipeline.genearte_batch(
            prompts, output_length) == expected_output

    #  Tests that the function returns a list of empty strings for a batch of prompts with output length 0
    def test_empty_output_length(self):
        pipeline = BatchPipeLine("gpt2")
        prompts = ["Hello", "How are you?"]
        output_length = 0
        expected_output = ["", ""]
        assert pipeline.genearte_batch(
            prompts, output_length) == expected_output

    #  Tests that the function returns a list of empty strings for an empty list of prompts
    def test_empty_prompts_list(self):
        pipeline = BatchPipeLine("gpt2")
        prompts = [""]
        output_length = 5
        with pytest.raises(ValueError):
            pipeline.genearte_batch(prompts, output_length)

    #  Tests that the function raises a ValueError if the prompts contain an empty string
    def test_empty_string_in_prompts(self):
        pipeline = BatchPipeLine("gpt2")
        prompts = ["Hello", ""]
        output_length = 5
        with pytest.raises(ValueError):
            pipeline.genearte_batch(prompts, output_length)

    #  Tests that the function raises a ValueError if output_length is negative
    def test_negative_output_length(self):
        pipeline = BatchPipeLine("gpt2")
        prompts = ["Hello", "How are you?"]
        output_length = -1
        with pytest.raises(ValueError):
            pipeline.genearte_batch(prompts, output_length)

    #  Tests that the function raises a ValueError if output_length is not an integer
    def test_non_integer_output_length(self):
        pipeline = BatchPipeLine("gpt2")
        prompts = ["Hello", "How are you?"]
        output_length = 1.5
        with pytest.raises(TypeError):  # noinspection PyTypeChecker
            pipeline.genearte_batch(prompts, output_length)

    @inference_mode()
    def test_step_by_step_pipeline(self):
        pipeline = BatchPipeLine("gpt2")
        prompts = ["Hello", "How are you?"]
        output_length = 5
        padded_tokens = pipeline.tokenize_and_pad(prompts, output_length)
        validate_padded_tokens(pipeline, padded_tokens)
        assert not padded_tokens.requires_grad
        logits = pipeline.tokens_batch_to_logit_matrices(
            padded_tokens, output_length)
        validate_logits(pipeline, logits, output_length)
        output_tokens = pipeline.logit_to_token_pipeline.batch_to_tokens(
            input_ids=padded_tokens,
            batch=logits,
            output_length=output_length,
        )
        validate_output_tokens(pipeline, output_tokens, output_length, 2)

    #  Tests that the function raises a ValueError if output_length is too large
    def test_huge_output_length(self):
        pipeline = BatchPipeLine("gpt2")
        prompts = ["Hello", "How are you?"]
        output_length = 1000000
        with pytest.raises(ValueError):
            pipeline.genearte_batch(prompts, output_length)

    # test that genearte_batch works correctrly when it gets a string as input
    def test_string_input(self):
        pipeline = BatchPipeLine("gpt2")
        prompt = "Hello"
        output_length = 5
        result = pipeline.genearte_batch(prompt, output_length)
        assert isinstance(result, list)
        assert len(result) == 1
        for output in result:
            assert isinstance(output, str), f"{output} is not a string"
            assert len(
                output) >= output_length, f"{len(output)} > {output_length}"
            # Each token is at least 1 character long
            output_tokens = pipeline.tokenizer.encode(output)
            rebuilt_output = pipeline.tokenizer.decode(output_tokens)
            assert rebuilt_output == output, f"{rebuilt_output} != {output}"
            assert (
                len(output_tokens) <= output_length
            ), f"{len(output_tokens)} != {output_length}"

    def test_init(self):
        pipeline = BatchPipeLine("gpt2")
        self.validate_pipeline(pipeline)

    def test_init_non_existing_model(self):
        with pytest.raises(RepositoryNotFoundError):
            BatchPipeLine("non_existing_model")

    @inference_mode()
    def test_init_8bits_model(self):
        import bitsandbytes

        assert (
            bitsandbytes.COMPILED_WITH_CUDA
        ), "bitsandbytes was not compiled with CUDA"
        pipeline = BatchPipeLine("fxmarty/tiny-llama-fast-tokenizer")
        prompts = ["Hello", "How are you?"]
        output_length = 5
        padded_tokens = pipeline.tokenize_and_pad(prompts, output_length)
        validate_padded_tokens(pipeline, padded_tokens)
        logits = pipeline.tokens_batch_to_logit_matrices(
            padded_tokens, output_length)
        validate_logits(pipeline, logits, output_length)
        output_tokens = pipeline.logit_to_token_pipeline.batch_to_tokens(
            input_ids=padded_tokens,
            batch=logits,
            output_length=output_length,
        )
        validate_output_tokens(pipeline, output_tokens, output_length, 2)
        # self.validate_pipeline(pipeline)

    def test_init_model_kwargs(self):
        config = AutoConfig.from_pretrained("gpt2")
        config.output_hidden_states = True
        pipeline = BatchPipeLine("gpt2", model_kwargs={"config": config})
        self.validate_pipeline(pipeline)

    def test_init_generation_config(self):
        config = GenerationConfig.from_pretrained("gpt2")
        config.top_k = 10
        config.top_p = 0.9
        pipeline = BatchPipeLine("gpt2", generation_config=config)
        self.validate_pipeline(pipeline)

    @staticmethod
    @inference_mode()
    def validate_pipeline(pipeline):
        assert pipeline.tokenizer is not None, "tokenizer is None"
        assert isinstance(pipeline.tokenizer, PreTrainedTokenizer) or isinstance(
            pipeline.tokenizer, PreTrainedTokenizerFast
        ), "tokenizer is not PreTrainedTokenizer or PreTrainedTokenizerFast"
        assert pipeline.model is not None, "model is None"
        assert (
            pipeline.logit_to_token_pipeline is not None
        ), "logit_to_token_pipeline is None"
        assert isinstance(
            pipeline.logit_to_token_pipeline, LogitVectorToTokenPipeLine
        ), "logit_to_token_pipeline is not LogitVectorToTokenPipeLine"
        assert pipeline.max_total_len is not None, "max_total_len is None"
        assert isinstance(pipeline.max_total_len,
                          int), "max_total_len is not int"
        assert pipeline.max_total_len > 0, "max_total_len <= 0"
        assert pipeline.device is not None, "device is None"
        assert isinstance(
            pipeline.device, torch.device), "device is not torch.device"
        # assert that the device is cuda
        assert (
            pipeline.device.type == "cuda"
        ), f"device is not cuda: {pipeline.device.type}"
        prompt = "Hello"
        pipeline.genearte_batch(prompt, 5)

    # noinspection PyTypeChecker
    def test_init_wrong_types(self):
        with pytest.raises(TypeError):
            BatchPipeLine(1)
        with pytest.raises(TypeError):
            BatchPipeLine("gpt2", model_kwargs=1)
        with pytest.raises(TypeError):
            BatchPipeLine("gpt2", generation_config=1)

    def test_generrate_huge_batch(self):
        pipeline = BatchPipeLine("gpt2", max_batch_size=128)
        number_of_prompts = 1024
        prompts = ["Hello"] * number_of_prompts
        output_length = 5
        result = pipeline.genearte_batch(prompts, output_length)
        assert isinstance(result, list)
        assert len(result) == number_of_prompts
        for output in result:
            assert isinstance(output, str), f"{output} is not a string"
            assert len(
                output) >= output_length, f"{len(output)} > {output_length}"
            # Each token is at least 1 character long
            output_tokens = pipeline.tokenizer.encode(output)
            rebuilt_output = pipeline.tokenizer.decode(output_tokens)
            assert rebuilt_output == output, f"{rebuilt_output} != {output}"
            assert (
                len(output_tokens) <= output_length
            ), f"{len(output_tokens)} != {output_length}"

    @staticmethod
    def random_prompt(length):
        letters = string.ascii_lowercase + string.ascii_uppercase + string.digits + " "
        return "".join(random.choice(letters) for _ in range(length))

    @inference_mode()
    def test_gpu_memory_is_freed(self):
        random.seed(0)
        my_pipeline = BatchPipeLine("fxmarty/tiny-llama-fast-tokenizer")
        prompts = [
            self.random_prompt(64) for _ in range(my_pipeline.max_batch_size * 2)
        ]
        my_pipeline.genearte_batch(prompts, 10)
        mid_memory = torch.cuda.memory_allocated()
        prompts = [
            self.random_prompt(64) for _ in range(my_pipeline.max_batch_size * 2)
        ]
        my_pipeline.genearte_batch(prompts, 10)
        end_memory = torch.cuda.memory_allocated()
        assert end_memory == mid_memory, f"{end_memory} != {mid_memory}"
