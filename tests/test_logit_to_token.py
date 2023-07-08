# Generated by CodiumAI

import pytest
import torch
from torch import FloatTensor, LongTensor, Tensor, long
from transformers import GenerationConfig

from src.grouped_sampling.logits_vec_to_token import LogitVectorToTokenPipeLine

"""
Code Analysis

Main functionalities:
The LogitVectorToTokenPipeLine class is a pipeline for converting logit vectors to token ids. It takes in a GenerationConfig object and uses it to create a LogitsProcessorList object, which is used to normalize the logits and convert them to probabilities. The pipeline can handle both single logit vectors and multiple logit vectors in parallel, and can either return the token with the highest probability or sample from the probability distribution.

Methods:
- __init__: initializes the pipeline by creating a copy of the GenerationConfig object, setting renormalize_logits to True, and creating a LogitsProcessorList object with a LogitNormalization processor. It also sets the do_sample field to the value in the GenerationConfig object.
- single_logit_vector_to_token: takes in an input sequence and a single logit vector, and returns the token id with the highest probability or a sampled token id.
- logit_vectors_to_token: takes in an input sequence and multiple logit vectors, and returns the token ids with the highest probability or sampled token ids in parallel.

Fields:
- logit_wrapper: a LogitsProcessorList object that is used to normalize the logits and convert them to probabilities.
- do_sample: a boolean field that determines whether to return the token with the highest probability or sample from the probability distribution.
"""


class TestLogitVectorToTokenPipeLine:
    exapmle_input_ids = LongTensor([1, 2, 3]).cuda()
    example_logits_vector = FloatTensor([0.1, 0.2, 0.7]).cuda()
    example_logits_vectors = FloatTensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3],
                                          [0.5, 0.1, 0.4]]).cuda()

    #  Tests that single_logit_vector_to_token returns a valid token id when given valid input_ids and logits
    def test_single_logit_vector_to_token_valid_input(self):
        generation_config = GenerationConfig()
        pipeline = LogitVectorToTokenPipeLine(generation_config)
        result = pipeline.single_logit_vector_to_token(
            self.exapmle_input_ids, self.example_logits_vector)
        assert result in self.exapmle_input_ids
        assert isinstance(result, torch.Tensor)
        assert result.shape == torch.Size([1])
        assert result.dtype == torch.int64

    #  Tests that logit_vectors_to_token returns valid token ids when given valid input_ids and logit_vectors
    def test_logit_vectors_to_token_valid_input(self):
        generation_config = GenerationConfig()
        pipeline = LogitVectorToTokenPipeLine(generation_config)
        result = pipeline.logit_matrix_to_tokens(self.exapmle_input_ids,
                                                 self.example_logits_vectors)
        assert result.shape == (
            3, ), f"Expected shape (3,), got {result.shape}. Result: {result}"
        assert all([r in [0, 1, 2] for r in result])

    #  Tests that single_logit_vector_to_token raises an error when given empty input_ids
    def test_single_logit_vector_to_token_empty_input_ids(self):
        generation_config = GenerationConfig()
        pipeline = LogitVectorToTokenPipeLine(generation_config)
        input_ids = LongTensor([]).cuda()
        with pytest.raises(ValueError):
            pipeline.single_logit_vector_to_token(input_ids,
                                                  self.example_logits_vector)

    #  Tests that single_logit_vector_to_token raises an error when given empty logits
    def test_single_logit_vector_to_token_empty_logits(self):
        generation_config = GenerationConfig()
        pipeline = LogitVectorToTokenPipeLine(generation_config)
        logits = FloatTensor([]).cuda()
        with pytest.raises(ValueError):
            pipeline.single_logit_vector_to_token(self.exapmle_input_ids,
                                                  logits)

    #  Tests that logit_vectors_to_token raises an error when given empty input_ids
    def test_logit_vectors_to_token_empty_input_ids(self):
        generation_config = GenerationConfig()
        pipeline = LogitVectorToTokenPipeLine(generation_config)
        input_ids = LongTensor([]).cuda()
        with pytest.raises(ValueError):
            pipeline.logit_matrix_to_tokens(input_ids,
                                            self.example_logits_vectors)

    #  Tests that logit_vectors_to_token raises an error when given empty logit_vectors
    def test_logit_vectors_to_token_empty_logit_vectors(self):
        generation_config = GenerationConfig()
        pipeline = LogitVectorToTokenPipeLine(generation_config)
        logit_vectors = FloatTensor([]).cuda()
        with pytest.raises(ValueError):
            pipeline.logit_matrix_to_tokens(self.exapmle_input_ids,
                                            logit_vectors)

    def test_logit_vectors_to_token_valid_input_do_sample(self):
        generation_config = GenerationConfig(do_sample=True)
        pipeline = LogitVectorToTokenPipeLine(generation_config)
        result = pipeline.logit_matrix_to_tokens(self.exapmle_input_ids,
                                                 self.example_logits_vectors)
        assert result.shape == (
            3, ), f"Expected shape (3,), got {result.shape}. Result: {result}"
        assert all([r in [0, 1, 2] for r in result])

    def test_logit_vectors_to_token_valid_input_do_sample_temperature(self):
        generation_config = GenerationConfig(do_sample=True, temperature=0.5)
        pipeline = LogitVectorToTokenPipeLine(generation_config)
        result = pipeline.logit_matrix_to_tokens(self.exapmle_input_ids,
                                                 self.example_logits_vectors)
        assert result.shape == (
            3, ), f"Expected shape (3,), got {result.shape}. Result: {result}"
        assert all([r in [0, 1, 2] for r in result])

    def test_logit_vectors_to_token_valid_input_do_sample_top_k(self):
        generation_config = GenerationConfig(do_sample=True, top_k=2)
        pipeline = LogitVectorToTokenPipeLine(generation_config)
        result = pipeline.logit_matrix_to_tokens(self.exapmle_input_ids,
                                                 self.example_logits_vectors)
        assert result.shape == (
            3, ), f"Expected shape (3,), got {result.shape}. Result: {result}"
        assert all([r in [0, 1, 2] for r in result])

    def test_logit_vectors_to_token_valid_input_do_sample_top_p(self):
        generation_config = GenerationConfig(do_sample=True, top_p=0.9)
        pipeline = LogitVectorToTokenPipeLine(generation_config)
        result = pipeline.logit_matrix_to_tokens(self.exapmle_input_ids,
                                                 self.example_logits_vectors)
        assert result.shape == (
            3, ), f"Expected shape (3,), got {result.shape}. Result: {result}"
        assert all([r in [0, 1, 2] for r in result])

    def test_logit_vectors_to_token_valid_input_do_sample_top_k_top_p(self):
        generation_config = GenerationConfig(do_sample=True,
                                             top_k=2,
                                             top_p=0.9)
        pipeline = LogitVectorToTokenPipeLine(generation_config)
        result = pipeline.logit_matrix_to_tokens(self.exapmle_input_ids,
                                                 self.example_logits_vectors)
        assert result.shape == (
            3, ), f"Expected shape (3,), got {result.shape}. Result: {result}"
        assert all([r in [0, 1, 2] for r in result])

    def test_logit_vectors_to_token_valid_input_do_sample_top_k_top_p_temperature(
            self):
        generation_config = GenerationConfig(do_sample=True,
                                             top_k=2,
                                             top_p=0.9,
                                             temperature=0.5)
        pipeline = LogitVectorToTokenPipeLine(generation_config)
        result = pipeline.logit_matrix_to_tokens(self.exapmle_input_ids,
                                                 self.example_logits_vectors)
        assert result.shape == (
            3, ), f"Expected shape (3,), got {result.shape}. Result: {result}"
        assert all([r in [0, 1, 2] for r in result])

    def test_logit_vectors_to_token_valid_input_do_sample_top_k_top_p_temperature_repetition_penalty(
            self, ):
        generation_config = GenerationConfig(do_sample=True,
                                             top_k=2,
                                             top_p=0.9,
                                             temperature=0.5,
                                             repetition_penalty=0.9)
        pipeline = LogitVectorToTokenPipeLine(generation_config)
        result = pipeline.logit_matrix_to_tokens(self.exapmle_input_ids,
                                                 self.example_logits_vectors)
        assert result.shape == (
            3, ), f"Expected shape (3,), got {result.shape}. Result: {result}"
        assert all([r in [0, 1, 2] for r in result])

    def test_logit_vectors_to_token_valid_input_do_sample_false(self):
        generation_config = GenerationConfig(do_sample=False)
        pipeline = LogitVectorToTokenPipeLine(generation_config)
        result = pipeline.logit_matrix_to_tokens(self.exapmle_input_ids,
                                                 self.example_logits_vectors)
        assert result.shape == (
            3, ), f"Expected shape (3,), got {result.shape}. Result: {result}"

    def test_logit_vectors_to_token_valid_input_do_sample_false_max_length(
            self):
        generation_config = GenerationConfig(do_sample=False, max_length=3)
        pipeline = LogitVectorToTokenPipeLine(generation_config)
        result = pipeline.logit_matrix_to_tokens(self.exapmle_input_ids,
                                                 self.example_logits_vectors)
        assert result.shape == (
            3, ), f"Expected shape (3,), got {result.shape}. Result: {result}"

    #  Tests that logit_matrix_to_tokens returns valid token ids with valid input_ids and logit_vectors
    def test_logit_matrix_to_tokens_valid_input(self):
        input_ids = torch.tensor([1, 2, 3])
        logit_vectors = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])
        pipeline = LogitVectorToTokenPipeLine(GenerationConfig())
        tokens = pipeline.logit_matrix_to_tokens(input_ids, logit_vectors)
        assert isinstance(tokens, Tensor)
        assert tokens.shape == (2, )
        assert tokens[0].dtype == long
        assert tokens[1].dtype == long

    # Tests that batch_to_tokens returns valid token ids with valid input_ids and batch
    def test_batch_to_tokens_valid_input(self):
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        batch = [
            torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.5, 0.2, 0.3]]),
            torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.5, 0.2, 0.3]]),
        ]
        pipeline = LogitVectorToTokenPipeLine(GenerationConfig())
        token_ids = pipeline.batch_to_tokens(input_ids, batch)
        assert isinstance(token_ids, list)
        assert len(token_ids) == 2
        assert all(isinstance(tokens, Tensor) for tokens in token_ids)
        assert all(token_ids[i].shape == (3, ) for i in range(2))
        assert all(token_id.dtype == long for tokens in token_ids
                   for token_id in tokens)

    #  Tests that logit_matrix_to_tokens raises a ValueError with empty logit_vectors
    def test_logit_matrix_to_tokens_empty_logit_vectors(self):
        input_ids = torch.tensor([1, 2, 3])
        logit_vectors = torch.tensor([])
        pipeline = LogitVectorToTokenPipeLine(GenerationConfig())
        with pytest.raises(ValueError):
            pipeline.logit_matrix_to_tokens(input_ids, logit_vectors)

    #  Tests that logit_matrix_to_tokens raises a ValueError with invalid input_ids
    def test_logit_matrix_to_tokens_invalid_input_ids(self):
        pipeline = LogitVectorToTokenPipeLine(GenerationConfig())
        input_ids = torch.tensor([])
        logit_vectors = torch.randn(3, 5)
        with pytest.raises(ValueError):
            pipeline.logit_matrix_to_tokens(input_ids, logit_vectors)

    #  Tests that batch_to_tokens raises a ValueError with invalid input_ids
    def test_batch_to_tokens_invalid_input_ids(self):
        pipeline = LogitVectorToTokenPipeLine(GenerationConfig())
        input_ids = torch.tensor([])
        batch = torch.randn(3, 4, 5)
        with pytest.raises(ValueError):
            pipeline.batch_to_tokens(input_ids, batch)

    def test_batch_to_tokens_invalid_batch(self):
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        batch = [
            [[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.5, 0.2, 0.3]],
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.2]],
        ]
        pipeline = LogitVectorToTokenPipeLine(GenerationConfig())
        with pytest.raises(ValueError):
            pipeline.batch_to_tokens(input_ids, batch)
