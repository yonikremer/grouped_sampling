# Generated by CodiumAI

import torch
from torch import LongTensor, FloatTensor, long, Tensor
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
    example_logits_vectors = FloatTensor(
        [[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.5, 0.1, 0.4]]
    ).cuda()

    #  Tests that single_logit_vector_to_token returns a valid token id when given valid input_ids and logits
    def test_single_logit_vector_to_token_valid_input(self):
        generation_config = GenerationConfig()
        pipeline = LogitVectorToTokenPipeLine(generation_config)
        result = pipeline.single_logit_vector_to_token(
            self.exapmle_input_ids, self.example_logits_vector
        )
        assert result in self.exapmle_input_ids
        assert isinstance(result, torch.Tensor)
        assert result.shape == torch.Size([1])
        assert result.dtype == torch.int64

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
        assert all(token_ids[i].shape == (3,) for i in range(2))
        assert all(
            token_id.dtype == long for tokens in token_ids for token_id in tokens
        )
