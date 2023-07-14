# Generated by CodiumAI

import torch
from torch import FloatTensor, LongTensor, Tensor, long
from transformers import GenerationConfig

from src.grouped_sampling.logits_vec_to_token import LogitVectorToTokenPipeLine

"""
Code Analysis

Main functionalities:
The LogitVectorToTokenPipeLine class is responsible for converting a batch of logit matrices to tokens. It prepares a generation config for the pipeline, initializes a LogitsProcessorList, and applies Softmax normalization of logits if do_sample is True. The batch_to_tokens method takes input_ids, a list_batch of Tensors with shape (output_seq_len, vocab_size) with the logits for every sequence in the batch, and output_length, and returns a Tensor of shape (batch_size, output_seq_len) with the tokens for every sequence in the batch.

Methods:
- prepare_generation_config: prepares a generation config for the LogitVectorToTokenPipeLine.
- __init__: initializes the LogitVectorToTokenPipeLine class and applies Softmax normalization of logits if do_sample is True.
- batch_to_tokens: converts a batch of logit matrices to tokens.

Fields:
- logit_wrapper: a LogitsProcessorList that wraps the logits processors.
- do_sample: a boolean that indicates whether to use sampling or not.
"""


class TestLogitVectorToTokenPipeLine:
    exapmle_input_ids = LongTensor([1, 2, 3]).cuda()
    example_logits_vector = FloatTensor([0.1, 0.2, 0.7]).cuda()
    example_logits_vectors = FloatTensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3],
                                          [0.5, 0.1, 0.4]]).cuda()

    # Tests that batch_to_tokens returns valid token ids with valid input_ids
    # and batch
    def test_batch_to_tokens_valid_input(self):
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], device="cuda")
        batch = torch.tensor(
            [
                [[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.5, 0.2, 0.3]],
                [[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.5, 0.2, 0.3]],
            ],
            device="cuda",
        )
        pipeline = LogitVectorToTokenPipeLine(GenerationConfig())
        output_length = 3
        token_ids = pipeline.logits_to_tokens(input_ids, batch, output_length)
        assert isinstance(token_ids, Tensor)
        assert token_ids.shape == torch.Size([2, output_length])
        assert token_ids.dtype == long
        assert token_ids.is_cuda

    def test_memory_leak(self):
        batch_size = 128
        vocab_size = 2048
        input_ids = torch.randint(0,
                                  vocab_size, (batch_size, 100),
                                  device="cuda")
        output_length = 100
        batch = torch.randn((batch_size, vocab_size, output_length),
                            device="cuda")
        pipeline = LogitVectorToTokenPipeLine(GenerationConfig())
        start_mem = torch.cuda.memory_allocated()
        pipeline.logits_to_tokens(input_ids, batch, output_length)
        end_mem = torch.cuda.memory_allocated()
        assert end_mem == start_mem
