from unittest.mock import patch

import pytest
import torch
from torch import FloatTensor, LongTensor, Tensor
from transformers import GenerationConfig

from src.grouped_sampling.logits_vec_to_token import LogitVectorToTokenPipeLine


class TestLogitVectorToTokenPipeLine:
    example_input_ids = LongTensor([1, 2, 3]).cuda()
    example_logits_vector = FloatTensor([0.1, 0.2, 0.7]).cuda()
    example_logits_vectors = FloatTensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3],
                                          [0.5, 0.1, 0.4]]).cuda()

    # Tests that batch_to_tokens returns valid token ids with valid input_ids
    # and batch
    def test_batch_to_tokens_valid_input(self):
        batch = torch.tensor(
            [
                [[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.5, 0.2, 0.3]],
                [[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.5, 0.2, 0.3]],
            ],
            device="cuda",
        )
        pipeline = LogitVectorToTokenPipeLine(GenerationConfig())
        output_length = 3
        token_ids = pipeline.logits_to_tokens_return_one(batch)
        assert isinstance(token_ids, Tensor)
        assert token_ids.shape == torch.Size([2, output_length])
        assert token_ids.dtype == torch.long
        assert token_ids.is_cuda

    def test_memory_leak(self):
        batch_size = 128
        vocab_size = 2048
        output_length = 100
        padding_id = 0
        single_example = [1, 2, 3] + [padding_id] * (output_length - 1)
        nested_list_input_ids = [single_example] * batch_size
        assert isinstance(nested_list_input_ids, list)
        assert all(isinstance(x, list) for x in nested_list_input_ids)
        input_ids = torch.tensor(nested_list_input_ids, device="cuda")
        assert input_ids.shape == torch.Size([batch_size, output_length + 2])
        last_non_padding_value = 2
        batch = torch.randn(
            (batch_size, vocab_size,
             output_length + last_non_padding_value + 1),
            device="cuda",
        )
        pipeline = LogitVectorToTokenPipeLine(GenerationConfig())
        start_mem = torch.cuda.memory_allocated()
        pipeline.logits_to_tokens_return_one(batch)
        end_mem = torch.cuda.memory_allocated()
        assert end_mem == start_mem

    def test_sampling(self):
        generation_config = GenerationConfig.from_pretrained("gpt2")
        generation_config.do_sample = True
        generation_config.top_k = 2
        generation_config.top_p = 0.95
        generation_config.temperature = 0.7
        generation_config.num_return_sequences = 1
        pipeline = LogitVectorToTokenPipeLine(generation_config)
        # Batch size 2:
        batch = torch.tensor(
            [
                [[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.5, 0.2, 0.3]],
                [[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.5, 0.2, 0.3]],
            ],
            device="cuda",
        )
        output_length = 3
        token_ids = pipeline.logits_to_tokens_return_one(batch)
        assert isinstance(token_ids, Tensor)
        assert token_ids.shape == torch.Size([2, output_length])
        assert token_ids.dtype == torch.int32
        assert token_ids.is_cuda

        # Batch size 1:
        batch = torch.tensor(
            [[[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.5, 0.2, 0.3]]],
            device="cuda",
        )
        output_length = 3
        token_ids = pipeline.logits_to_tokens_return_one(batch)
        assert isinstance(token_ids, Tensor)
        assert token_ids.shape == torch.Size([1, output_length])
        assert token_ids.dtype == torch.int32

    def test_beam_search_raises(self):
        config = GenerationConfig(num_beams=2)
        with pytest.raises(ValueError, match="Beam search is not supported."):
            LogitVectorToTokenPipeLine(config, 0)

    @patch(
        "src.grouped_sampling.logits_vec_to_token.flashinfer.sampling.top_k_top_p_sampling_from_logits"
    )
    def test_sample_logits_top_k_only(self, mock_sampling):
        config = GenerationConfig(do_sample=True, top_k=2, top_p=None)
        pipeline = LogitVectorToTokenPipeLine(config, 0)
        logits = torch.randn(4, 10)
        mock_sampling.return_value = torch.tensor([1, 2, 3, 4])
        result = pipeline.sample_logits(logits)
        mock_sampling.assert_called_once()
        args = mock_sampling.call_args.kwargs
        assert args["logits"].equal(logits.contiguous())
        assert args["top_k"] == 2
        assert args["top_p"] is None
        assert args["generator"] == pipeline.rng
        assert torch.equal(result, torch.tensor([1, 2, 3, 4]))

    @patch(
        "src.grouped_sampling.logits_vec_to_token.flashinfer.sampling.top_k_top_p_sampling_from_logits"
    )
    def test_sample_logits_top_p_only(self, mock_sampling):
        config = GenerationConfig(do_sample=True, top_k=None, top_p=0.8)
        pipeline = LogitVectorToTokenPipeLine(config, 0)
        logits = torch.randn(4, 10)
        mock_sampling.return_value = torch.tensor([4, 3, 2, 1])
        result = pipeline.sample_logits(logits)
        mock_sampling.assert_called_once()
        args = mock_sampling.call_args.kwargs
        assert args["logits"].equal(logits.contiguous())
        assert args["top_k"] is None
        assert args["top_p"] == 0.8
        assert args["generator"] == pipeline.rng
        assert torch.equal(result, torch.tensor([4, 3, 2, 1]))

    @patch(
        "src.grouped_sampling.logits_vec_to_token.flashinfer.sampling.top_k_top_p_sampling_from_logits"
    )
    def test_sample_logits_top_k_and_top_p(self, mock_sampling):
        config = GenerationConfig(do_sample=True, top_k=3, top_p=0.7)
        pipeline = LogitVectorToTokenPipeLine(config, 0)
        logits = torch.randn(4, 10)
        mock_sampling.return_value = torch.tensor([0, 1, 2, 3])
        result = pipeline.sample_logits(logits)
        mock_sampling.assert_called_once()
        args = mock_sampling.call_args.kwargs
        assert args["logits"].equal(logits.contiguous())
        assert args["top_k"] == 3
        assert args["top_p"] == 0.7
        assert args["generator"] == pipeline.rng
        assert torch.equal(result, torch.tensor([0, 1, 2, 3]))

    @patch(
        "src.grouped_sampling.logits_vec_to_token.flashinfer.sampling.top_k_top_p_sampling_from_logits"
    )
    def test_sample_logits_top_k_and_top_p_none(self, mock_sampling):
        config = GenerationConfig(do_sample=True, top_k=None, top_p=None)
        pipeline = LogitVectorToTokenPipeLine(config, 0)
        logits = torch.randn(4, 10)
        mock_sampling.return_value = torch.tensor([9, 8, 7, 6])
        result = pipeline.sample_logits(logits)
        # mock_sampling.assert_called_once_with(logits=logits.contiguous(), top_k=None, top_p=None, generator=pipeline.rng)
        mock_sampling.assert_called_once()
        args = mock_sampling.call_args.kwargs
        assert args["logits"].equal(logits.contiguous())
        assert args["top_k"] is None
        assert args["top_p"] is None
        assert args["generator"] == pipeline.rng
        assert torch.equal(result, torch.tensor([9, 8, 7, 6]))

    def test_sample_logits_argmax(self):
        config = GenerationConfig(do_sample=False)
        pipeline = LogitVectorToTokenPipeLine(config, 0)
        logits = torch.tensor([[0.1, 0.5, 0.4], [0.2, 0.1, 0.7]])
        result = pipeline.sample_logits(logits)
        assert torch.equal(result, torch.tensor([1, 2]))

    def test_top_p_zero_top_k_one_forces_argmax(self):
        config = GenerationConfig(do_sample=True, top_k=1, top_p=0)
        pipeline = LogitVectorToTokenPipeLine(config, 0)
        logits = torch.tensor([[0.1, 0.5, 0.4], [0.2, 0.1, 0.7]])
        result = pipeline.sample_logits(logits)
        assert torch.equal(result, torch.tensor([1, 2]))

    @patch(
        "src.grouped_sampling.logits_vec_to_token.flashinfer.sampling.top_k_top_p_sampling_from_logits"
    )
    def test_sample_logits_actual_flashinfer(self, mock_sampling):
        config = GenerationConfig(do_sample=True, top_k=5, top_p=0.9)
        pipeline = LogitVectorToTokenPipeLine(config, 0)
        logits = torch.randn(2, 10)
        mock_sampling.return_value = torch.tensor([2, 7])
        result = pipeline.sample_logits(logits)
        mock_sampling.assert_called_once()
        args = mock_sampling.call_args.kwargs
        assert args["logits"].equal(logits.contiguous())
        assert args["top_k"] == 5
        assert args["top_p"] == 0.9
        assert args["generator"] == pipeline.rng
        assert torch.equal(result, torch.tensor([2, 7]))

    def test_logits_to_tokens_return_many(self):
        generation_config = GenerationConfig(do_sample=True,
                                             top_k=2,
                                             top_p=0.95,
                                             temperature=1.0)
        pipeline = LogitVectorToTokenPipeLine(generation_config)
        batch = torch.tensor(
            [
                [[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.5, 0.2, 0.3]],
                [[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.5, 0.2, 0.3]],
            ],
            device="cuda",
        )
        num_return_sequences = 4
        output = pipeline.logits_to_tokens_return_many(batch,
                                                       num_return_sequences)
        assert isinstance(output, torch.Tensor)
        assert output.shape == (2, num_return_sequences, 3)
        assert output.dtype in [torch.int32, torch.long]
        assert output.is_cuda

    def test_logits_to_tokens_return_many_single_batch(self):
        generation_config = GenerationConfig(do_sample=True,
                                             top_k=2,
                                             top_p=0.95,
                                             temperature=1.0)
        pipeline = LogitVectorToTokenPipeLine(generation_config)
        batch = torch.tensor(
            [[[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.5, 0.2, 0.3]]],
            device="cuda",
        )
        num_return_sequences = 3
        output = pipeline.logits_to_tokens_return_many(batch,
                                                       num_return_sequences)
        assert output.shape == (1, num_return_sequences, 3)
        assert output.is_cuda

    def test_logits_to_tokens_return_many_single_token(self):
        generation_config = GenerationConfig(do_sample=True,
                                             top_k=2,
                                             top_p=0.95,
                                             temperature=1.0)
        pipeline = LogitVectorToTokenPipeLine(generation_config)
        batch = torch.tensor(
            [
                [[0.1, 0.9, 0.0]],
                [[0.2, 0.3, 0.5]],
            ],
            device="cuda",
        )
        num_return_sequences = 2
        output = pipeline.logits_to_tokens_return_many(batch,
                                                       num_return_sequences)
        assert output.shape == (2, num_return_sequences, 1)
        assert output.is_cuda

    def test_logits_to_tokens_return_many_one_return_sequence(self):
        generation_config = GenerationConfig(do_sample=True,
                                             top_k=2,
                                             top_p=0.95,
                                             temperature=1.0)
        pipeline = LogitVectorToTokenPipeLine(generation_config)
        batch = torch.tensor(
            [
                [[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]],
            ],
            device="cuda",
        )
        num_return_sequences = 1
        output = pipeline.logits_to_tokens_return_many(batch,
                                                       num_return_sequences)
        assert output.shape == (1, 1, 2)
        assert output.is_cuda

    def test_logits_to_tokens_return_many_zero_return_sequence(self):
        generation_config = GenerationConfig(do_sample=True,
                                             top_k=2,
                                             top_p=0.95,
                                             temperature=1.0)
        pipeline = LogitVectorToTokenPipeLine(generation_config)
        batch = torch.tensor(
            [
                [[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]],
            ],
            device="cuda",
        )
        num_return_sequences = 0
        with pytest.raises(ValueError,
                           match="num_return_sequences must be positive"):
            pipeline.logits_to_tokens_return_many(batch, num_return_sequences)

    def test_logits_to_tokens_return_many_empty_batch(self):
        generation_config = GenerationConfig(do_sample=True,
                                             top_k=2,
                                             top_p=0.95,
                                             temperature=1.0)
        pipeline = LogitVectorToTokenPipeLine(generation_config)
        batch = torch.empty((0, 2, 3), device="cuda")
        num_return_sequences = 2
        with pytest.raises(ValueError, match="logits should not be empty"):
            pipeline.logits_to_tokens_return_many(batch, num_return_sequences)

    def test_logits_to_tokens_return_many_raises_if_not_sampling(self):
        generation_config = GenerationConfig(do_sample=False)
        pipeline = LogitVectorToTokenPipeLine(generation_config)
        batch = torch.tensor(
            [
                [[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]],
            ],
            device="cuda",
        )
        with pytest.raises(
                ValueError,
                match="logits_to_tokens_return_many is only supported when do_sample is True",
        ):
            pipeline.logits_to_tokens_return_many(batch, 2)
