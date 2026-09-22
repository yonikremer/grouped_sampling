from __future__ import annotations

import flashinfer
import torch
from torch import Generator, Tensor, argmax, inference_mode
from transformers import GenerationConfig


class LogitVectorToTokenPipeLine:
    def __init__(
            self,
            generation_config: GenerationConfig,
            seed: int | None = 0,
    ):
        if isinstance(generation_config.num_beams, int) and generation_config.num_beams > 1:
            raise ValueError("Beam search is not supported.")
        self.do_sample = generation_config.do_sample
        self.top_p = generation_config.top_p
        self.top_k = generation_config.top_k
        self.temperature = generation_config.temperature
        if self.top_p == 0 or self.top_k == 1:
            self.do_sample = False
        self.rng = Generator(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        if seed is not None:
            self.rng.manual_seed(seed)

    def sample_logits(self, logits: Tensor) -> Tensor:
        """
        Take a batch of logits and return the sampled token for each item in the batch.
        If do_sample is False, return the token with maximum probability.
        If top_p is not None, use top-p sampling.
        If top_k is not None, use top-k sampling.
        If both top_p and top_k are not None, use top-k top-p sampling.

        args:
            logits: Tensor of shape (batch_size, vocab_size), containing the logits for each token in the vocabulary.
        return: Tensor of shape (batch_size) with the one sampled token for each logit vector in the batch.
        """
        if not self.do_sample:
            # return the token with maximum probability
            return argmax(logits, dim=-1)
        if not logits.is_contiguous():
            logits = logits.contiguous()
        if self.top_k is None:
            probs = torch.softmax(logits, dim=-1)
            return flashinfer.sampling.top_p_sampling_from_probs(
                probs=probs,
                top_p=self.top_p,
                generator=self.rng
            )
        return flashinfer.sampling.top_k_top_p_sampling_from_logits(
            logits=logits / self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            generator=self.rng
        )

    @inference_mode()
    def logits_to_tokens_return_one(
            self,
            logits: Tensor,
    ) -> Tensor:
        """
        Convert a batch of logit matrices to tokens.
        args:
            logits: Tensor of shape (batch_size, output_seq_len, vocab_size).
        Returns:
            A Tensor of shape (batch_size, output_seq_len) with the tokens for every sequence in the batch.
        """
        batch_size = logits.size(0)
        output_length = logits.size(1)
        vocab_size = logits.size(2)
        output_logits = logits.reshape(batch_size * output_length, vocab_size)
        sampled_tokens = self.sample_logits(output_logits)
        return sampled_tokens.reshape(batch_size, output_length)

    @inference_mode()
    def logits_to_tokens_return_many(
            self,
            logits: Tensor,
            num_return_sequences: int,
    ) -> Tensor:
        """
        Convert a batch of logit matrices to tokens.
        args:
            logits: Tensor of shape (batch_size, output_seq_len, vocab_size).
            num_return_sequences: int. The number of sequences to return for each input sequence.
        Returns:
            A Tensor of shape (batch_size, num_return_sequences, output_seq_len)
                with num_return_sequences generated output sequences for each prompt in the tokens.
        """
        if num_return_sequences <= 0:
            raise ValueError(f"num_return_sequences must be positive, got {num_return_sequences}")
        if not self.do_sample:
            raise ValueError("""
            logits_to_tokens_return_many is only supported when do_sample is True.
            return many with greedy decoding does not make sense, it would return the same output multiple times.
            """)
        if any(s == 0 for s in logits.size()):
            raise ValueError(f"logits should not be empty, got {logits.size()}")
        batch_size = logits.size(0)
        output_length = logits.size(1)
        vocab_size = logits.size(2)
        logits = logits.reshape(batch_size * output_length, vocab_size)
        if not logits.is_contiguous():
            logits = logits.contiguous()
        # from each logit vector I want to sample num_return_sequences tokens
        # so the indices should have each value from 0 to batch_size * output_length, output_length times
        indices = torch.arange(0, batch_size * output_length, device=logits.device).repeat_interleave(
            num_return_sequences)
        if self.top_k is None:
            probs = torch.softmax(logits, dim=-1)
            sampled_tokens = flashinfer.sampling.top_p_sampling_from_probs(
                probs=probs,
                top_p=self.top_p,
                generator=self.rng,
                indices=indices
            )
        else:
            sampled_tokens = flashinfer.sampling.top_k_top_p_sampling_from_logits(
                logits=logits / self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                generator=self.rng,
                indices=indices
            )
        return sampled_tokens.reshape(batch_size, num_return_sequences, output_length)
