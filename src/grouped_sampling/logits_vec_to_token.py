from typing import Optional

from torch import Tensor, argmax, inference_mode, Generator
from transformers import (
    GenerationConfig
)
import flashinfer


class LogitVectorToTokenPipeLine:
    def __init__(
            self,
            generation_config: GenerationConfig,
            seed: Optional[int] = 0,
    ):
        if isinstance(generation_config.num_beams, int) and generation_config.num_beams > 1:
            raise ValueError("Beam search is not supported.")
        self.do_sample = generation_config.do_sample
        self.top_p = generation_config.top_p
        self.top_k = generation_config.top_k
        if self.top_p == 0 or self.top_k == 1:
            self.do_sample = False
        self.rng = Generator(device='cuda')
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
        return flashinfer.sampling.top_k_top_p_sampling_from_logits(
            logits=logits,
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
        Raises:
            ValueError: If the output length and last_non_padding_indexes are incompatible with the number of logits in the batch.
        """
        batch_size = logits.size(0)
        output_length = logits.size(1)
        vocab_size = logits.size(2)
        output_logits = logits.reshape(batch_size * output_length, vocab_size)
        sampled_tokens = self.sample_logits(output_logits)
        sampled_tokens = sampled_tokens.reshape(batch_size, output_length)
        return sampled_tokens
