import copy
from typing import List

import torch
from torch import Tensor, long, FloatTensor, multinomial, argmax, exp
from transformers import (
    GenerationConfig,
    LogitsProcessorList,
    GenerationMixin,
    LogitsProcessor,
)
from transformers.generation import LogitNormalization


class SoftmaxLogitNormalization(LogitsProcessor):
    """
    LogitsProcessor for Softmax normalization of logits.
    """

    def __init__(self, temperature: float):
        self.temperature = temperature

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        scores = exp(scores / self.temperature)
        return scores / scores.sum(-1, keepdim=True)


class LogitVectorToTokenPipeLine:
    @staticmethod
    def prepare_generation_config(
        generation_config: GenerationConfig,
    ) -> GenerationConfig:
        """
        Prepare a generation config for the LogitVectorToTokenPipeLine.
        Args:
            generation_config: GenerationConfig. The generation config to prepare.
        Returns:
            A new GenerationConfig.
        """
        generation_config_copy = copy.deepcopy(generation_config)
        generation_config_copy.renormalize_logits = True
        generation_config_copy.num_beams = 1
        required_attrs = (
            "epsilon_cutoff",
            "temperature",
            "top_k",
            "top_p",
            "typical_p",
            "eta_cutoff",
        )
        for attr in required_attrs:
            if not hasattr(generation_config_copy, attr):
                setattr(generation_config_copy, attr, None)
        return generation_config_copy

    def __init__(
        self,
        generation_config: GenerationConfig,
    ):
        generation_config_copy = self.prepare_generation_config(generation_config)
        mixin = GenerationMixin()
        mixin.generation_config = generation_config_copy
        # noinspection PyProtectedMember
        self.logit_wrapper: LogitsProcessorList = mixin._get_logits_warper(
            generation_config_copy
        )

        self.do_sample = generation_config_copy.do_sample
        if isinstance(self.logit_wrapper[-1], LogitNormalization):
            self.logit_wrapper.pop(-1)
        if self.do_sample:
            softmax = SoftmaxLogitNormalization(
                temperature=generation_config_copy.temperature
            )
            self.logit_wrapper.append(softmax)

    def single_logit_vector_to_token(
        self, input_ids: Tensor, logits: FloatTensor, **kwargs
    ) -> long:
        """
        Convert a single logit vector to a token id.
        args:
            input_ids: Tensor of shape (input_seq_len, ) with the input sequence.
            logits: torch.FloatTensor of shape (vocab_size, ) with the logits for the next token.
        """
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        # noinspection PyTypeChecker
        wrapped_logits = self.logit_wrapper(
            input_ids=input_ids, scores=logits, **kwargs
        )
        if self.do_sample:
            return multinomial(wrapped_logits, num_samples=1)
        return argmax(wrapped_logits, dim=-1)

    def batch_to_tokens(
        self,
        input_ids: Tensor,
        batch: List[Tensor],
    ) -> List[Tensor]:
        """
        Convert a batch of logit matrices to tokens.
        args:
            input_ids: Tensor of shape (batch_size, input_seq_len) with the input sequences.
            batch: list of Tensors with shape (output_seq_len, vocab_size) with the logits for every sequence in the
                    batch.
        Returns:
            A list of Tensors with length output_seq_len with the output tokens for every sequence in the batch.
        """
        all_output_seqs = []
        for logit_matrix, curr_sequence in zip(batch, input_ids):
            curr_output_seq = torch.stack(
                [
                    self.single_logit_vector_to_token(curr_sequence, logit_vector)
                    for logit_vector in logit_matrix
                ],
            ).squeeze()
            all_output_seqs.append(curr_output_seq)
        return all_output_seqs
