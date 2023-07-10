import copy
from typing import List

import torch
from torch import Tensor, multinomial, argmax, exp
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

    def batch_to_tokens(
        self,
        input_ids: Tensor,
        batch: List[Tensor],
    ) -> Tensor:
        """
        Convert a batch of logit matrices to tokens.
        args:
            input_ids: Tensor of shape (batch_size, input_seq_len) with the input sequences.
            batch: list of Tensors with shape (output_seq_len, vocab_size) with the logits for every sequence in the
                    batch.
        Returns:
            A Tensor of shape (batch_size, output_seq_len) with the tokens for every sequence in the batch.
        """
        batch: Tensor = torch.stack(batch, dim=1)
        # batch shape: (output_seq_len, batch_size, vocab_size)
        answer = []
        for i in range(batch.shape[0]):
            # noinspection PyTypeChecker
            ith_tokens = self.logit_wrapper(
                input_ids=input_ids, scores=batch[i], **{}
            )  # a vector of size batch_size with the ith token for every sequence in the batch
            if self.do_sample:
                ith_tokens = multinomial(ith_tokens, num_samples=1)
            else:
                ith_tokens = argmax(ith_tokens, dim=-1)
            answer.append(ith_tokens)
        return torch.stack(answer, dim=1)
