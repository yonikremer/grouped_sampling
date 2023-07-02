import copy

import torch
from torch import Tensor, long, FloatTensor, LongTensor, multinomial, argmax, exp
from transformers import GenerationConfig, LogitsProcessorList, GenerationMixin, LogitsProcessor
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
    def __init__(
            self,
            generation_config: GenerationConfig,
    ):
        generation_config_copy = copy.deepcopy(generation_config)
        generation_config_copy.renormalize_logits = True
        mixin = GenerationMixin()
        mixin.generation_config = generation_config_copy
        self.logit_wrapper: LogitsProcessorList = mixin._get_logits_warper(generation_config_copy)

        self.do_sample = generation_config_copy.do_sample
        if self.do_sample and isinstance(self.logit_wrapper[-1], LogitNormalization):
            self.logit_wrapper[-1] = SoftmaxLogitNormalization(generation_config_copy.temperature)

    def single_logit_vector_to_token(
            self,
            input_ids: LongTensor,
            logits: FloatTensor,
            **kwargs
    ) -> long:
        """
        Convert a single logit vector to a token id.
        args:
            input_ids: torch.LongTensor of shape (input_seq_len, ) with the input sequence.
            logits: torch.FloatTensor of shape (vocab_size, ) with the logits for the next token.
        """
        if input_ids.dim() != 1 or input_ids.shape[0] == 0:
            raise ValueError(f"input_ids should be a 1D long tensor. "
                             f"Got input ids with shape {input_ids.shape} and dimention {input_ids.dim()}")
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        if logits.dim() != 2 or min(logits.shape) == 0:
            raise ValueError(f"logits should be a 1D float tensor."
                             f"Got logits with shape {logits.shape} and dimention {logits.dim()}")
        wrapped_logits = self.logit_wrapper(input_ids=input_ids, scores=logits, **kwargs)
        if self.do_sample:
            if torch.min(wrapped_logits) < 0:
                raise RuntimeError(f"Probabilities should be non-negative, got {wrapped_logits}")
            if torch.sum(wrapped_logits) > 1.0:
                raise RuntimeError(f"Probabilities should sum to 1.0, got {wrapped_logits}")
            return multinomial(wrapped_logits, num_samples=1)
        return argmax(wrapped_logits, dim=-1)

    def logit_vectors_to_token(
            self,
            input_ids: LongTensor,
            logit_vectors: FloatTensor,
            **kwargs
    ) -> Tensor:
        """
        Convert multipule logit vectors to token ids in parallel.
        args:
            input_ids: torch.LongTensor of shape (input_seq_len, ) with the input sequence.
            logits: torch.FloatTensor of shape (output_seq_len, vocab_size) with the logits for the next token.
            The input sequence is the same for all the logits.
        """
        if input_ids.dim() != 1 or input_ids.shape[0] == 0:
            raise ValueError(f"input_ids should be a 1D long tensor"
                             f"Got input ids with shape {input_ids.shape} and dimention {input_ids.dim()}")
        if logit_vectors.dim() != 2 or min(logit_vectors.shape) == 0:
            raise ValueError(f"logits should be a 2D float tensor"
                             f"Got logits with shape {logit_vectors.shape} and dimention {logit_vectors.dim()}")

        def logit_vector_to_token(vector) -> long:
            return self.single_logit_vector_to_token(input_ids, vector, **kwargs)

        tokens = []
        for logit_vector in logit_vectors:
            tokens.append(logit_vector_to_token(logit_vector))

        return torch.stack(tokens, dim=0).squeeze()
