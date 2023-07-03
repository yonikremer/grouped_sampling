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
        generation_config_copy.num_beams = 1
        mixin = GenerationMixin()
        mixin.generation_config = generation_config_copy
        self.logit_wrapper: LogitsProcessorList = mixin._get_logits_warper(generation_config_copy)

        self.do_sample = generation_config_copy.do_sample
        if isinstance(self.logit_wrapper[-1], LogitNormalization):
            self.logit_wrapper.pop(-1)
        if self.do_sample:
            softmax = SoftmaxLogitNormalization(temperature=generation_config_copy.temperature)
            self.logit_wrapper.append(softmax)

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
            if not torch.all(wrapped_logits >= 0):
                raise RuntimeError(f"Probabilities should be non-negative, got {wrapped_logits}")
            if not torch.isclose(torch.sum(wrapped_logits), torch.tensor(1.0)):
                raise RuntimeError(f"Probabilities should sum to 1.0, got {wrapped_logits}")
            return multinomial(wrapped_logits, num_samples=1)
        return argmax(wrapped_logits, dim=-1)

    def logit_matrix_to_tokens(
            self,
            input_ids: LongTensor,
            logit_vectors: FloatTensor,
            **kwargs
    ) -> LongTensor:
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

        tokens = [logit_vector_to_token(logit_vector) for logit_vector in logit_vectors]

        return torch.stack(tokens, dim=0).squeeze()

    def batch_to_tokens(
            self,
            input_ids: LongTensor,
            batch: FloatTensor,
    ):
        """
        Convert a batch of logit matrices to tokens.
        args:
            input_ids: LongTensor of shape (batch_size, input_seq_len) with the input sequences.
            batch: FloatTensor of shape (batch_size, output_seq_len, vocab_size) with the logits for the next token.
        Returns:
            A torch.LongTensor of shape (batch_size, output_seq_len) with the tokens.
        """
        if not isinstance(input_ids, LongTensor):
            raise ValueError(f"input_ids should be a LongTensor, got {type(input_ids)}")
        if not isinstance(batch, FloatTensor):
            raise ValueError(f"batch should be a FloatTensor, got {type(batch)}")
        if input_ids.dim() != 2 or min(input_ids.shape) == 0:
            raise ValueError(f"input_ids should be a 2D long tensor"
                             f"Got input ids with shape {input_ids.shape} and dimention {input_ids.dim()}")
        if batch.dim() != 3 or min(batch.shape) == 0:
            raise ValueError(f"batch should be a 3D float tensor"
                             f"Got batch with shape {batch.shape} and dimention {batch.dim()}")
        if batch.shape[0] != input_ids.shape[0]:
            raise ValueError(f"batch and input_ids should have the same batch size"
                             f"Got batch with shape {batch.shape} and input_ids with shape {input_ids.shape}")
        all_output_seqs = []
        for logit_matrix, curr_sequence in zip(batch, input_ids):
            curr_output_seq = torch.stack(
                [self.single_logit_vector_to_token(curr_sequence, logit_vector) for logit_vector in logit_matrix],
            ).squeeze()
            all_output_seqs.append(curr_output_seq)

        return torch.stack(all_output_seqs, dim=0)