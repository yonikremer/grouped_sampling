import torch
from torch import Tensor, multinomial, argmax, inference_mode
from transformers import (
    GenerationConfig,
    LogitsProcessorList,
    GenerationMixin,
)
from transformers.generation import LogitNormalization

from src.grouped_sampling.softmax import SoftmaxLogitNormalization


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
    if not isinstance(generation_config, GenerationConfig):
        raise TypeError(
            f"generation_config should be a GenerationConfig, got {type(generation_config)}"
        )
    generation_config_copy = GenerationConfig(**generation_config.to_dict())
    generation_config_copy.renormalize_logits = True
    generation_config_copy.num_beams = 1
    return generation_config_copy


class LogitVectorToTokenPipeLine:
    def __init__(
        self,
        generation_config: GenerationConfig,
    ):
        generation_config_copy = prepare_generation_config(generation_config)
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

    @inference_mode()
    def logits_to_tokens(
        self,
        input_ids: Tensor,
        logits: Tensor,
        output_length: int,
    ) -> Tensor:
        """
        Convert a batch of logit matrices to tokens.
        args:
            input_ids: Tensor of shape (batch_size, input_seq_len) with the input sequences.
            batch: Tesnor of shape (batch_size, output_seq_len, vocab_size).
            output_length: int. The length of the output sequences.
        Returns:
            A Tensor of shape (batch_size, output_seq_len) with the tokens for every sequence in the batch.
        """
        batch_size = input_ids.size(0)
        answer = torch.empty(
            (batch_size, output_length), dtype=torch.long, device=logits.device
        )
        for i in range(output_length):
            # noinspection PyTypeChecker
            logits[:, i, :] = self.logit_wrapper(
                input_ids=input_ids, scores=logits[:, i, :]
            )  # a vector of size batch_size with the ith token for every sequence in the batch
            if self.do_sample:
                answer[:, i] = multinomial(logits[:, i, :], num_samples=1)
            else:
                answer[:, i] = argmax(logits[:, i, :], dim=-1)
        return answer
