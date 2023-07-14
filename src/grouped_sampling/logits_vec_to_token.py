import torch
from torch import Tensor, multinomial, argmax, inference_mode, softmax
from transformers import (
    GenerationConfig,
    LogitsProcessorList,
    GenerationMixin,
)


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
    generation_config_copy.num_beams = 1
    return generation_config_copy


class LogitVectorToTokenPipeLine:
    def __init__(
        self,
        generation_config: GenerationConfig,
        pad_token_id: int,
    ):
        generation_config_copy = prepare_generation_config(generation_config)
        mixin = GenerationMixin()
        mixin.generation_config = generation_config_copy
        # noinspection PyProtectedMember
        self.logit_wrapper: LogitsProcessorList = mixin._get_logits_warper(
            generation_config_copy
        )
        self.do_sample = generation_config_copy.do_sample
        self.pad_token_id = pad_token_id

    @inference_mode()
    def logits_to_tokens(
        self,
        input_ids: Tensor,
        logits: Tensor,
        output_length: int,
        last_non_padding_indecies: Tensor,
    ) -> Tensor:
        """
        Convert a batch of logit matrices to tokens.
        args:
            input_ids: Tensor of shape (batch_size, input_seq_len) with the input sequences.
            batch: Tesnor of shape (batch_size, output_seq_len, vocab_size).
            output_length: int. The length of the output sequences.
            last_non_padding_indecies: Tensor of shape (batch_size)
                with the index of the last non-padding token in each sequence.
        Returns:
            A Tensor of shape (batch_size, output_seq_len) with the tokens for every sequence in the batch.
        """
        batch_size = input_ids.size(0)
        answer = torch.empty(
            (batch_size, output_length), dtype=torch.long, device=logits.device
        )
        first_padding_indecies = last_non_padding_indecies + 1
        extra_padding = torch.full(
            size=(batch_size, 1),
            fill_value=self.pad_token_id,
            dtype=torch.long,
            device=logits.device
        )
        dim1_indecies = torch.arange(batch_size)
        current_tokens = torch.cat([input_ids, extra_padding], dim=1)
        for i in range(output_length):
            # noinspection PyTypeChecker
            logits[:, i, :] = self.logit_wrapper(
                input_ids=current_tokens, scores=logits[:, i, :])
            if self.do_sample:
                probs = softmax(logits[:, i, :], dim=-1)
                answer[:, i] = multinomial(probs, num_samples=1).squeeze(-1)
            else:
                answer[:, i] = argmax(logits[:, i, :], dim=-1)
            current_tokens[dim1_indecies, first_padding_indecies] = answer[:, i]
            first_padding_indecies += 1
        return answer
