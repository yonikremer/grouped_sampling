from __future__ import annotations

import torch
from torch import no_grad
from transformers import GenerationConfig

from .base_pipeline import BasePipeLine


class ReturnOnePipeLine(BasePipeLine):
    """
    A pipeline for generating a single sequence for each prompt using grouped sampling.
    """

    def __init__(
            self,
            model_name: str,
            max_batch_size: int,
            seed: int | None = 0,
            model_kwargs: dict | None = None,
            generation_config: GenerationConfig | None = None,
    ):
        """
        Create a new ReturnOnePipeLine.
        For more details, see the documentation of BasePipeLine.
        """
        super().__init__(
            model_name=model_name,
            model_kwargs=model_kwargs,
            generation_config=generation_config,
            max_batch_size=max_batch_size,
            seed=seed
        )

    @no_grad()
    def generate_batch_return_one(
            self,
            prompts: list[str] | str,
            output_length: int,
    ) -> list[str]:
        """
        Given a batch of prompts and output length, generates a list of output strings.
        Args:
            prompts: a string or a list of strings.
            output_length: the length of the output strings (excluding the prompts) in tokens.
        Returns:
            A list of strings.
             The i-th string is the output of the i-th prompt.
             If the prompts is a string, the output is a list of length 1.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        if output_length == 0:
            return ["" for _ in prompts]
        self._validate_output_length(output_length)
        num_prompts = len(prompts)
        if num_prompts == 0:
            return []
        self._validate_prompts(prompts)
        padded_tokens = self.tokenize_and_pad(prompts, output_length)
        output_tokens_buffer = torch.zeros((num_prompts, output_length), dtype=padded_tokens.dtype, device=self.device)
        for i in range(0, num_prompts, self.max_batch_size):
            logits = self.tokens_batch_to_logit_matrices(
                padded_tokens[i:i + self.max_batch_size], output_length
            )
            output_tokens_buffer[i:i + self.max_batch_size] = self.logit_to_token_pipeline.logits_to_tokens_return_one(
                logits=logits
            )
        output_tokens = output_tokens_buffer.tolist()
        return self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
