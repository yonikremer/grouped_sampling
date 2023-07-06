from typing import List

import torch
from torch import Tensor, long, inference_mode, full, argmax, int8, eq, ones_like
from transformers import GenerationConfig

from src.grouped_sampling.logits_vec_to_token import LogitVectorToTokenPipeLine
from src.grouped_sampling.model import get_model
from src.grouped_sampling.tokenizer import get_tokenizer


class BatchEndToEndSingleSequencePipeLine:
    def __init__(
            self,
            model_name: str,
            load_in_8bit: bool = False,
    ):
        if not load_in_8bit:
            torch.set_float32_matmul_precision('high')
        self.tokenizer = get_tokenizer(model_name)
        self.model = get_model(
            model_name=model_name,
            load_in_8bit=load_in_8bit,
        )
        self.device = self.model.device
        self.max_total_len = self.model.config.max_position_embeddings
        generation_config = GenerationConfig.from_model_config(self.model.config)
        self.logit_to_token_pipeline = LogitVectorToTokenPipeLine(generation_config=generation_config)

    def tokenize_and_pad(
            self,
            prompts: List[str],
            output_length: int,
    ) -> Tensor:
        """A helper function that converts a list of strings to a padded tensor of tokens."""
        prompt_tokens_list = self.tokenizer.batch_encode_plus(
            prompts, add_special_tokens=True, padding=True, return_attention_mask=False,
        )['input_ids']
        prompt_tokens = torch.tensor(
            prompt_tokens_list,
            device=self.device,
            dtype=long,
        )
        max_input_length = prompt_tokens.shape[1]
        padding_length: int = max_input_length + output_length - 1
        if padding_length > self.max_total_len:
            raise ValueError(f"padding_length should be at most {self.max_total_len}, got {padding_length}")
        batch_size = prompt_tokens.shape[0]
        extra_padding = full(
            fill_value=self.tokenizer.pad_token_id,
            size=(batch_size, padding_length - max_input_length),
            dtype=long,
            device=self.device,
        )
        return torch.cat([prompt_tokens, extra_padding], dim=1)

    def tokens_batch_to_logit_matrices(
            self,
            padded_tokens: Tensor,
            output_length: int,
    ) -> List[Tensor]:
        """
        Given a batch of prompts where each prompt is a sequence of tokens, and an output_length,
        returns the logits matrices of shape (batch_size, output_length, vocab_size)
        where logits[i] is the logits matrix of the i-th prompt.
        """
        if padded_tokens.dim() != 2:
            raise ValueError(f"tokens should be a 2D tensor, got {padded_tokens.dim()}D tensor")
        if padded_tokens.requires_grad:
            raise ValueError("tokens should not require grad")
        if not isinstance(output_length, int):
            raise TypeError(f"output_length should be an int, got {type(output_length)}")
        if output_length <= 0:
            raise ValueError(f"output_length should be positive, got {output_length}")
        attenction_mask = ones_like(padded_tokens, dtype=torch.float, device=self.device, requires_grad=False)
        all_logits = self.model(
            output_attentions=False,
            output_hidden_states=False,
            input_ids=padded_tokens,
            attention_mask=attenction_mask,
        ).logits
        padding_int_tokens = eq(padded_tokens, self.tokenizer.pad_token_id).to(int8)
        last_non_pad_indices = argmax(padding_int_tokens, dim=1) - 1
        relavent_logits = [
            all_logits[i, index:index + output_length]
            for i, index in enumerate(last_non_pad_indices)
        ]
        return relavent_logits

    def _validate_output_length(self, output_length: int) -> None:
        if not isinstance(output_length, int):
            raise TypeError(f"output_length should be an int, got {type(output_length)}")
        if output_length <= 0:
            raise ValueError(f"output_length should be positive, got {output_length}")
        if output_length >= self.max_total_len:
            raise ValueError(f"output_length should be smaller than {self.max_total_len}, got {output_length}")

    @staticmethod
    def _validate_prompts(prompts: List[str]):
        if not isinstance(prompts, list):
            raise TypeError(f"strings should be a list, got {type(prompts)}")
        if min(len(string) for string in prompts) == 0:
            raise ValueError("strings should not contain empty strings")

    @inference_mode()
    def genearte_batch(
            self,
            prompts: List[str],
            output_length: int,
    ) -> List[str]:
        """
        Given a batch of prompts and output length, generates a list of output strings.
        Args:
            prompts: a list of strings
            output_length: the length of the output strings (excluding the prompts) in tokens
        """
        if output_length == 0:
            return ["" for _ in prompts]
        self._validate_output_length(output_length)
        if len(prompts) == 0:
            return []
        self._validate_prompts(prompts)
        padded_tokens = self.tokenize_and_pad(prompts, output_length)
        logits = self.tokens_batch_to_logit_matrices(padded_tokens, output_length)
        output_tokens = self.logit_to_token_pipeline.batch_to_tokens(
            input_ids=padded_tokens,
            batch=logits,
        )
        return self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
