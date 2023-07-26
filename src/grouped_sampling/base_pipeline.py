from typing import List, Optional, Tuple

import torch
from torch import Tensor, argmax, eq, full, int8, long, ones_like

from src.grouped_sampling.model import get_model
from src.grouped_sampling.tokenizer import get_tokenizer


class BasePipeLine:

    def __init__(
        self,
        model_name: str,
        model_kwargs: Optional[dict] = None,
        max_batch_size: int = 128,
    ):
        """
        Create a new BasePipeLine.
        Args:
            model_name: str. The name of the model to load from huggingfacehub.
            model_kwargs: Optional dict. Additional arguments to pass to the model's from_pretrained method.
                If None, no additional arguments will be passed.
            max_batch_size: int. The maximum batch size to use.
        Returns:
            A new BatchEndToEndSingleSequencePipeLine.
        Raises:
            RepositoryNotFoundError: If the model is not found in the model hub.
            TypeError: If one of the arguments is of the wrong type.
        """
        if not isinstance(model_name, str):
            raise TypeError(
                f"model_name should be a string, got {type(model_name)}")
        if model_kwargs is not None and not isinstance(model_kwargs, dict):
            raise TypeError(
                f"model_kwargs should be a dict or None, got {type(model_kwargs)}"
            )
        if not isinstance(max_batch_size, int):
            raise TypeError(
                f"max_batch_size should be an int, got {type(max_batch_size)}")
        if max_batch_size < 1:
            raise ValueError(
                f"max_batch_size should be at least 1, got {max_batch_size}")
        self.max_batch_size = max_batch_size
        self.tokenizer = get_tokenizer(model_name)
        if model_kwargs is None:
            model_kwargs = {}
        self.model = get_model(
            model_name=model_name,
            **model_kwargs,
        )
        self.device: torch.device = self.model.device
        self.max_total_len = self.model.config.max_position_embeddings

    def tokens_batch_to_logit_matrices(
        self,
        padded_tokens: Tensor,
        output_length: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Given a batch of prompts where each prompt is a sequence of tokens, and an output_length,
        returns the logits matrices of shape (batch_size, output_length, vocab_size)
        where logits[i] is the logits matrix of the i-th prompt.
        """
        if padded_tokens.dim() != 2:
            raise ValueError(
                f"tokens should be a 2D tensor, got {padded_tokens.dim()}D tensor"
            )
        if padded_tokens.requires_grad:
            raise ValueError("tokens should not require grad")
        if not isinstance(output_length, int):
            raise TypeError(
                f"output_length should be an int, got {type(output_length)}")
        if output_length <= 0:
            raise ValueError(
                f"output_length should be positive, got {output_length}")
        attenction_mask = ones_like(padded_tokens,
                                    dtype=torch.long,
                                    device=self.device,
                                    requires_grad=False)
        all_logits = self.model(
            output_attentions=False,
            output_hidden_states=False,
            input_ids=padded_tokens,
            attention_mask=attenction_mask,
        ).logits
        if all_logits.isnan().any():
            raise RuntimeError(f"Model returned NaN logits."
                               f" logits: {all_logits}"
                               f" tokens: {padded_tokens}"
                               f" attention_mask: {attenction_mask}")
        padding_int_tokens = eq(padded_tokens,
                                self.tokenizer.pad_token_id).to(int8)
        last_non_pad_indices = argmax(padding_int_tokens, dim=1) - 1
        batch_size = padded_tokens.shape[0]
        relavent_logits = torch.empty(
            (batch_size, output_length, all_logits.shape[-1]),
            device=self.device,
            dtype=all_logits.dtype,
        )
        for i, index in enumerate(last_non_pad_indices):
            relavent_logits[i, :, :] = all_logits[i,
                                                  index:index + output_length]
        return relavent_logits, last_non_pad_indices

    def _validate_output_length(self, output_length: int) -> None:
        if not isinstance(output_length, int):
            raise TypeError(
                f"output_length should be an int, got {type(output_length)}")
        if output_length <= 0:
            raise ValueError(
                f"output_length should be positive, got {output_length}")
        if output_length >= self.max_total_len:
            raise ValueError(
                f"output_length should be smaller than {self.max_total_len}, got {output_length}"
            )

    @staticmethod
    def _validate_prompts(prompts: List[str]):
        if not isinstance(prompts, list):
            raise TypeError(f"strings should be a list, got {type(prompts)}")
        if min(len(string) for string in prompts) == 0:
            raise ValueError("strings should not contain empty strings")

    def tokenize_and_pad(
        self,
        prompts: List[str],
        output_length: int,
    ) -> Tensor:
        """A helper function that converts a list of strings to a padded tensor of tokens."""
        prompt_tokens: Tensor = self.tokenizer.batch_encode_plus(
            prompts,
            add_special_tokens=True,
            padding=True,
            return_attention_mask=False,
            return_tensors="pt",
        )["input_ids"].to(self.device)
        max_input_length = prompt_tokens.shape[1]
        padding_length: int = max_input_length + output_length - 1
        if padding_length > self.max_total_len:
            raise ValueError(
                f"padding_length should be at most {self.max_total_len}, got {padding_length}"
            )
        batch_size = prompt_tokens.shape[0]
        extra_padding = full(
            fill_value=self.tokenizer.pad_token_id,
            size=(batch_size, padding_length - max_input_length),
            dtype=long,
            device=self.device,
        )
        return torch.cat([prompt_tokens, extra_padding], dim=1)
