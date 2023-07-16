from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, long, inference_mode, full, argmax, int8, eq, ones_like
import tqdm
from transformers import GenerationConfig

from src.grouped_sampling.logits_vec_to_token import LogitVectorToTokenPipeLine
from src.grouped_sampling.model import get_model
from src.grouped_sampling.tokenizer import get_tokenizer


class BatchPipeLine:
    def __init__(
        self,
        model_name: str,
        load_in_8bit: bool = False,
        model_kwargs: Optional[dict] = None,
        generation_config: Optional[GenerationConfig] = None,
        max_batch_size: int = 128,
    ):
        """
        Create a new BatchEndToEndSingleSequencePipeLine.
        Args:
            model_name: str. The name of the model to load from huggingfacehub.
            load_in_8bit: bool. If True, the model will be loaded in 8bit mode, which is faster but less accurate.
            model_kwargs: Optional dict. Additional arguments to pass to the model's from_pretrained method.
                If None, no additional arguments will be passed.
            generation_config: Optional GenerationConfig. The generation config for the model.
                If None, the method would create a generation config from the model's config.
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
        if not isinstance(load_in_8bit, bool):
            raise TypeError(
                f"load_in_8bit should be a bool, got {type(load_in_8bit)}")
        if model_kwargs is not None and not isinstance(model_kwargs, dict):
            raise TypeError(
                f"model_kwargs should be a dict or None, got {type(model_kwargs)}"
            )
        if generation_config is not None and not isinstance(
            generation_config, GenerationConfig
        ):
            raise TypeError(
                f"generation_config should be a GenerationConfig or None, got {type(generation_config)}"
            )
        if not isinstance(max_batch_size, int):
            raise TypeError(
                f"max_batch_size should be an int, got {type(max_batch_size)}"
            )
        if max_batch_size < 1:
            raise ValueError(
                f"max_batch_size should be at least 1, got {max_batch_size}"
            )
        self.max_batch_size = max_batch_size
        if not load_in_8bit:
            torch.set_float32_matmul_precision("high")
        self.tokenizer = get_tokenizer(model_name)
        if model_kwargs is None:
            model_kwargs = {}
        self.model = get_model(
            model_name=model_name,
            load_in_8bit=load_in_8bit,
            **model_kwargs,
        )
        self.device: torch.device = self.model.device
        self.max_total_len = self.model.config.max_position_embeddings
        if generation_config is None:
            generation_config = GenerationConfig.from_model_config(
                self.model.config)
        self.logit_to_token_pipeline = LogitVectorToTokenPipeLine(
            generation_config=generation_config,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    def tokenize_and_pad(
        self,
        prompts: List[str],
        output_length: int,
    ) -> Tensor:
        """A helper function that converts a list of strings to a padded tensor of tokens."""
        prompt_tokens_list = self.tokenizer.batch_encode_plus(
            prompts,
            add_special_tokens=True,
            padding=True,
            return_attention_mask=False,
        )["input_ids"]
        prompt_tokens = torch.tensor(
            prompt_tokens_list,
            device=self.device,
            dtype=long,
        )
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
                f"output_length should be an int, got {type(output_length)}"
            )
        if output_length <= 0:
            raise ValueError(
                f"output_length should be positive, got {output_length}")
        attenction_mask = ones_like(
            padded_tokens,
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
            raise RuntimeError(
                f"Model returned NaN logits."
                f" logits: {all_logits}"
                f" tokens: {padded_tokens}"
                f" attention_mask: {attenction_mask}"
            )
        padding_int_tokens = eq(
            padded_tokens,
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
                                                  index: index + output_length]
        return relavent_logits, last_non_pad_indices

    def _validate_output_length(self, output_length: int) -> None:
        if not isinstance(output_length, int):
            raise TypeError(
                f"output_length should be an int, got {type(output_length)}"
            )
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

    @inference_mode()
    def generate_batch_return_one(
        self,
        prompts: Union[List[str], str],
        output_length: int,
    ) -> List[str]:
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
        if len(prompts) == 0:
            return []
        if len(prompts) > self.max_batch_size:
            outputs: List[str] = []
            for i in tqdm.tqdm(range(0, len(prompts), self.max_batch_size)):
                batch = prompts[i: i + self.max_batch_size]
                outputs.extend(self.generate_batch_return_one(batch, output_length))
                torch.cuda.empty_cache()
            return outputs
        self._validate_prompts(prompts)
        padded_tokens = self.tokenize_and_pad(prompts, output_length)
        logits, last_non_padding_indecies = self.tokens_batch_to_logit_matrices(
            padded_tokens, output_length)
        output_tokens = self.logit_to_token_pipeline.logits_to_tokens_return_one(
            input_ids=padded_tokens,
            logits=logits,
            output_length=output_length,
            last_non_padding_indecies=last_non_padding_indecies,
        )
        return self.tokenizer.batch_decode(
            output_tokens, skip_special_tokens=True)
