from typing import List, Optional, Union

import torch
from torch import no_grad
import tqdm
from transformers import GenerationConfig

from src.grouped_sampling.base_pipeline import BasePipeLine
from src.grouped_sampling.logits_vec_to_token import LogitVectorToTokenPipeLine


class ReturnOnePipeLine(BasePipeLine):
    """
    A pipeline for generating a single sequence for each prompt using grouped sampling.
    """

    def __init__(
            self,
            model_name: str,
            model_kwargs: Optional[dict] = None,
            generation_config: Optional[GenerationConfig] = None,
            max_batch_size: int = 128,
            seed: Optional[int] = 0
    ):
        """
        Create a new ReturnOnePipeLine.
        Args:
            model_name: str. The name of the model to load from huggingfacehub.
            model_kwargs: Optional dict. Additional arguments to pass to the model's from_pretrained method.
                If None, no additional arguments will be passed.
            generation_config: Optional GenerationConfig. The generation config for the model.
                If None, the method would create a generation config from the model's config.
            max_batch_size: int. The maximum batch size to use.
            seed: Optional int. The seed to use for sampling.
        Returns:
            A new BatchEndToEndSingleSequencePipeLine.
        Raises:
            RepositoryNotFoundError: If the model is not found in the model hub.
            TypeError: If one of the arguments is of the wrong type.
        """
        super().__init__(
            model_name=model_name,
            model_kwargs=model_kwargs,
            max_batch_size=max_batch_size,
        )
        if generation_config is not None and not isinstance(
                generation_config, GenerationConfig
        ):
            raise TypeError(
                f"generation_config should be a GenerationConfig or None, got {type(generation_config)}"
            )
        if generation_config is None:
            generation_config = GenerationConfig.from_model_config(self.model.config)
        self.logit_to_token_pipeline = LogitVectorToTokenPipeLine(
            generation_config=generation_config,
            seed=seed,
        )

    @no_grad()
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
        num_prompts = len(prompts)
        if num_prompts == 0:
            return []
        self._validate_prompts(prompts)
        padded_tokens = self.tokenize_and_pad(prompts, output_length)
        output_tokens_buffer = torch.zeros((num_prompts, output_length), dtype=padded_tokens.dtype, device=self.device)
        for i in range(0, num_prompts, self.max_batch_size):
            logits = self.tokens_batch_to_logit_matrices(
                padded_tokens[i:i+self.max_batch_size], output_length
            )
            output_tokens_buffer[i:i+self.max_batch_size] = self.logit_to_token_pipeline.logits_to_tokens_return_one(
                logits=logits
            )
        output_tokens = output_tokens_buffer.tolist()
        return self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
