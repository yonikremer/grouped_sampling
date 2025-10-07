from typing import List, Optional

import torch
import tqdm
from torch import inference_mode
from transformers import GenerationConfig

from src.grouped_sampling.base_pipeline import BasePipeLine
from src.grouped_sampling.logits_vec_to_token import LogitVectorToTokenPipeLine


class ReturnManyPipeLine(BasePipeLine):
    """
    A pipeline for generating multiple sequences for each prompt using grouped sampling.
    """

    def __init__(
        self,
        model_name: str,
        model_kwargs: Optional[dict] = None,
        max_batch_size: int = 128,
        top_p: float = 1.0,
        top_k: int = 0,
        temperature: float = 1.0,
        seed: Optional[int] = 0,
    ):
        super(ReturnManyPipeLine, self).__init__(
            model_name=model_name,
            model_kwargs=model_kwargs,
            max_batch_size=max_batch_size,
        )
        if not isinstance(top_p, float):
            raise TypeError(f"top_p should be a float, got {type(top_p)}")
        if not isinstance(top_k, int):
            raise TypeError(f"top_k should be an int, got {type(top_k)}")
        if not isinstance(temperature, float):
            raise TypeError(
                f"temperature should be a float, got {type(temperature)}")
        if top_p < 0.0:
            raise ValueError(f"top_p should be at least 0.0, got {top_p}")
        if top_k < 0:
            raise ValueError(f"top_k should be at least 0, got {top_k}")
        if temperature <= 0.0:
            raise ValueError(
                f"temperature should be positive, got {temperature}")
        self.temperature = temperature
        if top_p == 0 or top_k == 1:
            raise ValueError("""
                return many pipeline does not support top_p=0 or top_k=1 (greedy decoding)
                Because greedy decoding does not return multiple sequences.
                """)
        generation_config = GenerationConfig(top_p=top_p,
                                             top_k=top_k,
                                             temperature=temperature,
                                             do_sample=True)
        self.logit_to_token_pipeline = LogitVectorToTokenPipeLine(
            generation_config=generation_config, seed=seed)

    @inference_mode()
    def generate_return_many(
        self,
        prompts: List[str],
        output_length: int,
        num_return_sequences: int,
    ) -> List[List[str]]:
        """
        Generates a pre-determined number of responses for each propmt
        Arguments:
            prompts: a list of strings - the prompts to generate responses for.
            output_length: int the number of tokens to generate for each prompt.
            num_return_sequences: int the number of responses to generate for each prompt.
        Returns:
            A list of lists of strings - the responses for each prompt.
        """
        self._validate_num_return_sequences(num_return_sequences)
        if isinstance(prompts, str):
            prompts = [prompts]
        if output_length == 0:
            return [[""] for _ in prompts]
        if num_return_sequences == 0:
            return []
        self._validate_output_length(output_length)
        if len(prompts) == 0:
            return []
        if len(prompts) > self.max_batch_size:
            outputs: List[List[str]] = []
            for i in tqdm.tqdm(range(0, len(prompts), self.max_batch_size)):
                batch = prompts[i:i + self.max_batch_size]
                outputs.extend(
                    self.generate_return_many(batch, output_length,
                                              num_return_sequences))
                torch.cuda.empty_cache()
            return outputs
        self._validate_prompts(prompts)
        padded_tokens = self.tokenize_and_pad(prompts, output_length)
        logits = self.tokens_batch_to_logit_matrices(padded_tokens,
                                                     output_length)
        tokens = self.logit_to_token_pipeline.logits_to_tokens_return_many(
            logits, num_return_sequences)
        assert tokens.dtype in {torch.int32, torch.int64, torch.long}
        return [
            self.tokenizer.batch_decode(tokens[i, :, :],
                                        skip_special_tokens=True)
            for i in range(len(prompts))
        ]

    @staticmethod
    def _validate_num_return_sequences(num_return_sequences):
        if not isinstance(num_return_sequences, int):
            raise TypeError(
                f"num_return_sequences should be an int, got {type(num_return_sequences)}"
            )
        if num_return_sequences <= 0:
            raise ValueError(
                f"num_return_sequences should be positive, got {num_return_sequences}"
            )
