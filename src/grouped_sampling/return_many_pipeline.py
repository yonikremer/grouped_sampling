from typing import Optional, List

import torch
import tqdm
from torch import Tensor, inference_mode, multinomial

from src.grouped_sampling.base_pipeline import BasePipeLine
from src.grouped_sampling.end_to_end_probability_processor import (
    EndToEndProbabilityProcessor,
)


class ReturnManyPipeLine(BasePipeLine):
    def __init__(
        self,
        model_name: str,
        load_in_8bit: bool = False,
        model_kwargs: Optional[dict] = None,
        max_batch_size: int = 128,
        top_p: float = 1.0,
        top_k: int = 0,
        minimum_tokens_to_keep: int = 1,
        temperature: float = 1.0,
    ):
        super(ReturnManyPipeLine, self).__init__(
            model_name=model_name,
            load_in_8bit=load_in_8bit,
            model_kwargs=model_kwargs,
            max_batch_size=max_batch_size,
        )
        if not isinstance(top_p, float):
            raise TypeError(f"top_p should be a float, got {type(top_p)}")
        if not isinstance(top_k, int):
            raise TypeError(f"top_k should be an int, got {type(top_k)}")
        if not isinstance(minimum_tokens_to_keep, int):
            raise TypeError(
                f"minimum_tokens_to_keep should be an int, got {type(minimum_tokens_to_keep)}"
            )
        if not isinstance(temperature, float):
            raise TypeError(f"temperature should be a float, got {type(temperature)}")
        if top_p < 0.0:
            raise ValueError(f"top_p should be at least 0.0, got {top_p}")
        if top_k < 0:
            raise ValueError(f"top_k should be at least 0, got {top_k}")
        if minimum_tokens_to_keep < 1:
            raise ValueError(
                f"minimum_tokens_to_keep should be at least 1, got {minimum_tokens_to_keep}"
            )
        if temperature <= 0.0:
            raise ValueError(f"temperature should be positive, got {temperature}")
        self.temperature = temperature
        self.probablity_processor = EndToEndProbabilityProcessor(
            minimum_tokens_to_keep=minimum_tokens_to_keep,
            top_p=top_p,
            top_k=top_k,
        )

    @inference_mode()
    def logits_to_tokens_return_many(
        self,
        logits: Tensor,
        num_return_sequences: int,
    ) -> Tensor:
        """
        Convert a batch of logit matrices to tokens.
        args:
            logits: Tesnor of shape (batch_size, output_seq_len, vocab_size).
            num_return_sequences: int. The number of sequences to return for each input sequence.
        Returns:
            A Tensor of shape (batch_size, num_return_sequences, output_seq_len)
                with num_return_sequences generated output sequences for each prompt in the tokens.
        """
        batch_size, output_seq_len, vocab_size = logits.shape
        if self.temperature != 1.0:
            logits.div_(self.temperature)
        probs = logits.softmax(dim=-1)
        probs = self.probablity_processor(probs).view(-1, vocab_size)
        if not torch.isclose(probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1))).any():
            raise RuntimeError(
                "The probabilities do not sum to 1.0. This is a bug in the code."
            )
        if probs.shape != (batch_size * output_seq_len, vocab_size):
            raise RuntimeError(
                f"probs has shape {probs.shape} but should have shape {(batch_size * output_seq_len, vocab_size)}"
            )
        return multinomial(
            probs, num_samples=num_return_sequences, replacement=True
        ).view(batch_size, num_return_sequences, output_seq_len)

    @inference_mode()
    def generate_return_many(
            self,
            prompts: List[str],
            output_length: int,
            num_return_sequences: int,
    ) -> List[List[str]]:
        """"""
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
                batch = prompts[i: i + self.max_batch_size]
                outputs.extend(self.generate_return_many(batch, output_length, num_return_sequences))
                torch.cuda.empty_cache()
            return outputs
        self._validate_prompts(prompts)
        padded_tokens = self.tokenize_and_pad(prompts, output_length)
        logits, _ = self.tokens_batch_to_logit_matrices(
            padded_tokens, output_length
        )
        tokens = self.logits_to_tokens_return_many(logits, num_return_sequences)
        return [
            self.tokenizer.batch_decode(tokens[i, :, :], skip_special_tokens=True)
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
