from typing import Optional, List

import torch
import tqdm
from torch import inference_mode, Tensor
from transformers import GenerationConfig

from src.grouped_sampling.base_pipeline import BasePipeLine


class ReturnManyPipeLine(BasePipeLine):
    """
    A pipeline for generating multiple sequences for each prompt using grouped sampling.
    """
    def __init__(
            self,
            model_name: str,
            max_batch_size: int,
            seed: Optional[int] = 0,
            model_kwargs: Optional[dict] = None,
            generation_config: Optional[GenerationConfig] = None,
    ):
        """
        Create a new ReturnManyPipeLine.
        For more details, see the documentation of BasePipeLine.
        """
        super().__init__(
            model_name=model_name,
            model_kwargs=model_kwargs,
            generation_config=generation_config,
            max_batch_size=max_batch_size,
            seed=seed
        )

    @inference_mode()
    def generate_return_many_tokens(
            self,
            prompts: Tensor,
            output_length: int,
            num_return_sequences: int,
    ) -> Tensor:
        """
        Generates a pre-determined number of responses for each prompt
        Arguments:
            prompts: a tensor with the prompt tokens, size (batch_size, max_input_length) - the output of tokenize_and_pad.
            output_length: int the number of tokens to generate for each prompt.
            num_return_sequences: int the number of responses to generate for each prompt.
        Returns:
            Tensor with the generated tokens, size (batch_size, num_return_sequences, output_length).
        """
        if output_length <= 0 or num_return_sequences <= 0:
            raise ValueError(f"output_length and num_return_sequences must be positive, got {output_length} and {num_return_sequences}")
        if prompts.dim() != 2:
            raise ValueError(
                f"prompts should be a 2D tensor, got {prompts.dim()}D tensor"
            )
        prompts.requires_grad = False
        batch_size = prompts.shape[0]
        if batch_size > self.max_batch_size:
            outputs: Tensor = torch.zeros(batch_size, num_return_sequence, output_length, dtype=prompts.dtype, device=prompts.device)
            for i in tqdm.tqdm(range(0, batch_size, self.max_batch_size)):
                curr_batch = prompts[i: i + self.max_batch_size, :]
                outputs[i: i + self.max_batch_size, :, :] = self.generate_return_many_tokens(
                    curr_batch,
                    output_length,
                    num_return_sequences
                )
                torch.cuda.empty_cache()
            return outputs
        logits = self.tokens_batch_to_logit_matrices(prompts, output_length)
        tokens = self.logit_to_token_pipeline.logits_to_tokens_return_many(logits, num_return_sequences)
        return tokens

    @inference_mode()
    def generate_return_many(
            self,
            prompts: List[str],
            output_length: int,
            num_return_sequences: int,
    ) -> List[List[str]]:
        """
        Generates a pre-determined number of responses for each prompt
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
        self._validate_prompts(prompts)
        num_prompts = len(prompts)
        if num_prompts == 0:
            return []
        input_tokens = self.tokenize_and_pad(prompts, output_length)
        output_tokens_buffer = torch.zeros((num_prompts, num_return_sequences, output_length), dtype=torch.int32, device=self.device)
        for i in range(0, len(prompts), self.max_batch_size):
            output_tokens_buffer[i: i + self.max_batch_size, :, :] = self.generate_return_many_tokens(
                input_tokens[i: i + self.max_batch_size, :],
                output_length,
                num_return_sequences
            )
        # detokenize
        output_tokens_buffer = output_tokens_buffer.reshape(num_prompts * num_return_sequences, output_length).tolist()
        output_strings = self.tokenizer.batch_decode(
            output_tokens_buffer,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        # transform to A nested list of strings
        output_strings = [
            output_strings[i * num_return_sequences:(i + 1) * num_return_sequences]
            for i in range(num_prompts)
        ]
        return output_strings

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
