import torch
from torch import Tensor, ones, long, cat, full, inference_mode
from transformers import GenerationConfig

from src.grouped_sampling.logits_vec_to_token import LogitVectorToTokenPipeLine
from src.grouped_sampling.model import get_model
from src.grouped_sampling.tokenizer import get_tokenizer


class EndToEndSingleSequencePipeLine:
    def __init__(
            self,
            model_name: str,
            load_in_8bit: bool = False,
    ):
        self.tokenizer = get_tokenizer(model_name)
        self.model = get_model(
            model_name=model_name,
            load_in_8bit=load_in_8bit,
        )
        self.device = self.model.device
        self.max_total_len = self.model.config.max_position_embeddings
        generation_config = GenerationConfig.from_model_config(self.model.config)
        self.logit_vector_to_token = LogitVectorToTokenPipeLine(generation_config=generation_config)

    @inference_mode()
    def tokens_to_logit_matrix(
            self,
            tokens: Tensor,
            output_length: int,
    ) -> Tensor:
        """
        Given a sequence of batch,
        returns the logits matrix of shape (output_length, vocab_size)
        where logits[i] is the logits vector of the i-th next token
        complexity: O(n^2 + output_length^2) where n is the target_length of the batch
        notice that the logits are not divided by the temperature in this function.
        args:
            tokens: the sequence of input tokens
            output_length: the target_length of the output in tokens.
        retuens:
            the logits matrix of shape (output_length, vocab_size) where logits[i, :] is the logits vector of the i-th
            token in the output.
        """
        padding_tokens: Tensor = full((output_length,), self.tokenizer.pad_token_id, dtype=long, device=self.device)
        padded_tokens: Tensor = cat(
            (tokens, padding_tokens),
            dim=0
        ).unsqueeze(0)
        # the target_length of padded_tokens is input_length + output_length - 1
        # so creating it is O(input_length + output_length - 1)
        full_length = padded_tokens.shape[1]  # input_length + output_length - 1
        if full_length > self.max_total_len:
            padded_tokens = padded_tokens[:, -self.max_total_len:]
            # O(self.max_input_len) which is constant so O(1)
            full_length = self.max_total_len
        attention_mask: Tensor = ones([1, full_length], dtype=long, device=self.device)
        outputs = self.model(
            output_attentions=False,
            output_hidden_states=False,
            input_ids=padded_tokens,
            attention_mask=attention_mask,
        )
        unscaled_relevant_logits = outputs.logits[0, -output_length:, :self.tokenizer.vocab_size]
        return unscaled_relevant_logits

    @staticmethod
    def _validate_prompt(prompt: str) -> None:
        if not isinstance(prompt, str):
            raise TypeError(f"prompt should be a string, got {type(prompt)}")
        if len(prompt) == 0:
            raise ValueError("prompt should not be empty")

    def _validate_output_length(self, output_length):
        if not isinstance(output_length, int):
            raise TypeError(f"output_length should be an int, got {type(output_length)}")
        if output_length < 0:
            raise ValueError(f"output_length should be positive, got {output_length}")
        if output_length >= self.max_total_len:
            raise ValueError(
                f"output_length should be less than {self.max_total_len}, got {output_length}"
            )

    def generate(
            self,
            prompt: str,
            output_length: int,
    ) -> str:
        """
        Given a prompt, returns a string of length output_length
        Args:
            prompt: the prompt to be completed
            output_length: the target_length of the output in tokens.
        Returns:
            a string of length output_length
        """
        self._validate_prompt(prompt)
        if output_length == 0:
            return ""
        self._validate_output_length(output_length)
        input_tokens = torch.tensor(
            self.tokenizer.encode(prompt),
            device=self.device,
            dtype=long,
            requires_grad=False,
        )
        logits_matrix = self.tokens_to_logit_matrix(input_tokens, output_length)
        output_tokens = self.logit_vector_to_token.logit_matrix_to_tokens(
            input_tokens,
            logits_matrix,
        )
        return self.tokenizer.decode(output_tokens)
