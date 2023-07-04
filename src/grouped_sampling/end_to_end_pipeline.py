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

    def string_to_tokens(
            self,
            string: str,
    ) -> Tensor:
        """Convert a string to a Tensor of tokens."""
        if not isinstance(string, str):
            raise TypeError(f"string should be a string, got {type(string)}")
        if len(string) == 0:
            raise ValueError("string should not be empty")
        tokens = torch.tensor(
            self.tokenizer.encode(string),
            device=self.device,
            dtype=long,
            requires_grad=False,
        )
        if tokens.dim() != 1:
            raise ValueError(f"tokens should be a 1D tensor, got {tokens.dim()}D tensor")
        return tokens

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
        if not isinstance(tokens, Tensor):
            raise TypeError(f"tokens should be a Tensor, got {type(tokens)}")
        if tokens.dim() != 1:
            raise ValueError(f"tokens should be a 1D tensor, got {tokens.dim()}D tensor")
        if not isinstance(output_length, int):
            raise TypeError(f"output_length should be an int, got {type(output_length)}")
        if output_length <= 0:
            raise ValueError(f"output_length should be positive, got {output_length}")
        if tokens.shape[0] + output_length > self.max_total_len:
            raise ValueError(f"the total length of the input and output should be at most {self.max_total_len}, "
                             f"got {tokens.shape[0] + output_length}")
        if tokens.shape[0] == 0:
            raise ValueError("tokens should not be empty")
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

    def tokens_to_string(self, tokens: Tensor) -> str:
        """Convert a Tensor of tokens to a string."""
        if not isinstance(tokens, Tensor):
            raise TypeError(f"tokens should be a Tensor, got {type(tokens)}")
        if tokens.dim() != 1:
            raise ValueError(f"tokens should be a 1D Tensor, got {tokens.dim()}D")
        if min(tokens.shape) == 0:
            raise ValueError("tokens should not be empty")
        if tokens.dtype not in [torch.int64, torch.int32, torch.int16, torch.int8, torch.long]:
            raise ValueError(f"tokens should be a Tensor of type integer/long, got {tokens.dtype}")
        return self.tokenizer.decode(tokens.tolist())

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
        if not isinstance(prompt, str):
            raise TypeError(f"prompt should be a string, got {type(prompt)}")
        if len(prompt) == 0:
            raise ValueError("prompt should not be empty")
        if not isinstance(output_length, int):
            raise TypeError(f"output_length should be an int, got {type(output_length)}")
        if output_length < 0:
            raise ValueError(f"output_length should be positive, got {output_length}")
        if output_length == 0:
            return ""
        if output_length >= self.max_total_len:
            raise ValueError(
                f"output_length should be less than {self.max_total_len}, got {output_length}"
            )
        input_tokens = self.string_to_tokens(prompt)
        if input_tokens.shape[0] + output_length > self.max_total_len:
            raise ValueError(
                f"the total target_length of input tokens and output tokens should be less than or equal to "
                f"{self.max_total_len}, got {input_tokens.shape[0] + output_length}"
            )
        logits_matrix = self.tokens_to_logit_matrix(input_tokens, output_length)
        output_tokens = self.logit_vector_to_token.logit_matrix_to_tokens(
            input_tokens,
            logits_matrix,
        )
        return self.tokens_to_string(output_tokens)
