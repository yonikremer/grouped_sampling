from typing import Tuple

from torch import LongTensor, cat
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import TruncationStrategy, BatchEncoding


class PreProcessor:
    framework: str = "pt"

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            max_input_len: int,
            truncation: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
    ):
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.max_input_len: int = max_input_len
        self.truncation: TruncationStrategy = truncation

    def get_token_tensor(
            self,
            text: str,
    ) -> LongTensor:
        """Complexity: O(n) where n is the number of characters in the text"""
        if len(text) == 0:
            return LongTensor([])
        # tokenizing a string is O(n) where n is the length of the string
        tokenized_text = self.tokenizer(
            text,
            return_tensors=self.framework,
            padding=False,
            add_special_tokens=False,
            truncation=self.truncation,
            max_length=self.max_input_len,
        )
        token_tensor: LongTensor = tokenized_text["input_ids"]
        # O(1) because we are accessing a single element
        # in a dictionary and saving the reference to it.
        token_tensor = token_tensor.squeeze()
        # O(n) where n is the number of tokens in the text
        # because we are copying n elements from one tensor to the other
        # the number of tokens in the text
        # is always less than the number of characters in the text
        return token_tensor

    def __call__(
            self,
            prompt: str,
            prefix: str = "",
            postfix: str = "",
    ) -> Tuple[LongTensor, int, int, int]:
        """A helper method for __call__ that tokenize the prompt
        all the arguments are sent directly from the __call__ method
        Returns:
            the tokenized prompt as a list of ints,
             the length of the prompt,
             the length of the prefix,
             the length of the postfix

        Complexity: O(a + b + c) where:
            'a' is the number of characters in the prefix
            'b' is the number of characters in the prompt
            'c' is the number of characters in the postfix"""

        prefix_tokens: LongTensor = self.get_token_tensor(prefix)
        # O(A) where A is the number of characters in the prefix.
        postfix_tokens: LongTensor = self.get_token_tensor(postfix)
        # O(B) where B is the number of characters in the postfix.
        prompt_tokens: LongTensor = self.get_token_tensor(prompt)
        # O(C) where C is the number of characters in the prompt.
        token_tensor: LongTensor = cat((prefix_tokens, prompt_tokens, postfix_tokens))
        # O(a + b + c) where 'a' is the number of tokens in the prefix.
        # 'b' is the number of tokens in the prompt.
        # 'c' is the number of tokens in the postfix.
        # we know that the number of tokens is less than the number of characters
        return token_tensor, len(prefix_tokens), len(prompt_tokens), len(postfix_tokens)
