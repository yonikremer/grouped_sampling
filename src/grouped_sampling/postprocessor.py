from torch import tensor
from transformers import PreTrainedTokenizer

from .token_ids import TokenIDS


class PostProcessor:
    """The post processor class is used to convert the token ids to text and tensors"""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
    ):
        self.tokenizer: PreTrainedTokenizer = tokenizer

    def __call__(
        self,
        token_ids: TokenIDS,
        num_new_tokens: int,
        prompt_len: int,
        return_text: bool,
        return_tensors: bool,
        return_full_text: bool,
        clean_up_tokenization_spaces: bool,
        prefix_len: int = 0,
        postfix_len: int = 0,
    ):
        """
        A helper method for __call__
        that converts the token ids to dictionary
        token_ids - the token ids from the _forward method
        prompt_len - the length of the tokenized prompt
        the rest of the arguments are the arguments
        from the __call__ method
        look up the documentation of the __call__ method for more info
        Complexity: O(n) where n is the number of tokens in token_ids
        """
        # define n as the length of the token_ids
        full_prompt_len = prefix_len + prompt_len + postfix_len

        generated_tokens = token_ids[full_prompt_len:]
        # O(num_new_tokens) because we are copying
        # num_new_tokens elements from one list to the other
        if return_full_text:
            prompt_tokens = token_ids[prefix_len: prefix_len + prompt_len]
            # Without prefix and postfix
            # O(prompt_len) because we are copying prompt_len
            # elements from one list to the other
            final_token_list = prompt_tokens + generated_tokens
            # O(prompt_len + num_new_tokens)
            # because we are copying elements from one list to the other
        else:
            final_token_list = generated_tokens
            # O(1) because we are saving the reference to the list
        final_ans = {}
        if return_tensors:
            final_ans["generated_token_ids"] = tensor(final_token_list)
            # O(prompt_len + num_new_tokens)
            # because we are copying at maximum
            # prompt_len + num_new_tokens elements from a list to a tensor
        if return_text:
            final_ans["generated_text"] = self.tokenizer.decode(
                final_token_list,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )
            # decoding is O(m)
            # where m is the number of tokens in the text
            # so the complexity of this line is O(n)
            # because final_token_list could be at most n tokens long
        return final_ans
