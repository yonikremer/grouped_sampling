from __future__ import annotations

import timeit
from enum import Enum
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Optional, List, Union, Dict, Tuple, Any

from torch import LongTensor, ones, cuda, tensor, no_grad, Tensor, cat
from torch.nn import Softmax
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          AutoConfig,
                          BatchEncoding,
                          PreTrainedTokenizer,
                          PreTrainedModel)
from transformers.tokenization_utils_base import TruncationStrategy

TokenIDS = Union[List[int], Tuple[int]]
SingleAnswer = Dict[str, Union[str, Tensor]]
MAX_MODEL_INPUT_SIZE = 8192


class GenerationType(Enum):
    """The type of generation to use"""
    GREEDY = "greedy"
    TOP_K = "top_k"
    TOP_P = "top_p"
    TREE = "tree"
    RANDOM = "random"


class TextGenerator(Callable, ABC):
    """An abstract base class for
    A callable object that given a func_prompt
     and length of wanted answer,
    generates text
    the text generator has a model,
    and some parameters
    (Defined in the subclasses)"""

    model_name: str
    tokenizer: PreTrainedTokenizer
    model: PreTrainedModel
    vocab_size: int
    temp: float
    group_size: int
    padding_tokens: List[int]
    generation_type: GenerationType
    end_of_sentence_id: Optional[int]
    end_of_sentence_stop: bool
    max_input_len: int
    framework: str = "pt"
    answer_length_multiplier: int = 16

    def __init__(self, model_name: str, group_size: int,
                 temp: float = 1.0,
                 end_of_sentence_stop: bool = False,
                 answer_length_multiplier: int = 16, ):
        """Model name: the name of the model
        used for loading from hugging face hub
        group size: int
        the number of tokens to be predicted at each model call
        temp: float
        temperature parameter for the softmax function
        answer_length_multiplier: int
            if the answer length is not given,
            the maximum answer length is set to:
            the length of the prompt * answer_length_multiplier
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name)
        except RuntimeError as e:
            raise RuntimeError(f"Failed to execute model loading,"
                               f"Using the AutoModelForCausalLM.from_pretrained({model_name}) command") from e
        if cuda.is_available():
            self.model = self.model.cuda()
        self.vocab_size = self.tokenizer.vocab_size
        self.temp = temp
        self.group_size = group_size
        self.end_of_sentence_id = self.tokenizer.eos_token_id
        self.end_of_sentence_stop = end_of_sentence_stop and self.end_of_sentence_id is not None
        self.max_input_len = self.tokenizer.model_max_length
        max_len_is_huge = self.max_input_len > MAX_MODEL_INPUT_SIZE
        if max_len_is_huge or self.max_input_len is None:
            config = AutoConfig.from_pretrained(model_name)
            self.max_input_len = config.max_position_embeddings
            max_len_is_still_huge = self.max_input_len > MAX_MODEL_INPUT_SIZE
            if max_len_is_still_huge or self.max_input_len is None:
                raise ValueError(
                    "The maximum length of the model is too big"
                )
        self.answer_length_multiplier = answer_length_multiplier

    @property
    @abstractmethod
    def generation_type(self) -> GenerationType:
        """A method that chooses the generation type
        Returns:
            a GenerationType object"""
        raise NotImplementedError

    @property
    def padding_tokens(self):
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0
        return [pad_id] * (self.group_size - 1)

    def get_prob_mat(self, token_list: List[int]) \
            -> Tensor:
        """Returns the probability matrix
         as a list of lists of floats
         Time complexity: O(n^2 + group_size^2)
         where n is the number of tokens"""
        # define n as the number of tokens in token_list
        padded_token_list = token_list + self.padding_tokens
        # the length of padded_token_list is n + group_size - 1
        if len(padded_token_list) > self.max_input_len:
            padded_token_list = padded_token_list[-self.max_input_len:]
        # the length of padded_token_list is now = min(n + group_size - 1, self.max_input_len)
        # but self.max_input_len is so big that it is unlikely to be reached
        # so the length of padded_token_list is n + group_size - 1
        attention_len = len(padded_token_list)
        longer_token_tensor = LongTensor([padded_token_list])
        attention_mask = ones([1, attention_len])
        if cuda.is_available():
            longer_token_tensor = longer_token_tensor.cuda()
            attention_mask = attention_mask.cuda()
        inputs = {"input_ids": longer_token_tensor,
                  "attention_mask": attention_mask,
                  "labels": longer_token_tensor}
        # the length of all the inputs is n + group_size - 1

        with no_grad():
            # The time complexity of causal language model`s __call__ function
            # is O(n^2) where n is the length of the inputs
            outputs = self.model(**inputs)
            # so the complexity of this line is O((n + group_size - 1)^2)
            # which is O(n^2 + group_size^2 + group_size * n)
            # we now that if a > b and a, b > 1 then a^2 > ab so the complexity is O(n^2 + group_size^2)

        unscaled_logits: Tensor = outputs.logits.squeeze(0)
        # output.logits is a tensor of shape (1, n + group_size - 1, vocab_size)
        # so unscaled_logits is a tensor of shape (n + group_size - 1, vocab_size).
        # outputs.logits.squeeze(0) is equivalent to outputs.logits[0]
        # so the complexity of this line should be O(n + group_size)
        # because we copy (n + group_size - 1) * vocab_size from one tensor to the other where vocab_size is constant
        unscaled_relevant_logits: Tensor
        unscaled_relevant_logits = unscaled_logits[-self.group_size:, :self.vocab_size]
        # The shape of unscaled_relevant_logits is (group_size, vocab_size)
        # So the complexity of this line should be O(group_size) because we are coping group_size * vocab_size
        # elements from one tensor to the another
        scaled_relevant_logits = unscaled_relevant_logits / self.temp
        # We are dividing group_size * vocab_size elements so the complexity is O(group_size)
        prob_tensor: Tensor = Softmax(dim=1)(scaled_relevant_logits)
        # We are doing a softmax operator of group_size different vectors of size vocab_size
        # The complexity of the softmax for each vector is O(1) because the size of the vector is constant
        # the complexity of this line is O(1) because the softmax for each vector is done in parallel (Assuming a GPU).
        if cuda.is_available():
            prob_tensor = prob_tensor.cpu().detach()
            # Coping a tensor so the complexity is O(group_size)
            cuda.empty_cache()
            # This is an async operation so the complexity is O(1) because we are not waiting for it to finish
        if not self.end_of_sentence_stop:
            for prob_vec in prob_tensor:
                prob_vec[self.end_of_sentence_id] = 0.0
                # This line is O(1) because we are accessing a single element in a vector
                # So the complexity of this loop is O(group_size)
        return prob_tensor

    def get_token_tensor(
            self,
            text: str,
            truncation: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
    ) -> Tensor:
        """Complexity: O(n) where n is the number of characters in the text"""
        # tokenizing a string is O(n) where n is the length of the string
        tokenized_text = self.tokenizer(
            text,
            return_tensors=self.framework,
            padding=False,
            add_special_tokens=False,
            truncation=truncation,
            max_length=self.max_input_len,
        )
        is_dict = isinstance(tokenized_text, dict)
        # O(1)
        is_batch_encoding = isinstance(tokenized_text,
                                       BatchEncoding)
        # O(1)
        if is_dict or is_batch_encoding:
            token_tensor: Tensor = tokenized_text["input_ids"]
            # O(1) because we are accessing a single element in a dictionary and saving the reference to it.
        elif isinstance(tokenized_text, Tensor):
            token_tensor: Tensor = tokenized_text
            # O(1) because we are saving the reference to the tensor
        else:
            raise TypeError("The tokenizer output is not one of:"
                            "dict, BatchEncoding, Tensor")

        token_tensor = token_tensor.squeeze()
        # O(n) where n is the number of tokens in the text
        # because we are copying n elements from one tensor to the other
        # the number of tokens in the text is always less than the number of characters in the text

        return token_tensor

    def preprocess(
            self,
            prompt: str,
            prefix: str = "",
            truncation: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
            postfix: str = "",
    ) -> Tuple[Tensor, int, int, int]:
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

        if len(prefix) > 0:
            prefix_tokens = self.get_token_tensor(prefix, truncation)
            # O(n) where n is the number of characters in the prefix.
        else:
            prefix_tokens = LongTensor([])
            # O(1)
        if len(postfix) > 0:
            postfix_tokens = self.get_token_tensor(postfix, truncation)
            # O(n) where n is the number of characters in the postfix.
        else:
            postfix_tokens = LongTensor([])
            # O(1)
        prompt_tokens = self.get_token_tensor(prompt, truncation)
        # O(n) where n is the number of characters in the prompt.
        token_tensor = cat((prefix_tokens, prompt_tokens, postfix_tokens))
        # O(a + b + c) where 'a' is the number of tokens in the prefix.
        # 'b' is the number of tokens in the prompt.
        # 'c' is the number of tokens in the postfix.
        return token_tensor, len(prefix_tokens), len(prompt_tokens), len(postfix_tokens)

    @abstractmethod
    def _forward(
            self,
            tokenized_prompt: Tensor,
            num_new_tokens: Optional[int] = None,
            num_return_sequences: int = 1,
    ) -> List[TokenIDS]:
        """A helper method for __call__ that generates the new tokens
        Has a unique implementation for each subclass
        Args:
            tokenized_prompt: List[int] - the tokenized prompt from the preprocess method
            num_new_tokens: int - the number of new tokens to generate
                from the __call__ method
        Returns:
            the prompt + generated text as a list/tuple of ints"""
        pass

    def postprocess(
            self,
            token_ids: TokenIDS,
            num_new_tokens: Optional[int],
            prompt_len: int,
            return_text: bool,
            return_tensors: bool,
            return_full_text: bool,
            clean_up_tokenization_spaces: bool,
            prefix_len: int = 0,
            postfix_len: int = 0,
    ):
        """A helper method for __call__
        that converts the token ids to dictionary
        token_ids - the token ids from the _forward method
        prompt_len - the length of the tokenized prompt
        the rest of the arguments are the arguments
        from the __call__ method
        look up the documentation of the __call__ method for more info
        Complexity: O(n) where n is the number of tokens in token_ids
        """
        # define n as the length of the token_ids
        if num_new_tokens is None:
            shorten_token_list = token_ids
            # O(1)
        else:
            final_num_tokens = prefix_len + prompt_len + postfix_len + num_new_tokens
            # O(1)
            if len(token_ids) > final_num_tokens:
                shorten_token_list = token_ids[:final_num_tokens]
                # O(final_num_tokens) because we are copying final_num_tokens elements from one list to the other
            else:
                shorten_token_list = token_ids
                # O(1)

        generated_tokens = shorten_token_list[prefix_len + prompt_len + postfix_len:]
        # O(num_new_tokens) because we are copying num_new_tokens elements from one list to the other
        if return_full_text:
            prompt_tokens = shorten_token_list[prefix_len:prefix_len + prompt_len]  # Without prefix and postfix
            # O(prompt_len) because we are copying prompt_len elements from one list to the other
            final_token_list = prompt_tokens + generated_tokens
            # O(prompt_len + num_new_tokens) because we are copying elements from one list to the other
        else:
            final_token_list = generated_tokens
            # O(1) because we are saving the reference to the list
        final_ans = {}
        if return_tensors:
            final_ans["generated_token_ids"] = tensor(final_token_list)
            # O(prompt_len + num_new_tokens)
            # because we are copying at maximum prompt_len + num_new_tokens elements from a list to a tensor
        if return_text:
            final_ans["generated_text"] = self.tokenizer.decode(
                final_token_list,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces
            )
            # decoding is O(m) where m is the number of tokens in the text
            # so the complexity of this line is O(n) because final_token_list could be at most n tokens long
        return final_ans

    def __call__(
            self,
            prompt_s: Union[str, List[str]],
            max_new_tokens: Optional[int] = None,
            return_tensors: bool = False,
            return_text: bool = True,
            return_full_text: bool = True,
            clean_up_tokenization_spaces: bool = False,
            prefix: str = "",
            num_return_sequences: int = 1,
            truncation: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
            postfix: str = "",
    ) -> SingleAnswer | List[SingleAnswer] | List[List[SingleAnswer]]:
        """The function that outside code should call to generate text
        Args:
            prompt_s: str or list of str - the prompt(s) to start the generation from
                (the text given by the user)
                if many prompts are given as a list,
                the function process each one independently and returns them as a list.
                (the same as calling [__call__(prompt, *args, **kwargs)
                 for prompt in prompts])])
            max_new_tokens: Optional[int] > 0 - the number of tokens to generate
                if None, the function will generate tokens until one of them is the end of sentence token
            return_tensors: bool - whether to return the generated token ids
            return_text: bool - whether to return the generated string
            return_full_text: bool - whether to return the full text
                (prompt + generated text)
                (if false, it will return only the generated text)
            clean_up_tokenization_spaces: bool - whether to clean up tokenization spaces
                This parameter is forwarded to the decode function of the AutoTokenizer class
            prefix (`str`, defaults to an empty string):
                Prefix added to prompt.
            num_return_sequences (`int`, defaults to 1):
                The number of independently generated answers to return for each prompt.
                For SamplingGenerator: each answer will be generated with different seed.
                For TreeGenerator: the num_return_sequences with the highest scores will be returned.
            truncation: TruncationStrategy - whether to truncate the prompt
            postfix: str - a postfix to add to the prompt
        Returns:
            Each result comes as a dictionary with the following keys:
            - "generated_text" (`str`, present when `return_text=True`) -- The generated text.
            - "generated_token_ids" (`torch.tensor`, present when `return_tensors=True`) -- The token
              ids of the generated text.
            """

        if max_new_tokens is None and not self.end_of_sentence_stop:
            raise ValueError("max_new_tokens must be given if end_of_sentence_stop is False")
        if isinstance(prompt_s, list):
            return [self.__call__(
                prompt_s=prompt,
                max_new_tokens=max_new_tokens,
                return_text=return_text,
                return_tensors=return_tensors,
                return_full_text=return_full_text,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                truncation=truncation,
                postfix=postfix,
            ) for prompt in prompt_s]
        tokens: Tensor
        prefix_len: int
        postfix_len: int
        prompt_len: int

        tokens, prefix_len, prompt_len, postfix_len = self.preprocess(
            prompt=prompt_s, prefix=prefix, truncation=truncation, postfix=postfix)
        # O(len(prompt) + len(prefix) + len(postfix))

        if max_new_tokens is None:
            max_new_tokens = prompt_len * self.answer_length_multiplier
            # O(1)
        tokenized_answers: List[TokenIDS]
        tokenized_answers = self._forward(
            tokens,
            max_new_tokens,
            num_return_sequences
        )

        if num_return_sequences > 1:
            # O(sum(len(tokenized_answer) for tokenized_answer in tokenized_answers))
            return [self.postprocess(
                token_ids=tokenized_answer,
                num_new_tokens=max_new_tokens,
                prompt_len=prompt_len,
                return_text=return_text,
                return_tensors=return_tensors,
                return_full_text=return_full_text,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                prefix_len=prefix_len,
                postfix_len=postfix_len,
            ) for tokenized_answer in tokenized_answers]
        else:
            # O(len(tokenized_answers[0]))
            return self.postprocess(
                token_ids=tokenized_answers[0],
                num_new_tokens=max_new_tokens,
                prompt_len=prompt_len,
                return_text=return_text,
                return_tensors=return_tensors,
                return_full_text=return_full_text,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                prefix_len=prefix_len,
                postfix_len=postfix_len,
            )

    @abstractmethod
    def __repr__(self):
        pass

    def __str__(self):
        return repr(self)

    @abstractmethod
    def as_dict(self) -> Dict[str, Any]:
        pass

    @classmethod
    def from_dict(cls, my_dict: Dict[str, Any]):
        """Creates an TextGenerator from a dictionary
        The dictionary should have the same format as the dictionary returned by the as_dict method"""
        if "generation_type" in my_dict.keys():
            my_dict.pop("generation_type")
        return cls(**my_dict)


class NoCompletionsFound(Exception):
    def __init__(
            self,
            text_generator: TextGenerator,
            additional_info: str = ""):
        super(NoCompletionsFound, self).__init__(
            f"text generator: {text_generator} \n"
            f"additional info: {additional_info}")


def compare_generators(
        non_grouped_generator: TextGenerator,
        prompt: str,
        num_tokens: int,
        group_size: int):
    """Compares grouped and non-grouped text generators"""
    print(f"Your prompt:")
    print(prompt)

    start_non_grouped = timeit.default_timer()
    non_grouped_ans: str = non_grouped_generator(
        prompt_s=prompt,
        max_new_tokens=num_tokens
    )["generated_text"]
    stop_non_grouped = timeit.default_timer()
    non_grouped_time = stop_non_grouped - start_non_grouped
    print(f"Text generated by Non grouped sampling"
          f" in {non_grouped_time} seconds:")
    print(non_grouped_ans)

    non_grouped_generator.group_size = group_size
    grouped_generator = non_grouped_generator
    start_grouped_generation = timeit.default_timer()
    grouped_ans: str = grouped_generator(
        prompt_s=prompt,
        max_new_tokens=num_tokens
    )["generated_text"]
    stop_grouped_generation = timeit.default_timer()
    grouped_time = stop_grouped_generation - start_grouped_generation
    print(f"Text generated by grouped sampling"
          f" in {grouped_time} seconds:")
    print(grouped_ans)
