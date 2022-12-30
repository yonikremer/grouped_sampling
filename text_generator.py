from __future__ import annotations

import timeit
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Optional, List, Union, Dict, Tuple, Any

from torch import LongTensor, tensor, cat
from transformers import (AutoTokenizer,
                          AutoConfig,
                          BatchEncoding,
                          PreTrainedTokenizer,
                          )
from transformers.tokenization_utils_base import TruncationStrategy

from model_wrapper import ModelWrapper
from globals import TokenIDS, GenerationType, CompletionDict

MAX_MODEL_INPUT_SIZE = 8192


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
    wrapped_model: ModelWrapper
    framework: str = "pt"
    answer_length_multiplier: float = 16
    descriptive_attrs = (
        "model_name",
        "generation_type",
        "answer_length_multiplier",
        "wrapped_model",
    )

    def __init__(
            self,
            model_name: str,
            group_size: int,
            temp: Optional[float] = None,
            end_of_sentence_stop: Optional[bool] = None,
            repetition_penalty_theta: Optional[float] = None,
            answer_length_multiplier: float = 16,
    ):
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
        repetition_penalty_theta: float
            the theta parameter for the repetition penalty computation.
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name)
        end_of_sentence_id = self.tokenizer.eos_token_id
        end_of_sentence_stop = end_of_sentence_stop and end_of_sentence_id is not None
        max_input_len = self.tokenizer.model_max_length
        max_len_is_huge = max_input_len > MAX_MODEL_INPUT_SIZE
        if max_len_is_huge or max_input_len is None:
            config = AutoConfig.from_pretrained(model_name)
            max_input_len = config.max_position_embeddings
            max_len_is_still_huge = max_input_len > MAX_MODEL_INPUT_SIZE
            if max_len_is_still_huge or max_input_len is None:
                raise ValueError(
                    "The maximum length of the model is too big"
                )
        self.wrapped_model = ModelWrapper(
            model_name=model_name,
            group_size=group_size,
            max_input_len=max_input_len,
            end_of_sentence_id=end_of_sentence_id,
            end_of_sentence_stop=end_of_sentence_stop,
            repetition_penalty_theta=repetition_penalty_theta,
            padding_id=self.padding_id,
            temp=temp,
            use_softmax=self.generation_type.requires_softmax(),
            vocab_size=self.tokenizer.vocab_size,
        )
        self.answer_length_multiplier = answer_length_multiplier

    @property
    def padding_id(self) -> int:
        padding_id = self.tokenizer.pad_token_id
        if not isinstance(padding_id, int):
            padding_id = self.tokenizer.unk_token_id
        if not isinstance(padding_id, int):
            padding_id = self.tokenizer.mask_token_id
        if not isinstance(padding_id, int):
            padding_id = self.tokenizer.mask_token_id
        if not isinstance(padding_id, int):
            raise RuntimeError(f"padding_id is {padding_id} and its type is {type(padding_id)}")
        return padding_id

    @property
    @abstractmethod
    def generation_type(self) -> GenerationType:
        """A method that chooses the generation type
        Returns:
            a GenerationType object"""
        raise NotImplementedError

    def get_token_tensor(
            self,
            text: str,
            truncation: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
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
            truncation=truncation,
            max_length=self.wrapped_model.max_input_len,
        )
        if isinstance(tokenized_text, dict) or isinstance(tokenized_text, BatchEncoding):
            token_tensor: LongTensor = tokenized_text["input_ids"]
            # O(1) because we are accessing a single element in a dictionary and saving the reference to it.
        elif isinstance(tokenized_text, LongTensor):
            token_tensor: LongTensor = tokenized_text
            # O(1) because we are saving the reference to the tensor
        else:
            raise TypeError("The tokenizer output is not one of:"
                            "dict, BatchEncoding, LongTensor")

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

        prefix_tokens: LongTensor = self.get_token_tensor(prefix, truncation)
        # O(A) where A is the number of characters in the prefix.
        postfix_tokens: LongTensor = self.get_token_tensor(postfix, truncation)
        # O(B) where B is the number of characters in the postfix.
        prompt_tokens: LongTensor = self.get_token_tensor(prompt, truncation)
        # O(C) where C is the number of characters in the prompt.
        token_tensor: LongTensor = cat((prefix_tokens, prompt_tokens, postfix_tokens))
        # O( + b + c) where 'a' is the number of tokens in the prefix.
        # 'b' is the number of tokens in the prompt.
        # 'c' is the number of tokens in the postfix.
        # we know that the number of tokens is less than the number of characters
        return token_tensor, len(prefix_tokens), len(prompt_tokens), len(postfix_tokens)

    @abstractmethod
    def _forward(
            self,
            tokenized_prompt: LongTensor,
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
    ) -> CompletionDict | List[CompletionDict] | List[List[CompletionDict]]:
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

        if max_new_tokens is None and not self.wrapped_model.end_of_sentence_stop:
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
        tokens: LongTensor
        prefix_len: int
        postfix_len: int
        prompt_len: int

        tokens, prefix_len, prompt_len, postfix_len = self.preprocess(
            prompt=prompt_s, prefix=prefix, truncation=truncation, postfix=postfix)
        # O(len(prompt) + len(prefix) + len(postfix))

        if max_new_tokens is None:
            max_new_tokens = int(prompt_len * self.answer_length_multiplier)
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

    def __repr__(self):
        return f"{self.__class__.__name__}" \
               f"{'/n'.join(f'{attr}={getattr(self, attr)}' for attr in self.descriptive_attrs)}"

    def __str__(self):
        return repr(self)

    def as_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation of the generator
        such that it can be saved and loaded using the from_dict method"""
        return {key: getattr(self, key) for key in self.descriptive_attrs}

    @classmethod
    def from_dict(cls, my_dict: Dict[str, Any]):
        """Creates an TextGenerator from a dictionary
        The dictionary should have the same format as the dictionary returned by the as_dict method"""
        if "generation_type" in my_dict.keys():
            my_dict.pop("generation_type")
        wrapped_model: ModelWrapper = my_dict.pop("wrapped_model")
        wrapped_model_dict = wrapped_model.as_dict()
        my_dict.update(wrapped_model_dict)
        return cls(**my_dict)


class NoCompletionsFound(Exception):
    def __init__(
            self,
            curr_text_generator: TextGenerator,
            additional_info: str = ""):
        super(NoCompletionsFound, self).__init__(
            f"text generator: {curr_text_generator} \n"
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
