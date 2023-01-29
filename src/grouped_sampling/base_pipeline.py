from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union

from torch import LongTensor, Tensor
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
)

from .completion_dict import CompletionDict
from .generation_type import GenerationType
from .generation_utils import GroupedGenerationUtils
from .postprocessor import PostProcessor
from .preprocessor import PreProcessor
from .repetition_penalty import DEFAULT_REPETITION_PENALTY, RepetitionPenaltyStrategy
from .token_ids import TokenIDS

MAX_MODEL_INPUT_SIZE = 32768


def remove_nones(d: Dict[str, Any]) -> Dict[str, Any]:
    """Returns a copy of a dictionary with all the not None values"""
    return {key: d[key] for key in d.keys() if d[key] is not None}


def get_padding_id(tokenizer: PreTrainedTokenizer):
    padding_id = tokenizer.pad_token_id
    if not isinstance(padding_id, int):
        padding_id = tokenizer.mask_token_id
    if not isinstance(padding_id, int):
        padding_id = tokenizer.unk_token_id
    if not isinstance(padding_id, int):
        raise RuntimeError(
            f"padding_id is {padding_id} and its type is {type(padding_id)}")
    return padding_id


class GroupedGenerationPipeLine(Callable, ABC):
    """An abstract base class for
    A callable object that given a func_prompt
     and length of wanted answer,
    generates text
    the text pipeline has a model,
    and some parameters
    (Defined in the subclasses)"""

    framework: str = "pt"
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
        repetition_penalty_strategy:
        RepetitionPenaltyStrategy = DEFAULT_REPETITION_PENALTY,
        answer_length_multiplier: float = 16,
        max_batch_size: int = 32,
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
        repetition_penalty_strategy: RepetitionPenaltyStrategy
            The strategy for the repetition penalty
        max_batch_size: int
            The maximum batch size to be used
        """
        self.answer_length_multiplier: float = answer_length_multiplier
        self.max_batch_size: int = max_batch_size
        self.model_name: str = model_name
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        end_of_sentence_id = tokenizer.eos_token_id
        end_of_sentence_stop = end_of_sentence_stop and end_of_sentence_id is not None
        max_input_len = tokenizer.model_max_length
        self.pre_processing_strategy: PreProcessor = PreProcessor(
            tokenizer=tokenizer,
            max_input_len=max_input_len,
        )
        self.post_processing_strategy: PostProcessor = PostProcessor(
            tokenizer=tokenizer, )
        wrapped_model_kwargs: Dict[str, Any] = {
            "model_name": model_name,
            "group_size": group_size,
            "max_input_len": max_input_len,
            "end_of_sentence_id": end_of_sentence_id,
            "end_of_sentence_stop": end_of_sentence_stop,
            "repetition_penalty_strategy": repetition_penalty_strategy,
            "padding_id": get_padding_id(tokenizer),
            "temp": temp,
            "use_softmax": self.generation_type.requires_softmax(),
            "vocab_size": tokenizer.vocab_size,
        }
        self.wrapped_model: GroupedGenerationUtils = GroupedGenerationUtils(
            **remove_nones(wrapped_model_kwargs))

    @property
    @abstractmethod
    def generation_type(self) -> GenerationType:
        """A method that chooses the generation type
        Returns:
            a GenerationType object"""

    @abstractmethod
    def _forward(
        self,
        tokenized_prompt: LongTensor,
        num_new_tokens: int,
    ) -> TokenIDS:
        """A helper method for __call__ that generates the new tokens
        Has a unique implementation for each subclass
        Args:
            tokenized_prompt: List[int]
                the tokenized prompt from the preprocess method
            num_new_tokens: int - the number of new tokens to generate
                from the __call__ method
        Returns:
            the prompt + generated text as a list/tuple of ints"""

    def __call__(
        self,
        prompt_s: Union[str, Iterable[str]],
        max_new_tokens: Optional[int] = None,
        return_tensors: bool = False,
        return_text: bool = True,
        return_full_text: bool = True,
        clean_up_tokenization_spaces: bool = False,
        prefix: str = "",
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
            max_new_tokens: Optional[int] > 0
             - the number of tokens to generate
                if None, the function will generate tokens
                 until one of them is the end of sentence token
            return_tensors: bool - whether to return the generated token ids
            return_text: bool - whether to return the generated string
            return_full_text: bool - whether to return the full text
                (prompt + generated text)
                (if false, it will return only the generated text)
            clean_up_tokenization_spaces: bool
             - whether to clean up tokenization spaces
                This parameter is forwarded to the decode function of the AutoTokenizer class
            prefix (`str`, defaults to an empty string):
                Prefix added to prompt.
            postfix: str
             - a postfix to add to the prompt
        Returns:
            Each result comes as a dictionary with the following keys:
            - "generated_text"
            (`str`, present when `return_text=True`)
             -- The generated text.
            - "generated_token_ids" (
            `torch.tensor`, present when `return_tensors=True`)
             -- The token
              ids of the generated text.
        """
        if max_new_tokens is None and not self.wrapped_model.end_of_sentence_stop:
            raise ValueError(
                "max_new_tokens must be given if end_of_sentence_stop is False"
            )
        if isinstance(prompt_s, str):
            return self.call_single_prompt(
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                max_new_tokens=max_new_tokens,
                postfix=postfix,
                prefix=prefix,
                prompt=prompt_s,
                return_full_text=return_full_text,
                return_tensors=return_tensors,
                return_text=return_text,
            )

        # split the prompts into batches
        prompts_batches: Iterable[List[str], None,
                                  None] = self.split_to_batches(prompt_s)
        answers: List[CompletionDict] = []
        # generate the batches
        for prompts_batch in prompts_batches:
            prompts_batch: List[str]
            if len(prompts_batch) == 1:
                answers.append(
                    self.call_single_prompt(
                        prompt=prompts_batch[0],
                        max_new_tokens=max_new_tokens,
                        return_tensors=return_tensors,
                        return_text=return_text,
                        return_full_text=return_full_text,
                        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                        prefix=prefix,
                        postfix=postfix,
                    ))
            else:
                answers.extend(
                    self.call_batch(
                        prompts_batch,
                        max_new_tokens=max_new_tokens,
                        return_tensors=return_tensors,
                        return_text=return_text,
                        return_full_text=return_full_text,
                        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                        prefix=prefix,
                        postfix=postfix,
                    ))
        return answers

    def call_single_prompt(
        self,
        clean_up_tokenization_spaces,
        max_new_tokens,
        postfix,
        prefix,
        prompt,
        return_full_text,
        return_tensors,
        return_text,
    ) -> CompletionDict:
        tokens: LongTensor
        prefix_len: int
        postfix_len: int
        prompt_len: int
        tokens, prefix_len, prompt_len, postfix_len = self.pre_processing_strategy(
            prompt=prompt, prefix=prefix, postfix=postfix)
        # O(len(prompt) + len(prefix) + len(postfix))
        if max_new_tokens is None:
            max_new_tokens = int(prompt_len * self.answer_length_multiplier)
            # O(1)
        tokenized_answers: List[TokenIDS]
        tokenized_answers = self._forward(
            tokens,
            max_new_tokens,
        )
        # O(len(tokenized_answers[0]))
        return self.post_processing_strategy(
            token_ids=tokenized_answers,
            num_new_tokens=max_new_tokens,
            prompt_len=prompt_len,
            return_text=return_text,
            return_tensors=return_tensors,
            return_full_text=return_full_text,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            prefix_len=prefix_len,
            postfix_len=postfix_len,
        )

    def call_batch(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
        return_tensors: bool = False,
        return_text: bool = True,
        return_full_text: bool = True,
        clean_up_tokenization_spaces: bool = False,
        prefix: str = "",
        postfix: str = "",
    ) -> List[CompletionDict]:
        """A helper for __call__ that handles the case when many prompts are given
        Args:
            prompts: List of strings - the prompts to start the generation from (the text given by the user)
            should satisfy: self.max_batch_size >= len(prompts) >= 0
            max_new_tokens: Optional[int] > 0 - the number of tokens to generate
             if None, the function will generate tokens until one of them is the end of sentence token
            return_tensors: bool - whether to return the generated token ids
            return_text: bool - whether to return the generated string
            return_full_text: bool - whether to return the full text
                (prompt + generated text)
                (if false, it will return only the generated text)
            clean_up_tokenization_spaces: bool - whether to clean up tokenization spaces
                This parameter is forwarded to the decode function of the AutoTokenizer class
            prefix: str Prefix added to prompt and not returned to the user,
                even if return_full_text is true.
            postfix: str - a postfix to add to the prompt and not returned to the user,
                even if return_full_text is true."""
        # check the parameters
        if not return_tensors and not return_text:
            raise ValueError(
                "You must return at least one of the return_tensors and return_text parameters"
            )
        if not isinstance(prompts, list):
            raise TypeError
        if len(prompts) > self.max_batch_size:
            raise ValueError("The maximum batch size is " +
                             str(self.max_batch_size))
        if len(prompts) == 1:
            return [
                self(
                    prompts[0],
                    max_new_tokens=max_new_tokens,
                    return_tensors=return_tensors,
                    return_text=return_text,
                    return_full_text=return_full_text,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                    prefix=prefix,
                    postfix=postfix,
                )
            ]
        if len(prompts) == 0:
            return []
        # tokenize the prompts
        tokenized_prompts_lengths: List[Tuple[Tensor, int, int, int]] = [
            self.pre_processing_strategy(prompt,
                                         prefix=prefix,
                                         postfix=postfix) for prompt in prompts
        ]
        tokenized_prompts: List[Tensor] = [
            tokenized_prompt
            for tokenized_prompt, _, _, _ in tokenized_prompts_lengths
        ]

        tokenized_prompts: List[Tensor] = [tokenized_prompt for tokenized_prompt, _, _, _ in tokenized_prompts_lengths]
        prefix_lengths: List[int] = [prefix_length for _, prefix_length, _, _ in tokenized_prompts_lengths]
        prompts_lengths: List[int] = [prompt_length for _, _, prompt_length, _ in tokenized_prompts_lengths]
        postfix_lengths: List[int] = [postfix_length for _, _, _, postfix_length in tokenized_prompts_lengths]

        if max_new_tokens is None:
            max_new_tokens = int(max(prompts_lengths) * self.answer_length_multiplier)

        # generate the sequences
        generated_sequences: List[List[List[int]]] = self.forward_batch(
            tokenized_prompts,
            num_new_tokens=max_new_tokens,
        )
        # post process the sequences
        return [
            self.post_processing_strategy(
                token_ids=generated_sequence,
                num_new_tokens=max_new_tokens,
                prefix_len=prefix_length,
                prompt_len=prompt_length,
                postfix_len=postfix_length,
                return_tensors=return_tensors,
                return_text=return_text,
                return_full_text=return_full_text,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            ) for generated_sequence, prefix_length, prompt_length,
            postfix_length in zip(
                generated_sequences,
                prefix_lengths,
                prompts_lengths,
                postfix_lengths,
            )
        ]

    def __repr__(self):
        attrs_description = ", ".join(f"{attr}={getattr(self, attr)}"
                                      for attr in self.descriptive_attrs)
        return f"{self.__class__.__name__}: " + attrs_description

    def __str__(self):
        return repr(self)

    def as_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation
         of the pipeline
        such that it can be saved and loaded
         using the from_dict method"""
        return {key: getattr(self, key) for key in self.descriptive_attrs}

    @classmethod
    def from_dict(cls, my_dict: Dict[str, Any]):
        """Creates an GroupedGenerationPipeLine from a dictionary
        The dictionary should have the same format
         as the dictionary returned by the as_dict method"""
        if "generation_type" in my_dict.keys():
            my_dict.pop("generation_type")
        wrapped_model: GroupedGenerationUtils = my_dict.pop("wrapped_model")
        wrapped_model_dict = wrapped_model.as_dict()
        my_dict.update(wrapped_model_dict)
        return cls(**my_dict)

    @abstractmethod
    def forward_batch(self, tokenized_prompts: List[LongTensor],
                      num_new_tokens: int) -> List[TokenIDS]:
        """Generates a batch of sequences"""
        return [
            self._forward(tokenized_prompt, num_new_tokens)
            for tokenized_prompt in tokenized_prompts
        ]

    def split_to_batches(
            self, prompts: Iterable[str]) -> Generator[List[str], None, None]:
        """Splits the prompts into batches"""
        curr_batch = []
        for prompt in prompts:
            if len(curr_batch) >= self.max_batch_size:
                yield curr_batch
                curr_batch = []
            curr_batch.append(prompt)
        yield curr_batch
