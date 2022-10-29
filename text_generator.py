import timeit
from enum import Enum
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Optional, List, Union, Dict, Tuple, Any

from torch import LongTensor, ones, cuda, tensor
from torch.nn import Softmax
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          AutoConfig,
                          BatchEncoding,
                          PreTrainedTokenizer,
                          PreTrainedModel)
from transformers.tokenization_utils_base import TruncationStrategy

TokenIDS = Union[List[int], Tuple[int]]
SingleAnswer = Dict[str, Union[str, tensor]]


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
    maximum_length: int
    framework: str = "pt"

    def __init__(self, model_name: str, group_size: int,
                 temp: float = 1.0,
                 end_of_sentence_stop: bool = False):
        """Model name: the name of the model
        used for loading from hugging face hub
        group size: int
        the number of tokens to be predicted at each model call
        temp: float
        temperature parameter for the softmax function"""
        self.model_name = model_name
        config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name)
        if cuda.is_available():
            self.model = self.model.cuda()
        self.vocab_size = config.vocab_size
        self.temp = temp
        self.group_size = group_size
        pad_id = self.tokenizer.pad_token_id
        self.end_of_sentence_id = self.tokenizer.eos_token_id
        if not isinstance(pad_id, int):
            pad_id = 0
        self.padding_tokens = [pad_id] * (self.group_size - 1)
        self.end_of_sentence_stop = end_of_sentence_stop
        self.maximum_length = self.tokenizer.model_max_length

    def set_group_size(self, new_group_size):
        self.group_size = new_group_size
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0
        self.padding_tokens = [pad_id] * (self.group_size - 1)

    def get_prob_mat(self, token_list: List[int]) \
            -> List[List[float]]:
        """Returns the probability matrix
         as a list of lists of floats"""

        attention_len = len(token_list) + self.group_size - 1

        padded_token_list = token_list + self.padding_tokens
        if len(padded_token_list) > self.maximum_length:
            padded_token_list = padded_token_list[-self.maximum_length:]
            attention_len = self.maximum_length
        longer_token_tensor = LongTensor([padded_token_list])
        attention_mask = ones([1, attention_len])
        if cuda.is_available():
            longer_token_tensor = longer_token_tensor.cuda()
            attention_mask = attention_mask.cuda()
        inputs = {"input_ids": longer_token_tensor,
                  "attention_mask": attention_mask,
                  "labels": longer_token_tensor}

        try:
            outputs = self.model(**inputs)
        except RuntimeError as e:
            print("The inputs that caused the error:")
            print(map(lambda x: x.to_list(), inputs.values()))
            raise e

        logits_tensor = outputs.logits.squeeze(0) / self.temp

        if cuda.is_available():
            logits_tensor_copy = logits_tensor
            logits_tensor_copy.cpu()
            prob_tensor = Softmax(dim=1)(logits_tensor_copy)
        else:
            prob_tensor = Softmax(dim=1)(logits_tensor)
        if self.group_size <= prob_tensor.shape[0]:
            prob_tensor = prob_tensor[-self.group_size:, :]
            prob_mat = [prob_tensor[i, :].tolist()
                        for i in range(self.group_size)]
        else:
            print("Warning: the group size is bigger than")
            print("the length of the model's output")
            print("If the length of the model input is n,")
            print("n will be length of the model's output")
            num_tokens = self.group_size - prob_tensor.shape[0]
            print(f"the predicted text will be \
                {num_tokens} tokens shorter")
            prob_mat = [prob_tensor[i, :].tolist()
                        for i in range(prob_tensor.shape[0])]
        if (not self.end_of_sentence_stop) and self.end_of_sentence_id is not None:
            for prob_vec in prob_mat:
                prob_vec[self.end_of_sentence_id] = 0.0
        return prob_mat

    def preprocess(
            self,
            prompt: str,
            prefix: str = "",
            truncation: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
    ) -> List[int]:
        """A helper method for __call__ that tokenize the prompt
        all the arguments are sent directly from the __call__ method
        Returns:
            the tokenized prompt as a list of ints"""

        prompt = prefix + prompt
        tokenizer_output = self.tokenizer(
            prompt,
            return_tensors=self.framework,
            padding=False,
            add_special_tokens=False,
            truncation=truncation,
            max_length=self.maximum_length,
        )
        is_dict = isinstance(tokenizer_output, dict)
        is_batch_encoding = isinstance(tokenizer_output,
                                       BatchEncoding)
        if is_dict or is_batch_encoding:
            token_tensor = tokenizer_output["input_ids"]
        else:
            token_tensor = tokenizer_output

        while token_tensor.shape[0] == 1:
            token_tensor = token_tensor[0]

        tokenized_prompt: List[int]
        tokenized_prompt = token_tensor.tolist()
        return tokenized_prompt

    @abstractmethod
    def _forward(
            self,
            tokenized_prompt: List[int],
            num_new_tokens: Optional[int] = None,
            num_return_sequences: int = 1,
    ) -> List[TokenIDS]:
        """A helper method for __call__ that generates the new tokens
        Has a unique implementation for each subclass
        Args:
            tokenized_prompt: List[int] - the tokenized prompt from the preprocess method
            num_new_tokens: int - the number of new tokens to generate from the __call__ method
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
            clean_up_tokenization_spaces: bool
    ):

        """A helper method for __call__ that converts the token ids to dictionary
        all the arguments are sent directly from the __call__ method
        token_ids - the token ids from the _forward method
        prompt_len - the length of the tokenized prompt
        the rest of the arguments are the arguments from the __call__ method
        """
        if num_new_tokens is None:
            shorten_token_list = token_ids
        else:
            final_num_tokens = prompt_len + num_new_tokens
            if len(token_ids) > final_num_tokens:
                shorten_token_list = token_ids[:final_num_tokens]
            else:
                shorten_token_list = token_ids
        if not return_full_text:
            shorten_token_list = shorten_token_list[prompt_len:]
        final_ans = {}
        if return_tensors:
            final_ans["generated_token_ids"] = tensor(shorten_token_list)
        if return_text:
            final_ans["generated_text"] = self.tokenizer.decode(
                shorten_token_list,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces
            )
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
    ) -> Union[SingleAnswer, List[SingleAnswer], List[List[SingleAnswer]]]:
        """The function that outside code should call to generate text
        Args:
            prompt_s: str or list of str - the prompt(s) to start the generation from
                (the text given by the user)
                if many prompts are given as a list,
                the function process each one independently and returns them as a list.
                (the same as calling [__call__(prompt, *args, **kwargs) for prompt in prompts])])
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
                truncation=truncation) for prompt in prompt_s]

        curr_token_list: List[int] = self.preprocess(
            prompt=prompt_s, prefix=prefix, truncation=truncation)

        prompt_len: int = len(curr_token_list)
        tokenized_answers: List[TokenIDS]
        tokenized_answers = self._forward(
            curr_token_list,
            max_new_tokens,
            num_return_sequences
        )

        if num_return_sequences > 1:
            return [self.postprocess(
                token_ids=tokenized_answer,
                num_new_tokens=max_new_tokens,
                prompt_len=prompt_len,
                return_text=return_text,
                return_tensors=return_tensors,
                return_full_text=return_full_text,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces
            ) for tokenized_answer in tokenized_answers]
        else:
            return self.postprocess(
                token_ids=tokenized_answers[0],
                num_new_tokens=max_new_tokens,
                prompt_len=prompt_len,
                return_text=return_text,
                return_tensors=return_tensors,
                return_full_text=return_full_text,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces
            )

    @abstractmethod
    def __repr__(self):
        pass

    def __str__(self):
        return repr(self)

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass


class NoCompletionsFound(Exception):
    def __init__(
            self,
            text_generator: TextGenerator,
            additional_info: str = ""):
        super(NoCompletionsFound, self).__init__(
            f"text generator: {text_generator} \n"
            f"additional info: {additional_info}")


def compare_generators(non_grouped_generator: TextGenerator, prompt: str, num_tokens: int, group_size: int):
    """Compares grouped and non-grouped text generators"""
    print(f"Your prompt:")
    print(prompt)

    start_non_grouped = timeit.default_timer()
    non_grouped_ans: str = non_grouped_generator(prompt_s=prompt, max_new_tokens=num_tokens)["generated_text"]
    stop_non_grouped = timeit.default_timer()
    non_grouped_time = stop_non_grouped - start_non_grouped
    print(f"Text generated by Non grouped sampling in {non_grouped_time} seconds:")
    print(non_grouped_ans)

    non_grouped_generator.set_group_size(group_size)
    grouped_generator = non_grouped_generator
    start_grouped_generation = timeit.default_timer()
    grouped_ans: str = grouped_generator(prompt_s=prompt, max_new_tokens=num_tokens)["generated_text"]
    stop_grouped_generation = timeit.default_timer()
    grouped_time = stop_grouped_generation - start_grouped_generation
    print(f"Text generated by grouped sampling in {grouped_time} seconds:")
    print(grouped_ans)
