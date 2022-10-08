from enum import Enum
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Optional, List

from torch import LongTensor, ones, no_grad, Tensor, cuda
from torch.nn import Softmax
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          AutoConfig,
                          PreTrainedTokenizer,
                          PreTrainedModel)


def get_second_item(sliceable):
    return sliceable[1]


class GenerationType(Enum):
    """The type of generation to use"""
    GREEDY = "greedy"
    TOP_K = "top_k"
    TOP_P = "top_p"
    TREE = "tree"


class TextGenerator(Callable, ABC):
    """An abstract base class for
    A callable object that given a prompt
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

    def __init__(self, model_name: str, group_size: int,
                 temp: float = 1.0):
        """Model name: the name of the model
        used for loading from hugging face hub
        group size: int
        the number of tokens to be predicted at each model call
        temp: float
        temperature parameter for the softmax function"""
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if cuda.is_available():
            self.model = self.model.cuda()
        config = AutoConfig.from_pretrained(model_name)
        self.vocab_size = config.vocab_size
        self.temp = temp
        self.group_size = group_size
        pad_id = self.tokenizer.pad_token_id
        if not isinstance(pad_id, int):
            pad_id = 0
        self.padding_tokens = [pad_id] * (self.group_size - 1)

    def get_prob_mat(self, prompt: Optional[str],
                     token_list: Optional[List[int]])\
            -> List[List[float]]:
        """Returns the probability matrix
         as a list of lists of floats"""
        attention_len = len(token_list) + self.group_size - 1
        if token_list is None:
            tokenized_prompt = self.tokenizer(prompt,
                                              return_tensors="pt")
            if isinstance(tokenized_prompt, list):
                token_list = tokenized_prompt
            elif isinstance(tokenized_prompt, Tensor):
                token_list = tokenized_prompt.tolist()
            else:
                token_list = tokenized_prompt["input_ids"]

        longer_token_list = token_list + self.padding_tokens
        longer_token_tensor = LongTensor([longer_token_list])
        attention_mask = ones([1, attention_len])
        inputs = {"input_ids": longer_token_tensor,
                  "attention_mask": attention_mask}
        if cuda.is_available():
            inputs = {name: tensor.cuda()
                      for name, tensor in inputs.items()}

        if not isinstance(inputs, dict):
            with no_grad():
                outputs = self.model(**inputs)
        elif "input_ids" not in inputs.keys():
            with no_grad():
                outputs = self.model(**inputs)
        else:
            with no_grad():
                outputs = self.model(**inputs,
                                     labels=inputs["input_ids"])

        logits_tensor = outputs.logits.squeeze(0) / self.temp

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

        return prob_mat

    @abstractmethod
    def __call__(self, prompt: str, num_new_tokens: int) -> str:
        pass
