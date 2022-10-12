from random import choices, seed
from copy import deepcopy
from math import ceil
from typing import Callable, List, Dict, Optional

from transformers import BatchEncoding
from torch import tensor

from text_generator import TextGenerator, get_second_item, GenerationType, NoCompletionsFound


class SamplingGenerator(TextGenerator):
    """A TextGenerator that generates text 
    using random sampling
    with top-k or top-p filtering."""
    filter_tokens: Callable[[Dict[int, float]], Dict[int, float]]
    top_k: Optional[int] = None
    top_p: Optional[float] = None

    def __init__(self, model_name: str, group_size: int,
                 temp: float = 1.0, top_k: Optional[int] = None,
                 top_p: Optional[float] = None):
        super().__init__(model_name, group_size, temp)
        seed(0)
        if top_k == 1 or top_p == 0:
            self.generation_type = GenerationType.GREEDY
            self.filter_tokens = SamplingGenerator.highest_prob_token
        elif top_p is None and top_k is not None:
            self.top_k = top_k
            self.filter_tokens = self.top_k_tokens
            self.generation_type = GenerationType.TOP_K
        elif top_k is None and top_p is not None:
            self.top_p = top_p
            self.filter_tokens = self.top_p_tokens
            self.generation_type = GenerationType.TOP_P
        else:
            raise ValueError("Either top_k or top_p \
                              should be set.")

    @staticmethod
    def highest_prob_token(sorted_probs: Dict[int, float]) \
            -> Dict[int, float]:
        """Gets a token id: probability mapping
        as a sorted dictionary
        returns the token with the highest probability.
        as a {token: 1.0} dictionary."""
        highest_prob_token_id: int = next(iter(sorted_probs))
        return {highest_prob_token_id: 1.0}

    def top_p_tokens(self,
                     sorted_probs: Dict[int, float]) \
            -> Dict[int, float]:
        """Gets a token id: probability mapping
        returns the tokens with the highest probability
        such that their sum is <= self.top_p.
        or the token with the highest probability
        if it's higher than top_p."""
        top_p_probs: Dict[int, float] = {}
        prob_sum: float = 0.0
        for i, (curr_token, curr_prob) \
                in enumerate(sorted_probs.items()):
            if i <= 0 or curr_prob <= self.top_p:
                prob_sum += curr_prob
                top_p_probs[curr_token] = curr_prob
                continue
            break
        if prob_sum == 0:
            raise NoCompletionsFound(self,
                                     "The probabilities of all the tokens "
                                     "is (rounded to) 0 . "
                                     "Please try a higher temperature.")
        weighted_probs = {k: v / prob_sum
                          for k, v in top_p_probs.items()}
        return weighted_probs

    def top_k_tokens(self, sorted_probs: Dict[int, float])\
            -> Dict[int, float]:
        """Gets a token id: probability mapping
        returns the TOP_K tokens
        with the highest probability."""

        keys_list = list(sorted_probs.keys())
        sorted_top_k_keys = keys_list[:self.top_k]
        top_k_probs: Dict[int, float]
        top_k_probs = {k: sorted_probs[k]
                       for k in sorted_top_k_keys}
        prob_sum = sum(top_k_probs.values())
        weighted_probs = {k: v / prob_sum
                          for k, v in top_k_probs.items()}
        return weighted_probs

    def add_group(self, prob_mat: List[List[float]],
                  org_used_tokens: List[int]) -> List[int]:
        """Generates a group of tokens
         using the choice_function."""
        used_tokens = deepcopy(org_used_tokens)
        answer = []
        for curr_token_probs in prob_mat:
            for used_token in used_tokens:
                curr_token_probs[used_token] = 0.0

            indexed_prob: Dict[int, float]
            indexed_prob = {i: prob for i, prob
                            in enumerate(curr_token_probs)}
            # O(vocab_size)
            items = indexed_prob.items()
            sorted_probs: Dict[int, float]
            # noinspection PyTypeChecker
            sorted_probs = dict(sorted(items,
                                       key=get_second_item,
                                       reverse=True))
            weighted_probs = self.filter_tokens(sorted_probs)
            keys_list = list(weighted_probs.keys())
            weights_list = list(weighted_probs.values())
            sampled_token = choices(keys_list,
                                    weights_list,
                                    k=1)[0]
            answer.append(sampled_token)
            used_tokens.append(sampled_token)
        return answer

    def __call__(self, prompt: str, num_new_tokens: int) -> str:
        if num_new_tokens == 0:
            return prompt
        num_groups = ceil(num_new_tokens / self.group_size)
        tokenizer_output = self.tokenizer(prompt,
                                          return_tensors="pt")
        is_dict = isinstance(tokenizer_output, dict)
        is_batch_encoding = isinstance(tokenizer_output,
                                       BatchEncoding)
        if is_dict or is_batch_encoding:
            token_tensor = tokenizer_output["input_ids"]
        else:
            token_tensor = tokenizer_output
        tokenizer_output: tensor

        curr_token_list: list[int] = token_tensor.squeeze().tolist()
        prompt_len: int = len(curr_token_list)

        for _ in range(num_groups):
            prob_mat = self.get_prob_mat(None,
                                         curr_token_list)
            new_tokens = self.add_group(prob_mat,
                                        curr_token_list)
            curr_token_list.extend(new_tokens)

        final_num_tokens = prompt_len + num_new_tokens
        shorten_token_list = curr_token_list[:final_num_tokens]
        final_ans = self.tokenizer.decode(
            shorten_token_list,
            skip_special_tokens=True)
        return final_ans

    def __repr__(self):
        return f"SamplingGenerator: " \
               f"model name: {self.model_name}, " \
               f"group size: {self.group_size}, " \
               f"temperature: {self.temp}, " \
               f"generation type: {self.generation_type}, " \
               f"top_p: {self.top_p}, " \
               f"top_k: {self.top_k}"
