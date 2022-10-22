from random import choices, seed
from copy import deepcopy
from math import ceil
from typing import Callable, List, Dict, Optional, Union

from transformers import BatchEncoding
from torch import tensor

from text_generator import TextGenerator, GenerationType, NoCompletionsFound


class SamplingGenerator(TextGenerator):
    """A TextGenerator that generates text 
    using random sampling
    with top-k or top-p filtering."""
    filter_tokens: Callable[[Dict[int, float]], Dict[int, float]]
    top_k: Optional[int] = None
    top_p: Optional[float] = None

    def __init__(self, model_name: str, group_size: int,
                 temp: float = 1.0, top_k: Optional[int] = None,
                 top_p: Optional[float] = None, end_of_sentence_stop: bool = False):
        super().__init__(
            model_name=model_name,
            group_size=group_size,
            temp=temp,
            end_of_sentence_stop=end_of_sentence_stop)
        seed(0)
        if top_k is None and top_p is None:
            self.top_p = 1.0
            self.top_k = self.vocab_size
            self.generation_type = GenerationType.RANDOM
            self.filter_tokens = SamplingGenerator.all_tokens
        elif top_p is None and top_k is not None:
            self.top_k = top_k
            self.filter_tokens = self.top_k_tokens
            self.generation_type = GenerationType.TOP_K
        elif top_k is None and top_p is not None:
            self.top_p = top_p
            self.filter_tokens = self.top_p_tokens
            self.generation_type = GenerationType.TOP_P
        elif top_p == 1.0 or top_k >= self.vocab_size:
            self.top_p = 1.0
            self.top_k = self.vocab_size
            self.generation_type = GenerationType.RANDOM
            self.filter_tokens = SamplingGenerator.all_tokens
        elif top_k == 1 or top_p == 0:
            self.generation_type = GenerationType.GREEDY
            self.filter_tokens = SamplingGenerator.highest_prob_token
        else:
            raise ValueError(
                "Either top_k or top_p \
                should be set.")

    @staticmethod
    def all_tokens(sorted_probs: Dict[int, float]) \
            -> Dict[int, float]:
        """A filtering function that doesn't filter any tokens.
        returns all the tokens with their probabilities."""
        return sorted_probs

    @staticmethod
    def highest_prob_token(
            sorted_probs: Dict[int, float]) \
            -> Dict[int, float]:
        """Gets a token id: probability mapping
        as a sorted dictionary
        returns the token with the highest probability.
        as a {token: 1.0} dictionary."""
        highest_prob_token_id = next(iter(sorted_probs))
        return {highest_prob_token_id: 1.0}

    def top_p_tokens(
            self, sorted_probs: Dict[int, float]) \
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
            raise NoCompletionsFound(
                self, "The probabilities of all the tokens "
                      "is (rounded to) 0 . "
                      "Please try a higher temperature.")
        weighted_probs = {
            k: v / prob_sum
            for k, v in top_p_probs.items()}
        return weighted_probs

    def top_k_tokens(
            self, sorted_probs: Dict[int, float]) \
            -> Dict[int, float]:
        """Gets a token id: probability mapping
        returns the TOP_K tokens
        with the highest probability."""

        keys_list = list(sorted_probs.keys())
        sorted_top_k_keys = keys_list[:self.top_k]
        top_k_probs: Dict[int, float]
        top_k_probs = {
            k: sorted_probs[k]
            for k in sorted_top_k_keys}
        prob_sum = sum(top_k_probs.values())
        weighted_probs = {
            k: v / prob_sum
            for k, v in top_k_probs.items()}
        return weighted_probs

    def generate_group(
            self, prob_mat: List[List[float]],
            org_used_tokens: List[int]) -> List[int]:
        """Generates a group of tokens
         using the choice_function."""
        used_tokens = deepcopy(org_used_tokens)
        new_group = []
        for curr_token_probs in prob_mat:
            for used_token in used_tokens:
                curr_token_probs[used_token] = 0.0

            indexed_prob: Dict[int, float]
            indexed_prob = {
                i: prob for i, prob
                in enumerate(curr_token_probs)}
            # O(vocab_size)
            items = indexed_prob.items()
            sorted_probs: Dict[int, float]
            # noinspection PyTypeChecker
            sorted_probs = dict(sorted(
                items, key=lambda x: x[1],
                reverse=True))
            weighted_probs = self.filter_tokens(sorted_probs)
            keys_list = list(weighted_probs.keys())
            weights_list = list(weighted_probs.values())
            sampled_token: int = choices(
                keys_list, weights_list, k=1)[0]
            new_group.append(sampled_token)
            if sampled_token == self.end_of_sentence_id:
                return new_group
            used_tokens.append(sampled_token)
        return new_group

    def __call__(
            self,
            prompt: str,
            num_new_tokens: int,
            return_text: bool = True,
            return_tensors: bool = False,
            return_full_text: bool = True,
            clean_up_tokenization_spaces: bool = False
    ) -> Dict[str, Union[str, tensor]]:
        if num_new_tokens == 0:
            return prompt
        curr_token_list, prompt_len, num_groups = self.preprocess(
            prompt=prompt,
            num_new_tokens=num_new_tokens
        )

        for _ in range(num_groups):
            prob_mat = self.get_prob_mat(curr_token_list)
            new_tokens = self.generate_group(
                prob_mat, curr_token_list)
            if self.end_of_sentence_id in new_tokens:
                end_of_sentence_index = new_tokens.index(self.end_of_sentence_id)
                new_tokens = new_tokens[:end_of_sentence_index]
                curr_token_list.extend(new_tokens)
                break
            curr_token_list.extend(new_tokens)

        return self.postprocess(
            token_ids=curr_token_list,
            num_new_tokens=num_new_tokens,
            prompt_len=prompt_len,
            return_text=return_text,
            return_tensors=return_tensors,
            return_full_text=return_full_text,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces
        )

    def __repr__(self):
        return f"SamplingGenerator: " \
               f"model name: {self.model_name}, " \
               f"group size: {self.group_size}, " \
               f"temperature: {self.temp}, " \
               f"generation type: {self.generation_type}, " \
               f"top_p: {self.top_p}, " \
               f"top_k: {self.top_k}"
