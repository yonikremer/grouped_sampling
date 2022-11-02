import itertools
from collections.abc import Iterator
from random import choices, seed
from typing import Callable, List, Dict, Optional, Any, Set

from torch import Tensor

from text_generator import TextGenerator, GenerationType, NoCompletionsFound


class ChangingSeed(Iterator):
    """Context manager for changing the seed of the random module.
    How to use:
    with ChangingSeed(first_seed, number_of_different_seeds) as changing_seed:
        for _ in changing_seed:
            # do something with random module"""
    default_seed: int
    curr_seed: int
    max_num_calls: int
    curr_num_calls: int

    def __init__(self, default_seed: int, max_num_calls: int):
        self.default_seed = default_seed
        self.curr_seed = self.default_seed
        self.max_num_calls = max_num_calls
        self.curr_num_calls = 0

    def __enter__(self):
        self.curr_num_calls = 0
        self.curr_seed = self.default_seed
        return self

    def __exit__(self, *args):
        self.curr_seed = self.default_seed
        seed(self.default_seed)

    def __iter__(self):
        self.curr_seed = self.default_seed
        self.default_seed = self.default_seed
        return self

    def __next__(self):
        self.curr_seed += 1
        seed(self.curr_seed)
        self.curr_num_calls += 1
        if self.curr_num_calls > self.max_num_calls:
            raise StopIteration


class SamplingGenerator(TextGenerator):
    """A TextGenerator that generates text
    using random sampling
    with top-k or top-p filtering."""
    filter_tokens: Callable[[Dict[int, float]], Dict[int, float]]
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    default_seed: int = 0
    seed(default_seed)

    def __init__(self, model_name: str, group_size: int,
                 temp: float = 1.0, top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 end_of_sentence_stop: bool = False,
                 answer_length_multiplier: int = 16,):
        super().__init__(
            model_name=model_name,
            group_size=group_size,
            temp=temp,
            end_of_sentence_stop=end_of_sentence_stop,
            answer_length_multiplier=answer_length_multiplier,
        )
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
            self, prob_mat: Tensor,
            org_used_tokens: List[int]) -> List[int]:
        """Generates a group of tokens
         using the choice_function."""
        prob_mat.cpu()
        used_tokens: Set[int] = set(org_used_tokens)
        new_group = []
        for curr_token_probs in prob_mat:
            curr_token_probs: Tensor
            for used_token in used_tokens:
                curr_token_probs[used_token] = 0.0
            # for tokens with index >= self.vocab_size
            if len(curr_token_probs) > self.vocab_size:
                curr_token_probs = curr_token_probs[:self.vocab_size]

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
            used_tokens.add(sampled_token)
        return new_group

    def _forward(
            self,
            tokenized_prompt: Tensor,
            num_new_tokens: Optional[int] = None,
            num_return_sequences: int = 1,
    ) -> List[List[int]]:
        answers: List[List[int]] = []
        curr_token_list: List[int] = tokenized_prompt.tolist()
        for _ in ChangingSeed(
                default_seed=self.default_seed,
                max_num_calls=num_return_sequences):
            if num_new_tokens is None:
                raise RuntimeError("num_new_tokens is None")
            else:
                num_groups = num_new_tokens // self.group_size
                the_range = range(num_groups)
            for _ in the_range:
                prob_mat: Tensor = self.get_prob_mat(curr_token_list)
                new_tokens = self.generate_group(
                    prob_mat, curr_token_list)
                if self.end_of_sentence_id in new_tokens:
                    end_of_sentence_index = new_tokens.index(self.end_of_sentence_id)
                    new_tokens = new_tokens[:end_of_sentence_index]
                curr_token_list.extend(new_tokens)
            answers.append(curr_token_list)
        return answers

    def __repr__(self):
        return f"SamplingGenerator: " \
               f"model name: {self.model_name}, " \
               f"group size: {self.group_size}, " \
               f"temperature: {self.temp}, " \
               f"generation type: {self.generation_type}, " \
               f"top_p: {self.top_p}, " \
               f"top_k: {self.top_k}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "group_size": self.group_size,
            "temperature": self.temp,
            "generation_type": self.generation_type,
            "top_p": self.top_p,
            "top_k": self.top_k,

        }
