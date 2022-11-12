import heapq
from collections.abc import Iterator
from random import seed
from typing import Callable, List, Dict, Optional, Any, Set

import torch
from torch import Tensor, tensor

from text_generator import TextGenerator, GenerationType, NoCompletionsFound


ProbDict = Dict[int, Tensor]


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
        torch.manual_seed(self.default_seed)

    def __iter__(self):
        self.curr_seed = self.default_seed
        return self

    def __next__(self):
        self.curr_seed += 1
        seed(self.curr_seed)
        torch.manual_seed(self.curr_seed)
        self.curr_num_calls += 1
        if self.curr_num_calls > self.max_num_calls:
            raise StopIteration


class TokenProb:
    """Class for storing the probability of a token and the token itself.
    Used to store the probabilities of the next tokens in the sampling generator.
    Is useful because it supports the < and > operators, which are used in the
    heapq module
    The < and > are the opposite of each other because the heapq module is only supporting minimum heaps
    and I need a maximum heap"""
    __slots__ = ['token_id', 'prob']
    token_id: int
    prob: Tensor  # of shape (1,) and dtype float

    def __init__(self, token_id: int, prob: Tensor):
        self.token_id = token_id
        self.prob = prob

    def __lt__(self, other: "TokenProb"):
        """Overrides the < operator
        Comparison is done by the probability"""
        return self.prob > other.prob

    def __gt__(self, other: "TokenProb"):
        """Overrides the > operator
        Comparison is done by the probability"""
        return self.prob < other.prob


class SamplingGenerator(TextGenerator):
    """A TextGenerator that generates text
    using random sampling
    with top-k or top-p filtering."""
    filter_tokens: Callable[[ProbDict], ProbDict]
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    default_seed: int = 0
    seed(default_seed)
    torch.manual_seed(default_seed)

    def __init__(self, model_name: str, group_size: int,
                 temp: float = 1.0, top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 end_of_sentence_stop: bool = False,
                 answer_length_multiplier: int = 16,):
        self.top_p, self.top_k = top_p, top_k
        super().__init__(
            model_name=model_name,
            group_size=group_size,
            temp=temp,
            end_of_sentence_stop=end_of_sentence_stop,
            answer_length_multiplier=answer_length_multiplier,
        )
        gen_type_to_filter_method = {
            GenerationType.TOP_K: self.top_k_tokens,
            GenerationType.TOP_P: self.top_p_tokens,
            GenerationType.GREEDY: SamplingGenerator.highest_prob_token,
            GenerationType.RANDOM: SamplingGenerator.all_tokens,
        }
        self.filter_tokens = gen_type_to_filter_method[self.generation_type]

    def choose_generation_type(self) -> GenerationType:
        top_k = self.top_k
        top_p = self.top_p
        if top_k is None and top_p is None:
            return GenerationType.RANDOM
        if top_k == 1 or top_p == 0.0:
            return GenerationType.GREEDY
        if top_k is not None:
            if top_k < self.vocab_size:
                return GenerationType.TOP_K
            return GenerationType.RANDOM
        if top_p is not None:
            if top_p < 1.0:
                return GenerationType.TOP_P
            return GenerationType.RANDOM
        raise RuntimeError

    @staticmethod
    def all_tokens(probs: ProbDict) \
            -> ProbDict:
        """A filtering function that doesn't filter any tokens.
        returns all the tokens with their probabilities."""
        return probs

    @staticmethod
    def highest_prob_token(
            probs: ProbDict) \
            -> ProbDict:
        """Gets a token id: probability mapping
        as a sorted dictionary
        returns the token with the highest probability.
        as a {token: 1.0} dictionary."""
        highest_prob_token_id: int = max(probs, key=probs.get)
        return {highest_prob_token_id: tensor(1.0)}

    def top_p_tokens(
            self, probs: ProbDict) \
            -> ProbDict:
        """Gets a token id: probability mapping
        returns the tokens with the highest probability
        such that their sum is <= self.top_p.
        or the token with the highest probability
        if it's higher than top_p.
        assuming the values of the dict are between 0 and 1, inclusive
        this is the bottleneck of the sampling generator."""
        top_p_probs: ProbDict = {}
        prob_sum: float = 0.0
        converted_probs: List[TokenProb]
        converted_probs = [TokenProb(token_id, prob) for token_id, prob in probs.items()]
        heapq.heapify(converted_probs)
        while prob_sum < self.top_p and len(converted_probs) > 0:
            curr_prob: TokenProb = heapq.heappop(converted_probs)
            token_id = curr_prob.token_id
            prob_sum += probs[token_id]
            top_p_probs[token_id] = probs[token_id]
        if len(top_p_probs) == 0:
            raise NoCompletionsFound(
                self, "The probabilities of all the tokens "
                      "is (rounded to) 0 . "
                      "Please try a higher temperature.")
        weighted_probs = {
            k: v / prob_sum
            for k, v in top_p_probs.items()}
        return weighted_probs

    def top_k_tokens(
            self, probs: ProbDict) \
            -> ProbDict:
        """Gets a token id: probability mapping
        returns the TOP_K tokens
        with the highest probability.
        this is the bottleneck of the sampling generator."""
        top_k_keys: List[int] = heapq.nlargest(self.top_k, probs, key=probs.get)
        top_k_probs = {
            k: probs[k]
            for k in top_k_keys}
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
        new_group: List[int] = []
        for curr_token_probs in prob_mat:
            curr_token_probs: Tensor
            for used_token in used_tokens:
                curr_token_probs[used_token] = 0.0
            # for tokens with index >= self.vocab_size
            if len(curr_token_probs) > self.vocab_size:
                curr_token_probs = curr_token_probs[:self.vocab_size]
            sampled_token: int = torch.multinomial(curr_token_probs, 1).item()
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

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "group_size": self.group_size,
            "temperature": self.temp,
            "generation_type": self.generation_type,
            "top_p": self.top_p,
            "top_k": self.top_k,

        }
