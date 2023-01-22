from __future__ import annotations

import heapq
from collections.abc import Iterator
from multiprocessing import Pool
from random import seed
from typing import Callable, List, Dict, Optional, Any, Iterable

from torch import Tensor, zeros, argmax, multinomial, manual_seed

from .generation_type import GenerationType
from .base_pipeline import GroupedGenerationPipeLine


class TokenProb:
    """Class for storing the probability of a token and the token itself.
    Used to store the probabilities of the next tokens in the sampling pipeline.
    Is useful because it supports the < and > operators, which are used in the
    heapq module
    The < and > are the opposite of each other because the heapq module is only supporting minimum heaps
    and I need a maximum heap"""
    __slots__ = ['token_id', 'prob']

    def __init__(self, token_id: int, prob: Tensor):
        self.token_id: int = token_id
        self.prob: Tensor = prob

    def __lt__(self, other: "TokenProb"):
        """Overrides the < operator
        Comparison is done by the probability"""
        return self.prob > other.prob  # pragma: no cover

    def __gt__(self, other: "TokenProb"):
        """Overrides the > operator
        Comparison is done by the probability"""
        return self.prob < other.prob  # pragma: no cover


class GroupedSamplingPipeLine(GroupedGenerationPipeLine):
    """A GroupedGenerationPipeLine that generates text
    using random sampling
    with top-k or top-p filtering."""
    default_seed: int = 0
    seed(default_seed)
    manual_seed(default_seed)
    unique_attrs = "top_k", "top_p"

    def __init__(self, top_k: Optional[int] = None,
                 top_p: Optional[float] = None, *args, **kwargs):
        self.top_p: Optional[float] = top_p
        self.top_k: Optional[int] = top_k
        super().__init__(*args, **kwargs)

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if key == "default_seed":
            seed(value)  # pragma: no cover
            manual_seed(value)  # pragma: no cover

    @property
    def generation_type(self) -> GenerationType:
        if self.top_k is None and self.top_p is None:
            return GenerationType.RANDOM
        if self.top_k == 1 or self.top_p == 0.0:
            return GenerationType.GREEDY
        if self.top_k is not None:
            return GenerationType.TOP_K
        if self.top_p is not None:
            if self.top_p < 1.0:
                return GenerationType.TOP_P
            return GenerationType.RANDOM
        raise RuntimeError("Uncovered case in generation_type property")

    @property
    def sampling_func(self) -> Callable[[Tensor], int]:
        gen_type_to_filter_method: Dict[GenerationType, Callable[[Tensor, ], int]] = {
            GenerationType.TOP_K: self.top_k_sampling,
            GenerationType.TOP_P: self.top_p_sampling,
            GenerationType.GREEDY: GroupedSamplingPipeLine.highest_prob_token,
            GenerationType.RANDOM: GroupedSamplingPipeLine.unfiltered_sampling,
        }
        return gen_type_to_filter_method[self.generation_type]

    @staticmethod
    def unfiltered_sampling(prob_vec: Tensor) -> int:
        """A sampling function that doesn't filter any tokens.
        returns a random token id sampled from the probability vector"""
        return multinomial(prob_vec, 1).item()

    @staticmethod
    def highest_prob_token(prob_vec: Tensor) -> int:
        """Gets a probability vector of shape (vocab_size,)
        returns the token id with the highest probability"""
        return argmax(prob_vec).item()

    def top_p_sampling(self, prob_vec: Tensor) -> int:
        """Gets a probability vector of shape (vocab_size,)
        computes a probability vector with the top p tokens
        such that their sum in the original vector is <= self.top_p.
        and samples from that vector.
        If token with the highest probability
        have a probability higher than top_p, it will be sampled"""
        prob_sum: float = 0.0
        converted_probs = [
            TokenProb(i, prob) for i, prob in enumerate(prob_vec)
        ]
        heapq.heapify(converted_probs)
        new_probs = zeros(prob_vec.shape, dtype=float)
        while prob_sum < self.top_p and len(converted_probs) > 0:
            curr_token_prob: TokenProb = heapq.heappop(converted_probs)
            token_id = curr_token_prob.token_id
            if curr_token_prob.prob <= 0.0:
                break
            if curr_token_prob.prob > 1:
                raise ValueError(
                    f"Probability of token {token_id} "
                    f"in the vector {prob_vec} "
                    f"is {curr_token_prob.prob}"
                    f" which is higher than 1")
            prob_sum += curr_token_prob.prob
            new_probs[token_id] = curr_token_prob.prob
        if prob_sum == 0.0:
            return converted_probs[0].token_id
        return GroupedSamplingPipeLine.unfiltered_sampling(new_probs)

    def top_k_sampling(self, prob_vec: Tensor) -> int:
        """Gets a token id: probability mapping
        returns the TOP_K tokens
        with the highest probability.
        this is the bottleneck of the sampling pipeline."""
        top_k_keys: List[int] = heapq.nlargest(
            self.top_k,
            range(prob_vec.shape[0]),
            key=lambda x: prob_vec[x]
        )
        prob_sum = sum(prob_vec[token_id] for token_id in top_k_keys)
        new_probs = zeros(prob_vec.shape, dtype=float)
        for token_id in top_k_keys:
            new_probs[token_id] = prob_vec[token_id] / prob_sum
        return GroupedSamplingPipeLine.unfiltered_sampling(new_probs)

    def generate_group(self, prob_mat: Tensor) -> List[int]:
        """
        Gets a probability matrix of shape (group_size, vocab_size)
        Generates a group of tokens
         using the choice_function.
         Complexity: O(group_size)"""
        prob_mat.cpu()
        # coping a tensor of size (group_size, vocab_size)
        # so the complexity is O(group_size)
        # (vocab_size is constant)
        new_group: List[int] = [
            self.sampling_func(prob_vec)
            for prob_vec in prob_mat
        ]
        # the complexity of the loop is O(group_size)
        # because self.sampling_func gets a tensor
        # of constant size (vocab_size,)
        # and therefore must be O(1) in complexity
        # and the loop has group_size iterations.
        del prob_mat
        for i, token_id in enumerate(new_group):
            if token_id == self.wrapped_model.end_of_sentence_id:
                return new_group[:i + 1]
                # return the group until the end of sentence token included
                # the complexity of this line is O(group_size)
                # because it is coping a list with maximum size of group_size
        return new_group

    def generate_group_batch(self, prob_tensor: Tensor) -> Iterable[List[int]]:
        """prob_tensor: tensor of shape (batch_size, group_size, vocab_size)"""
        pool = Pool()
        # create a list of arguments for each process
        args = [prob_mat for prob_mat in prob_tensor]
        new_tokens = pool.map(self.generate_group, args)
        # close the pool
        pool.close()
        # wait for the processes to finish
        pool.join()
        return new_tokens

    def _forward(
            self,
            tokenized_prompt: Tensor,
            num_new_tokens: int,
    ) -> List[List[int]]:
        """Complexity:
            O(
                ((n ^ 3) / group_size) +
                ((n * l ^ 2) / group_size) +
                group_size +
                n
            )
        where l is the number of tokens in the prompt
        and n is the number of new tokens to generate"""
        # let's define l = len(tokenized_prompt), n = num_new_tokens
        # coping a tensor of size lso O(l)
        curr_token_list: List[int] = tokenized_prompt.tolist()
        for _ in range(num_new_tokens // self.wrapped_model.group_size):
            # and each iteration is
            # O(n ^ 2 + l ^ 2 + group_size ^ 2 + group_size)
            # so the complexity of the loop is
            # O((n ^ 3) / group_size + (n * l ^ 2) / group_size + group_size + n)
            prob_mat: Tensor = self.wrapped_model.get_prob_mat(
                curr_token_list, len(tokenized_prompt)
            )
            # complexity: O(group_size ^ 2 + len(curr_token_list) ^ 2)
            # len(curr_token_list) <= n + l
            # so the complexity is
            # O(group_size ^ 2 + (n + l) ^ 2) is equals to
            # O(n ^ 2 + nl + l ^ 2 + group_size ^ 2)
            # but nl <= max(n^2, l^2) so the complexity
            # is O(n ^ 2 + l ^ 2 + group_size ^ 2)
            new_tokens = self.generate_group(prob_mat)
            # complexity: O(group_size)
            # len(curr_token_list) <= n + l
            # so the complexity is O(group_size * (n + l + group_size))
            # len(new_tokens) = group_size
            if self.wrapped_model.end_of_sentence_id in new_tokens:
                # the check is O(group_size)
                end_of_sentence_index = new_tokens.index(
                    self.wrapped_model.end_of_sentence_id)
                # O(group_size) because len(new_tokens) <= group_size
                new_tokens = new_tokens[:end_of_sentence_index]
                # O(group_size) because end_of_sentence_index < group_size
            curr_token_list.extend(new_tokens)
            # O(group_size) because len(new_tokens) <= group_size
        return curr_token_list

    def forward_batch(
            self,
            tokenized_prompts: List[Tensor],
            num_new_tokens: List[int],
    ) -> List[List[int]]:
        generation_start_indexes = [len(prompt) for prompt in tokenized_prompts]
        curr_sequences: List[List[int]] = [tokenized_prompt.tolist() for tokenized_prompt in tokenized_prompts]
        for _ in range(num_new_tokens // self.wrapped_model.group_size):
            prob_tensor = self.wrapped_model.get_prob_mat_batch(
                tokens=tokenized_prompts,
                generation_start_indexes=generation_start_indexes,
            )  # tensor of shape (batch_size, group_size, vocab_size)
            new_tokens: List[List[int]] = self.generate_group_batch(prob_tensor)
            for i, new_token in enumerate(new_tokens):
                curr_sequences[i].extend(new_token)
        return curr_sequences

    def __repr__(self):
        super_representation = super().__repr__()
        unique_representation = '/n'.join(
            f"{unique_attr_name}={getattr(self, unique_attr_name)}"
            for unique_attr_name in self.unique_attrs)
        return super_representation + unique_representation

    def as_dict(self) -> Dict[str, Any]:
        super_dict = super(GroupedSamplingPipeLine, self).as_dict()
        super_dict.update(
            {unique_attr: self.__getattribute__(unique_attr)
             for unique_attr in self.unique_attrs}
        )
        return super_dict
