import heapq
from abc import ABC
from typing import List
from warnings import warn

from torch import Tensor, argmax, multinomial, zeros


class TokenProb:
    """
    Class for storing the probability of a token and the token itself.
    Used to store the probabilities of the next tokens in the sampling pipeline.
    Is useful because it supports the < and > operators, which are used in the
    heapq module
    The < and > are the opposite of each other because the heapq module is only supporting minimum heaps
    and I need a maximum heap
    """

    __slots__ = ["token_id", "prob"]

    def __init__(self, token_id: int, prob: Tensor):
        self.token_id: int = token_id
        self.prob: Tensor = prob

    def __lt__(self, other: "TokenProb"):
        """
        Overrides the < operator
        Comparison is done by the probability
        """
        return self.prob > other.prob  # pragma: no cover

    def __gt__(self, other: "TokenProb"):
        """
        Overrides the > operator
        Comparison is done by the probability
        """
        return self.prob < other.prob  # pragma: no cover


class SamplingStrategy(ABC):
    """Defines the function that gets a probability vector and returns a token id"""

    def __call__(self, prob_vec: Tensor) -> int:
        raise NotImplementedError("SamplingStrategy is an abstract class")


class GreedySamplingStrategy(SamplingStrategy):
    """A SamplingStrategy that samples the token with the highest probability"""

    def __call__(self, prob_vec: Tensor) -> int:
        return argmax(prob_vec).item()


class PureSamplingStrategy(SamplingStrategy):
    """
    A SamplingStrategy that samples from the probability vector
    without any filtering
    """

    def __call__(self, prob_vec: Tensor) -> int:
        return multinomial(prob_vec, 1).item()


class TopPSamplingStrategy(SamplingStrategy):
    """
    A SamplingStrategy that samples from the top p tokens
    such that their sum in the original vector is <= self.top_p.
    If token with the highest probability
    have a probability higher than top_p, it will be sampled
    """

    pure_sampling_function: SamplingStrategy = PureSamplingStrategy()

    def __init__(self, top_p: float):
        if not 0 <= top_p <= 1:
            raise ValueError(f"top_p must be between 0 and 1, got {top_p}")
        if top_p == 0:
            warn(
                "top_p is 0, use the more efficient GreedySamplingStrategy instead"
            )
        if top_p == 1:
            warn(
                "top_p is 1, use the more efficient PureSamplingStrategy instead"
            )
        self.top_p = top_p

    def __call__(self, prob_vec: Tensor) -> int:
        prob_sum: float = 0.0
        converted_probs = [
            TokenProb(i, prob) for i, prob in enumerate(prob_vec)
        ]
        heapq.heapify(converted_probs)
        new_probs: Tensor = zeros(prob_vec.shape, dtype=float)
        while prob_sum < self.top_p and len(converted_probs) > 0:
            curr_token_prob: TokenProb = heapq.heappop(converted_probs)
            token_id = curr_token_prob.token_id
            if curr_token_prob.prob <= 0.0:
                break
            if curr_token_prob.prob > 1:
                raise ValueError(f"Probability of token {token_id} "
                                 f"in the vector {prob_vec} is higher than 1")
            new_probs[token_id] = curr_token_prob.prob
            prob_sum += curr_token_prob.prob
        return self.pure_sampling_function(new_probs)


class TopKSamplingStrategy(SamplingStrategy):
    """Defines a SamplingStrategy that samples from the top k tokens"""

    pure_sampling_function: SamplingStrategy = PureSamplingStrategy()

    def __init__(self, top_k: int):
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if top_k == 1:
            warn(
                "top_k is 1, use the more efficient GreedySamplingStrategy instead"
            )
        self.top_k = top_k

    def __call__(self, prob_vec: Tensor) -> int:
        top_k_keys: List[int] = heapq.nlargest(self.top_k,
                                               range(prob_vec.shape[0]),
                                               key=lambda x: prob_vec[x])
        prob_sum = sum(prob_vec[token_id] for token_id in top_k_keys)
        new_probs = zeros(prob_vec.shape, dtype=float)
        for token_id in top_k_keys:
            new_probs[token_id] = prob_vec[token_id] / prob_sum
        return self.pure_sampling_function(new_probs)
