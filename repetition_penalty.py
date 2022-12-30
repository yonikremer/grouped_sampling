from abc import ABC, abstractmethod
from typing import Set, Optional

from torch import Tensor

from globals import TokenIDS


class RepetitionPenaltyStrategy(ABC):
    theta: float

    @staticmethod
    def get_already_generated_tokens(tokens: TokenIDS, generation_start: int) -> Set[int]:
        return set(tokens[generation_start:])

    @abstractmethod
    def __call__(self, logits: Tensor, tokens: TokenIDS, generation_start: int) -> Tensor:
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(theta={self.theta})"

    def __repr__(self) -> str:
        return str(self)


class LogitScalingRepetitionPenalty(RepetitionPenaltyStrategy):
    def __init__(self, theta: float):
        if theta <= 1:
            raise ValueError("Theta must be > 1")
        self.theta = theta

    def __call__(self, logits: Tensor, tokens: TokenIDS, generation_start: int) -> Tensor:
        """applies repetition penalty,
        using the repetition_penalty_theta parameter defined in the class
        the formula from the paper is:
        softmax(original_logits, T) = exp(original_logits_i/(T · h(i)) / sum(exp(original_logits_i/(T · h(i)))
        which is equivalent to:
        pi = exp((xi/h(i)) / T / sum(exp(xj/(T · h(j)))
        and to:
        penalized_logits = original_logits / h(i)
        pi = softmax(original_logits / h(i), T)
        where:
            h(i) = θ if i in generated_tokens else 1
            T is the temperature parameter of softmax
        Args:
            logits: the logits matrix of shape (group_size, vocab_size)
            tokens: the sequence of tokes from the prompt and the generated
            generation_start: The index of the first
        Returns:
            original_logits / h(i)
        complexity: O(group_size * n) where n is the number of tokens generated by the algorithm
        """
        generated_tokens = self.get_already_generated_tokens(tokens, generation_start)
        for token_id in generated_tokens:  # len(generated_tokens) < max(n, vocab_size)
            logits[:, token_id] /= self.theta
            # O(group_size) because we are taking a slice of size group_size
        return logits


class NoRepetitionPenalty(RepetitionPenaltyStrategy):
    theta = 1.0

    def __call__(self, logits: Tensor, tokens: TokenIDS, generation_start: int) -> Tensor:
        return logits


DEFAULT_REPETITION_PENALTY = LogitScalingRepetitionPenalty(1.2)


def repetition_penalty_factory(theta: Optional[float]) -> RepetitionPenaltyStrategy:
    if theta is None:
        return DEFAULT_REPETITION_PENALTY
    if theta == 1:
        return NoRepetitionPenalty()
    elif theta > 1:
        return LogitScalingRepetitionPenalty(theta)
    else:
        raise ValueError("theta must be greater than 1")
