from abc import ABC

import torch
from torch import Tensor


class ProbabilityProcessor(ABC):

    def __call__(self, probs: Tensor) -> Tensor:
        """
        Args:
            probs: Tensor of shape (batch_size, output_length, vocab_size).
        Returns:
            A Tensor of shape (batch_size, output_length, vocab_size) with the proccessed probs.
        """
        raise NotImplementedError

    @staticmethod
    def _validate_probs(probs):
        if not torch.is_tensor(probs):
            raise TypeError("probs must be a tensor")
        if probs.dim() != 3:
            raise ValueError(
                f"probs must be a 3 dimensional tensor. Got {probs.dim()} dimensions."
            )
        if min(probs.shape) == 0:
            raise ValueError(
                f"probs must be a non empty tensor. Got {probs.shape} shape.")
        if (torch.isnan(probs)).any():
            raise ValueError("probs must not contain NaN values.")
        if (torch.isinf(probs)).any():
            raise ValueError("probs must not contain Inf values.")
        if (probs < 0).any():
            raise ValueError("probs must not contain negative values.")
        if (probs > 1).any():
            raise ValueError("probs must not contain values greater than 1.")
        probs_sum: Tensor = probs.sum(dim=-1)
        if (probs_sum - 1 > 1e-4).any():
            raise ValueError(
                "probs must sum to 1 or less in the last dimension."
                f"\nGot {probs_sum}")
