import torch
from torch import Tensor

from src.grouped_sampling.probability_processor import ProbabilityProcessor


class TopPProbabilityProcessor(ProbabilityProcessor):
    """
    A class for proccessing probabilities by keeping only the top p values.
    And setting the rest to 0.
    The class keeps the top values for each vector in the last dimension such that their sum is top p or more.
    """

    def __init__(self, minimum_tokens_to_keep: int, top_p: float):
        super(TopPProbabilityProcessor, self).__init__()
        try:
            top_p = float(top_p)
        except ValueError:
            raise TypeError("top_p must be a float")
        if top_p <= 0 or top_p >= 1:
            raise ValueError("top_p must be in (0, 1)")
        try:
            minimum_tokens_to_keep = int(minimum_tokens_to_keep)
        except ValueError:
            raise TypeError("minimum_tokens_to_keep must be an int")
        if minimum_tokens_to_keep <= 0:
            raise ValueError("minimum_tokens_to_keep must be positive")
        self.top_p = top_p
        self.minimum_tokens_to_keep = minimum_tokens_to_keep

    def __call__(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            probs: Tensor of shape (batch_size, output_length, vocab_size).
                containing the probabilities of each token in the vocabulary.
        """
        self._validate_probs(probs)
        sorted_probs, sorted_indices = probs.sort(dim=-1, descending=False)
        cumulative_probs = sorted_probs.cumsum(dim=-1)
        # Remove tokens with cumulative top_p above the threshold
        sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., -self.minimum_tokens_to_keep :] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        probs.masked_fill_(indices_to_remove, 0)
        return probs

    def _validate_probs(self, probs: Tensor):
        super()._validate_probs(probs)
        # Chack that there is at least self.minimum_tokens_to_keep tokens with probability > 0
        if (probs > 0).sum(dim=-1).min() < self.minimum_tokens_to_keep:
            raise ValueError(
                f"probs must have at least {self.minimum_tokens_to_keep} tokens with probability > 0."
            )
