import torch

from src.grouped_sampling.probability_processor import ProbabilityProcessor


class TopKProbabilityProcessor(ProbabilityProcessor):
    """
    A class for proccessing probabilities by keeping only the top k values.
    And setting the rest to 0.
    The class keeps the top k values for each row in the last dimension.
    And sets the rest to 0.
    """

    def __init__(self, top_k: int):
        super(TopKProbabilityProcessor, self).__init__()
        try:
            top_k = int(top_k)
        except ValueError:
            raise TypeError("top_k must be an int")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.top_k = top_k

    def __call__(self, probs: torch.Tensor) -> torch.Tensor:
        self._validate_probs(probs)
        if self.top_k == probs.shape[-1]:
            return probs
        self._validate_probs(probs)
        topk_vals, topk_inds = probs.topk(self.top_k, dim=-1)
        zeros_tensor = torch.zeros_like(probs, device=probs.device)
        zeros_tensor.scatter_(-1, topk_inds, topk_vals)
        return zeros_tensor

    def _validate_probs(self, probs):
        super()._validate_probs(probs)
        if probs.shape[-1] < self.top_k:
            raise ValueError(
                f"probs must have at least {self.top_k} tokens with probability > 0."
            )
