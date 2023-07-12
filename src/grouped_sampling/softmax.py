from torch import Tensor, exp
from transformers import LogitsProcessor


class SoftmaxLogitNormalization(LogitsProcessor):
    """
    LogitsProcessor for Softmax normalization of logits.
    """

    def __init__(self, temperature: float):
        self.temperature = temperature

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        scores = exp(scores / self.temperature)
        return scores / scores.sum(-1, keepdim=True)