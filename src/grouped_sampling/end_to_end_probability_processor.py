import torch

from src.grouped_sampling.probability_processor import ProbabilityProcessor
from src.grouped_sampling.top_k import TopKProbabilityProcessor
from src.grouped_sampling.top_p import TopPProbabilityProcessor


class EndToEndProbabilityProcessor(ProbabilityProcessor):
    def __init__(
        self,
        minimum_tokens_to_keep: int = 1,
        top_p: float = 1.0,
        top_k: int = 0,
    ):
        super(EndToEndProbabilityProcessor, self).__init__()
        self.processors = []
        if top_p != 1.0:
            self.processors.append(
                TopPProbabilityProcessor(minimum_tokens_to_keep, top_p)
            )
        if top_k != 0:
            self.processors.append(TopKProbabilityProcessor(top_k))

    def __call__(self, probs: torch.Tensor) -> torch.Tensor:
        self._validate_probs(probs)
        for processor in self.processors:
            probs = processor(probs)
        prob_sums = probs.sum(dim=-1)
        probs.div_(prob_sums.unsqueeze(-1))
        return probs
