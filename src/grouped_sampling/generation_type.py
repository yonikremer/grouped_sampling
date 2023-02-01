from enum import Enum


class GenerationType(Enum):
    """The type of generation to use"""

    GREEDY = "greedy"
    TOP_K = "top_k"
    TOP_P = "top_p"
    TREE = "tree"
    RANDOM = "random"

    def requires_softmax(self) -> bool:
        """Whether the generation type requires a softmax"""
        return self in (self.TOP_K, self.TOP_P, self.TREE, self.RANDOM)
