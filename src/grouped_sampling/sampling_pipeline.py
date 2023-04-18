from __future__ import annotations

from math import ceil
from random import seed
from typing import Any, Dict, Iterable, List, Optional, Generator

from torch import Tensor, manual_seed

from .base_pipeline import GroupedGenerationPipeLine
from .generation_type import GenerationType
from .sampling_stradegy import TopPSamplingStrategy, TopKSamplingStrategy, SamplingStrategy, GreedySamplingStrategy, \
    PureSamplingStrategy


class GroupedSamplingPipeLine(GroupedGenerationPipeLine):
    """
    A GroupedGenerationPipeLine that generates text
    using random sampling
    with top-k or top-p filtering.
    """

    default_seed: int = 0
    seed(default_seed)
    manual_seed(default_seed)
    unique_attrs = "top_k", "top_p"

    def __init__(
            self,
            *args,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            **kwargs,
    ):
        self.top_p: Optional[float] = top_p
        self.top_k: Optional[int] = top_k
        super().__init__(*args, **kwargs)

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if (key == "top_k" or key == "top_p") and hasattr(self, "wrapped_model"):
            self.wrapped_model.use_softmax = True
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
    def sampling_func(self) -> SamplingStrategy:
        if self.top_k is None and self.top_p is None:
            return PureSamplingStrategy()
        if self.top_k == 1 or self.top_p == 0.0:
            return GreedySamplingStrategy()
        if self.top_k is not None:
            if self.top_k > 1:
                return TopKSamplingStrategy(self.top_k)
            raise ValueError("top_k must be a positive integer")
        if self.top_p is not None:
            if self.top_p < 1.0:
                return TopPSamplingStrategy(self.top_p)
            return PureSamplingStrategy()
        raise RuntimeError("Uncovered case in generation_type property")

    def generate_group(self, prob_mat: Tensor) -> List[int]:
        """
        Gets a probability matrix of shape (group_size, vocab_size)
        Generates a group of tokens
        using the choice_function.
        Complexity: O(group_size)
        """
        prob_mat.cpu()
        # coping a tensor of size (group_size, vocab_size)
        # so the complexity is O(group_size)
        # (vocab_size is constant)
        new_group: List[int] = [
            self.sampling_func(prob_vec) for prob_vec in prob_mat
        ]
        # the complexity of the loop is O(group_size)
        # because self.sampling_func gets a tensor
        # of constant size (vocab_size,)
        # and therefore must be O(1) in complexity
        # and the loop has group_size iterations.
        for i, token_id in enumerate(new_group):
            if token_id == self.wrapped_model.end_of_sentence_id:
                return new_group[:i + 1]
                # return the group until the end of sentence token included
                # the complexity of this line is O(group_size)
                # because it is coping a list with maximum size of group_size
        return new_group

    def generate_group_batch(self, prob_tensor: Tensor) -> Generator[List[int], None, None]:
        """prob_tensor: tensor of shape (batch_size, group_size, vocab_size)"""
        for prob_mat in prob_tensor:
            yield self.generate_group(prob_mat)
            # the complexity of the loop is O(batch size * group_size)

    def _forward(
            self,
            tokenized_prompt: Tensor,
            num_new_tokens: int,
    ) -> List[int]:
        """
        Complexity:
            O(
                ((n ^ 3) / group_size) +
                ((n * l ^ 2) / group_size) +
                group_size +
                n
            )
        where l is the number of tokens in the prompt
        and n is the number of new tokens to generate
        """
        # let's define l = len(tokenized_prompt), n = num_new_tokens
        # coping a tensor of size lso O(l)
        curr_token_list: List[int] = tokenized_prompt.tolist()
        for _ in range(ceil(num_new_tokens / self.wrapped_model.group_size)):
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
            num_new_tokens: int,
    ) -> List[List[int]]:
        generation_start_indexes = [
            len(prompt) for prompt in tokenized_prompts
        ]
        curr_sequences: List[List[int]] = [
            tokenized_prompt.tolist() for tokenized_prompt in tokenized_prompts
        ]
        for _ in range(ceil(num_new_tokens / self.wrapped_model.group_size)):
            prob_tensor = self.wrapped_model.get_prob_mat_batch(
                tokens=curr_sequences,
                generation_start_indexes=generation_start_indexes,
            )  # tensor of shape (batch_size, group_size, vocab_size)
            new_sequences: Iterable[List[int]] = self.generate_group_batch(
                prob_tensor
            )
            for i, new_sequence in enumerate(new_sequences):
                new_sequence: List[int]
                if self.wrapped_model.end_of_sentence_id not in new_sequence:
                    if self.wrapped_model.end_of_sentence_id in new_sequence:
                        # new tokens id is the tokens before and including the end_of_sentence_id
                        end_of_sentence_index: int = new_sequence.index(self.wrapped_model.end_of_sentence_id)
                        new_sequence: List[int] = new_sequence[:end_of_sentence_index + 1]
                    curr_sequences[i].extend(new_sequence)
        return curr_sequences

    def __repr__(self):
        super_representation = super().__repr__()
        unique_representation = "/n".join(
            f"{unique_attr_name}={getattr(self, unique_attr_name)}"
            for unique_attr_name in self.unique_attrs)
        return super_representation + unique_representation

    def as_dict(self) -> Dict[str, Any]:
        super_dict = super(GroupedSamplingPipeLine, self).as_dict()
        super_dict.update({
            unique_attr: self.__getattribute__(unique_attr)
            for unique_attr in self.unique_attrs
        })
        return super_dict
