from __future__ import annotations

import sys
from math import ceil
from typing import List, Dict, Tuple, Sequence, Any, Optional

from torch import Tensor, LongTensor

from .generation_type import GenerationType
from .base_pipeline import GroupedGenerationPipeLine
from .token_ids import TokenIDS


class NoCompletionsFound(Exception):
    def __init__(
            self,
            curr_text_generator: GroupedGenerationPipeLine,
            additional_info: str = ""):
        super(NoCompletionsFound, self).__init__(
            f"text generator: {curr_text_generator} \n"
            f"additional info: {additional_info}")


class GroupedTreePipeLine(GroupedGenerationPipeLine):
    """A GroupedGenerationPipeLine that generates text
     in a TREE like fashion,
     without random sampling."""
    unique_attrs = "top_k", "top_p"

    def __init__(self, top_k: Optional[int], top_p: Optional[float], *args, **kwargs):
        self.top_p = top_p
        self.top_k = top_k
        super().__init__(
            *args, **kwargs
        )

    @property
    def generation_type(self) -> GenerationType:
        if self.top_k == 1 or self.top_p == 0.0:
            return GenerationType.GREEDY
        return GenerationType.TREE

    @property
    def actual_top_k(self) -> int:
        """The maximum number of tokens to consider for each position
        It is always smaller than or equal to the vocab size"""
        return min(self.top_k, int(self.tokenizer.vocab_size * self.top_p))

    @staticmethod
    def no_duplicates(
            my_sequence: List[Any],
            prompt_length: int) -> bool:
        """Return if there isn't a repetition in the sequence
        complexity: O(n) where n is the length of the sequence"""
        generated_tokens = my_sequence[prompt_length:]
        return len(generated_tokens) == len(set(generated_tokens))

    @staticmethod
    def combinations(
            mat: Sequence[Sequence[Any]],
            prompt_length: int) -> List[List[Any]]:
        """Returns all the lists such that list[j] is in mat[j]
        complexity: O(prod(len(vec) for vec in mat))
        the maximal length of the returned list is the product of the lengths of the vectors in mat"""
        mat_at_zero = mat[0]
        if len(mat) == 1:
            return [[item] for item in mat_at_zero]  # complexity: O(len(mat_at_zero))
        res: List[List[Any]] = []
        for item in mat_at_zero:  # complexity: O(len(mat_at_zero))
            for j in GroupedTreePipeLine.combinations(mat[1:], prompt_length):
                res.append([item] + j)
        filtered_res: List[List[Any]]
        filtered_res = list(filter(lambda x: GroupedTreePipeLine.no_duplicates(x, prompt_length), res))
        return filtered_res

    @staticmethod
    def seq_prob(tokens, prob_mat,
                 org_prompt_prob) -> float:
        """Given the probability matrix
         and a list of tokens,
        returns the probability
        of the sequence.
        prob_mat[a][b] is the probability of
        the token with id b the a-th
        token in the sequence
        complexity: O(n) where n is the length of the sequence"""
        probability = org_prompt_prob
        for i, curr_token in enumerate(tokens):
            probability *= prob_mat[i][curr_token]  # complexity: O(1)
        return probability

    def remove_duplicates(
            self,
            completions: List[List[int]],
            probs: List[float],
    ) -> Dict[Tuple[int], float]:
        """Given a list of tokenized answers
         and the probability of each completion,
        removes every repeated completion
         and every completion that have repeated tokens
         complexity: sum([len(completion) for completion in completions])"""
        filtered_completions: Dict[Tuple[int], float]
        filtered_completions = dict()
        for curr_comp, curr_prob in zip(completions, probs):
            if self.wrapped_model.end_of_sentence_id in curr_comp:
                # complexity: O(m) where m is the length of the sequence
                end_of_sentence_index = curr_comp.index(self.wrapped_model.end_of_sentence_id)
                # complexity: O(m)
                curr_comp = curr_comp[:end_of_sentence_index + 1]
                # complexity: O(m)
            curr_comp_tuple = tuple(curr_comp)  # complexity: O(m)
            filtered_completions[curr_comp_tuple] = curr_prob  # complexity: O(1)
        if len(filtered_completions) == 0:  # complexity: O(1)
            raise NoCompletionsFound(self, f"all completions contained duplicates, "
                                           f"completions: {completions}")
        return filtered_completions

    def generate_group(
            self, prob_mat: Tensor,
            org_prompt: TokenIDS) -> List[List[int]]:
        """given a matrix of probabilities,
        returns a list of lists of tokens.
        samples the tokens such that
        for each place in the group,
        at most top_k tokens are sampled
        and at least one token is sampled
        and the added probability of all the tokens is
        less than or equal top_p
        returns a list of where every item is
         a tuple of a sequence and probability
        over all complexity of the function is
        O(group_size)
        the maximum length of the returned list is
        actual_top_k ^ group_size"""
        # let's call len(org_prompt) = n
        # prob_tensor.shape is (group_size, vocab_size)
        possible_tokens: List[List[int]] = []  # complexity: O(1)
        for token_prob in prob_mat:  # group_size times
            token_prob: Tensor  # token_prob.shape is (vocab_size,)
            indexed_prob = enumerate(token_prob)
            # creating an enumerator so the complexity is O(1)
            sorted_indexed_prob: List[Tuple[int, Tensor]] = list(sorted(indexed_prob, key=lambda x: x[0], reverse=True))
            # O(vocab_size*log(vocab_size)) so O(1)
            curr_k = 0
            total_prob = 0
            curr_indices: List[int] = []
            token: LongTensor
            for prob, token in sorted_indexed_prob:  # constant (vocab_size) iterations
                # O(TOP_K)
                token_id: int = int(token.item())
                top_p_break: bool = total_prob + prob > self.top_p
                top_k_break: bool = curr_k == self.top_k
                if top_p_break or top_k_break:
                    break
                curr_k += 1
                total_prob += prob
                curr_indices.append(token_id)  # O(1)
            # the complexity of this loop is O(1)
            # curr_indices maximum length is max(top_k, vocab_size * top_p) and I will call it actual_top_k
            if len(curr_indices) == 0:
                # If we didn't find any tokens to sample from,
                # we sample the most probable token
                highest_prob_token_id = int(sorted_indexed_prob[0][1].item())  # O(1)
                curr_indices.append(highest_prob_token_id)  # O(1)
            possible_tokens.append(curr_indices)  # O(1)
        # possible_tokens maximum size is group_size * k
        new_sequences: List[List[int]]
        new_sequences = GroupedTreePipeLine.combinations(possible_tokens, len(org_prompt))
        # complexity: O(prod([len(mat[i]) for i in range(len(mat))]))
        # so it's O(prod(k for i in range(group_size))) which is O(k^group_size)
        # The maximum length for new_sequences is actual_top_k ** group_size
        if len(new_sequences) == 0:
            raise NoCompletionsFound(self)
        # O(sum(len(indices[i]))
        # for i in range(group_size)))
        # len(indices[i]) < actual_top_k
        # therefore the complexity is
        # O(actual_top_k * group_size) = O(vocab_size * group_size) = O(group_size)

        return new_sequences

    def rec_gen(self, org_prompt: TokenIDS,
                num_tokens: Optional[int],
                org_prompt_prob: float,
                prompt_length: int) \
            -> Dict[Tuple[int], float]:
        """Recursively generates the next group of tokens
         in a TREE like behavior"""
        # n is len(org_prompt)
        # m is num_tokens
        if self.wrapped_model.end_of_sentence_id in org_prompt:  # O(n)
            end_of_sentence_index = org_prompt.index(self.wrapped_model.end_of_sentence_id)  # O(n)
            return {tuple(org_prompt[:end_of_sentence_index]): org_prompt_prob}  # O(n)
        prob_mat: Tensor = self.wrapped_model.get_prob_mat(org_prompt, prompt_length)  # O(n^2 + group_size^2)
        tokenized_ans_list: List[List[int]] = self.generate_group(prob_mat, org_prompt)  # O(group_size)
        # tokenized_ans_list maximum length is actual_top_k ** group_size
        prob_list: List[float]
        prob_list = [GroupedTreePipeLine.seq_prob(
            seq,
            prob_mat,
            org_prompt_prob
        ) for seq in tokenized_ans_list]  # O(sum(len(seq) for seq in tokenized_ans_list))
        # so O(group_size * len(tokenized_ans_list)) so O(group_size * (actual_top_k ** group_size))
        new_prompts: List[List[int]]
        ans: List[int]
        new_prompts = [list(org_prompt) + list(ans) for ans in tokenized_ans_list]
        # O(sum(len(seq) for seq in tokenized_ans_list) + n * len(tokenized_ans_list))
        # so O(group_size * (actual_top_k ** group_size + n))
        # new_prompts shape is [(actual_top_k ** group_size + n), (group_size + n)]
        completion_probs: Dict[Tuple[int], float]
        completion_probs = self.remove_duplicates(new_prompts, prob_list)
        # O(group_size * (actual_top_k ** group_size + n))
        # the maximum length of new_prompts is (actual_top_k ** group_size + n)
        all_completions_ended: bool = all(self.wrapped_model.end_of_sentence_id in tokenized_ans
                                          for tokenized_ans in completion_probs.keys())
        # O(group_size * len(completion_probs))
        num_groups: Optional[int] = None
        if num_tokens is not None:
            num_groups = ceil(num_tokens / self.wrapped_model.group_size)
            #  num_groups is the maximum recursion depth of this function
        if num_groups == 1 or all_completions_ended:
            return completion_probs

        new_completions: Dict[Tuple[int], float] = dict()
        if num_tokens is None:
            new_number_tokens = None
        else:
            new_number_tokens = num_tokens - self.wrapped_model.group_size
        for curr_new_prompt, curr_new_prompt_prob in completion_probs.items():
            # at maximum there are group_size * (actual_top_k ** group_size) iterations
            curr_new_prompt: List[int]
            curr_completions: Dict[Tuple[int], float]
            curr_completions = self.rec_gen(
                org_prompt=curr_new_prompt,
                num_tokens=new_number_tokens,
                org_prompt_prob=curr_new_prompt_prob,
                prompt_length=prompt_length
            )
            tokens: Tuple[int]
            prob: float
            for tokens, prob in curr_completions.items():  # O(len(curr_completions))
                # so O(number of generated tokens)
                new_completions[tokens] = prob  # O(1)
        return new_completions

    def _forward(
            self,
            tokenized_prompt: Tensor,
            num_new_tokens: Optional[int] = None,
            num_return_sequences: int = 1,
    ) -> List[List[int]]:
        if num_return_sequences > 1 and self.generation_type == GenerationType.GREEDY:
            raise ValueError("greedy generation with tree can't return more than one sequence")

        if num_new_tokens is not None:
            num_groups = ceil(num_new_tokens / self.wrapped_model.group_size)
            if num_groups >= sys.getrecursionlimit():
                sys.setrecursionlimit(num_groups + 100)
        seq_prob_dict: Dict[Tuple[int], float]
        seq_prob_dict = self.rec_gen(
            org_prompt=tokenized_prompt.tolist(),
            num_tokens=num_new_tokens,
            org_prompt_prob=1.0,
            prompt_length=len(tokenized_prompt)
        )
        if len(seq_prob_dict) < num_return_sequences:
            raise NoCompletionsFound(
                self,
                f"Not enough completions found,"
                f" {len(seq_prob_dict)} found"
                f" but {num_return_sequences} requested"
            )
        sorted_seq_prob_dict = sorted(
            seq_prob_dict.items(), key=lambda x: x[1], reverse=True
        )  # O(n log n) where n is len(seq_prob_dict)
        highest_prob_answers = sorted_seq_prob_dict[:num_return_sequences]  # O(num_return_sequences)
        return [tokens for tokens, prob
                in highest_prob_answers]

    def __repr__(self):
        super_representation = super().__repr__()
        unique_representation = '/n'.join(f"{unique_attr_name}={getattr(self, unique_attr_name)}"
                                          for unique_attr_name in self.unique_attrs)
        return super_representation + unique_representation

    def as_dict(self) -> Dict[str, Any]:
        super_dict = super(GroupedTreePipeLine, self).as_dict()
        super_dict.update({unique_attr: self.__getattribute__(unique_attr) for unique_attr in self.unique_attrs})
        return super_dict
