from __future__ import annotations

import sys
from math import ceil
from typing import List, Dict, Tuple, Sequence, Any, Optional

from torch import Tensor

from text_generator import TextGenerator, NoCompletionsFound, GenerationType, TokenIDS


class TreeGenerator(TextGenerator):
    """A TextGenerator that generates text
     in a TREE like fashion,
     without random sampling."""
    top_p: float
    top_k: int

    def __init__(self, model_name: str, group_size: int,
                 top_k: Optional[int], top_p: Optional[float],
                 temp: float = 1.0,
                 end_of_sentence_stop: bool = False,
                 answer_length_multiplier: int = 16):
        self.top_p, self.top_k = top_p, top_k
        super().__init__(
            model_name=model_name,
            group_size=group_size,
            temp=temp,
            end_of_sentence_stop=end_of_sentence_stop,
            answer_length_multiplier=answer_length_multiplier
        )

    @property
    def generation_type(self) -> GenerationType:
        top_p, top_k = self.top_p, self.top_k
        if top_k is None and top_p is None:
            self.top_k = self.vocab_size
            self.top_p = 1.0
            return GenerationType.TREE
        if top_k == 1 or top_p == 0.0:
            return GenerationType.GREEDY
        if top_k is not None:
            if top_k < self.vocab_size:
                return GenerationType.TOP_K
            return GenerationType.TREE
        if top_p is not None:
            if top_p < 1.0:
                return GenerationType.TOP_P
            return GenerationType.TREE
        raise RuntimeError("Uncovered case in generation_type property")

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
        runtime function:
        prod([len(mat[i]) for i in range(len(mat))])"""
        if len(mat) == 1:
            return [[mat[0][i]]
                    for i in range(len(mat[0]))]
        res: List[List[Any]] = []
        for i in mat[0]:
            for j in TreeGenerator.combinations(
                    mat[1:],
                    prompt_length
            ):
                res.append([i] + j)
        filtered_res: List[List[Any]]
        filtered_res = list(filter(
            lambda x: TreeGenerator.no_duplicates(x, prompt_length),
            res))
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

    @staticmethod
    def flatten(ids: TokenIDS) -> List:
        """Gets a list where some elements might be lists
        and adds every item
        in the inner list to the outer list.
        example:
        [1, [2, 3], 4, [[5]]] -> [1, 2, 3, 4, 5]
        Complexity:
        n - len(the flatten list)
        m - the number of different lists
        O(m + n)"""
        new_list = []
        for item in ids:
            if isinstance(item, list) or isinstance(item, tuple):
                new_list.extend(
                    TreeGenerator.flatten(item))
            else:
                new_list.append(item)
        return new_list

    def remove_duplicates(
            self,
            completions: List[List[int]],
            probs: List[float],
            prompt_length: int
    ) -> Dict[Tuple[int], float]:
        """Given a list of tokenized answers
         and the probability of each completion,
        removes every repeated completion
         and every completion that have repeated tokens
         complexity: prod([len(completion) for completion in completions])"""
        filtered_completions: Dict[Tuple[int], float]
        filtered_completions = dict()
        for curr_comp, curr_prob in zip(completions, probs):
            if self.end_of_sentence_id in curr_comp:  # complexity: O(m) where m is the length of the sequence
                end_of_sentence_index = curr_comp.index(self.end_of_sentence_id)
                # complexity: O(m)
                completion_body = curr_comp[prompt_length:end_of_sentence_index]
                # complexity: O(m)
            else:
                completion_body = curr_comp[prompt_length:]
                # complexity: O(prompt_length)
            if len(completion_body) == len(set(completion_body)):  # complexity: O(m)
                curr_comp_tuple = tuple(curr_comp)  # complexity: O(m)
                filtered_completions[curr_comp_tuple] = curr_prob  # complexity: O(1)
            else:
                print("This method is useful")
        if len(filtered_completions) == 0:  # complexity: O(1)
            raise NoCompletionsFound(self, f"all completions contained duplicates, "
                                           f"completions: {completions}")
        return filtered_completions

    def generate_group(
            self, prob_mat: Tensor,
            org_prompt: TokenIDS) -> List[List[int]]:
        """given a matrix of probabilities,
        returns a list of lists of tokens.
        the matrix is of size group_size x vocab_size
        where matrix[i, j] is
        the probability of token j the i-th token in the group
        samples the tokens such that
        for each place in the group,
        at most top_k tokens are sampled
        and at least one token is sampled
        and the added probability of all the tokens is
        less than or equal top_p
        returns a list of where every item is
         a tuple of a sequence and probability
        over all complexity of the function is
        O(group_size * vocab_size * log(vocab_size))"""

        # prob_tensor.shape is now (group_size, vocab_size)
        possible_tokens = []

        already_predicted = set(TreeGenerator.flatten(org_prompt))
        for token_prob in prob_mat:  # group_size times
            token_prob: Tensor  # token_prob.shape is (vocab_size,)
            for token_id in already_predicted:
                token_prob[token_id] = 0.0
                # We never use the same token twice,
                # so the probability
                # of a token that
                # was already predicted is 0

            vocab_size = len(token_prob)  # O(1)
            indexed_prob = list(
                zip(token_prob, range(vocab_size)))
            # O(vocab_size)
            sorted_indexed_prob = sorted(
                indexed_prob, key=lambda x: x[0],
                reverse=True)

            # O(vocab_size*log(vocab_size))
            curr_k = 0
            total_prob = 0
            curr_indices = []
            for prob, token in sorted_indexed_prob:
                # O(TOP_K)
                top_p_break = total_prob + prob > self.top_p
                top_k_break = curr_k == self.top_k
                if top_p_break or top_k_break:
                    break
                if token not in already_predicted:
                    already_predicted.add(token)
                    curr_k += 1
                    total_prob += prob
                    curr_indices.append(token)
            if len(curr_indices) == 0:
                # If we didn't find any tokens to sample from,
                # we sample the most probable token
                highest_prob_token_id = sorted_indexed_prob[0][1]
                curr_indices.append(highest_prob_token_id)
                already_predicted.add(highest_prob_token_id)

            possible_tokens.append(curr_indices)  # O(1)
        new_sequences: List[List[int]]
        new_sequences = TreeGenerator.combinations(
            possible_tokens, len(org_prompt)
        )
        if len(new_sequences) == 0:
            raise NoCompletionsFound(self)
        # O(prod(len(indices[i]))
        # for i in range(group_size)))
        # len(indices[i]) < min(TOP_K, vocab_size)
        # therefore the complexity is
        # O(min(TOP_K, vocab_size) * group_size) = O(vocab_size * group_size) = O(group_size)

        return new_sequences

    def rec_gen(self, org_prompt: TokenIDS | Tensor,
                num_tokens: Optional[int],
                org_prompt_prob: float = 1.0) \
            -> Dict[Tuple[int], float]:
        """Recursively generates the next group of tokens
         in a TREE like behavior"""
        if self.end_of_sentence_id in org_prompt:
            if isinstance(org_prompt, Tensor):
                org_prompt = org_prompt.tolist()
            return {tuple(org_prompt): org_prompt_prob}
        num_groups: Optional[int] = None
        if num_tokens is not None:
            num_groups = ceil(num_tokens / self.group_size)
        tokens_list: List[int]
        if isinstance(org_prompt, list) or isinstance(org_prompt, tuple):
            tokens_list = TreeGenerator.flatten(org_prompt)
        elif isinstance(org_prompt, Tensor):
            tokens_list = org_prompt.tolist()
        else:
            raise TypeError("org_prompt must be a list or a tuple."
                            f" got {type(org_prompt)} instead")
        prob_mat: Tensor = self.get_prob_mat(tokens_list)
        tokenized_ans_list: List[List[int]] = self.generate_group(prob_mat, org_prompt)
        prob_list: List[float]
        prob_list = [TreeGenerator.seq_prob(
            seq,
            prob_mat,
            org_prompt_prob
        ) for seq in tokenized_ans_list]
        new_prompts: List[List[int]]
        ans: List[int]
        new_prompts = [tokens_list + ans
                       for ans in tokenized_ans_list]
        completion_probs: Dict[Tuple[int], float]
        completion_probs = self.remove_duplicates(
            new_prompts, prob_list, len(org_prompt))
        all_completions_ended = all(
            self.end_of_sentence_id in tokenized_ans
            for tokenized_ans in completion_probs.keys()
        )
        if num_groups == 1 or all_completions_ended:
            return completion_probs

        new_completions: Dict[Tuple[int], float] = dict()
        if num_tokens is None:
            new_number_tokens = None
        else:
            new_number_tokens = num_tokens - self.group_size
        for curr_new_prompt, curr_new_prompt_prob in completion_probs.items():
            curr_new_prompt: List[int]
            curr_completions: Dict[Tuple[int], float]
            curr_completions = self.rec_gen(
                curr_new_prompt, new_number_tokens,
                curr_new_prompt_prob)
            tokens: Tuple[int]
            prob: float
            for tokens, prob in curr_completions.items():
                new_completions[tokens] = prob
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
            num_groups = ceil(num_new_tokens / self.group_size)
            if num_groups >= sys.getrecursionlimit():
                sys.setrecursionlimit(num_groups + 100)
        seq_prob_dict: Dict[Tuple[int], float]
        seq_prob_dict = self.rec_gen(
            tokenized_prompt, num_new_tokens)
        if len(seq_prob_dict) < num_return_sequences:
            raise NoCompletionsFound(
                self,
                f"Not enough completions found,"
                f" {len(seq_prob_dict)} found"
                f" but {num_return_sequences} requested"
            )
        sorted_seq_prob_dict = sorted(
            seq_prob_dict.items(), key=lambda x: x[1], reverse=True
        )
        highest_prob_answers = sorted_seq_prob_dict[:num_return_sequences]
        assert len(highest_prob_answers) == num_return_sequences
        return [tokens for tokens, prob
                in highest_prob_answers]

    def __repr__(self):
        return f"TreeGenerator: " \
               f"model name: {self.model_name}, " \
               f"group size: {self.group_size}, " \
               f"temperature: {self.temp}, " \
               f"generation type: {self.generation_type}, " \
               f"top_p: {self.top_p}, " \
               f"top_k: {self.top_k}"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "group_size": self.group_size,
            "temp": self.temp,
            "generation_type": self.generation_type,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "end_of_sentence_stop": self.end_of_sentence_stop,
            "answer_length_multiplier": self.answer_length_multiplier
        }
