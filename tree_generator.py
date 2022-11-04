import sys
from math import ceil
from typing import List, Dict, Union, Tuple, Sequence, Any, Optional

from torch import Tensor

from text_generator import TextGenerator, NoCompletionsFound, GenerationType

tokenIDS = Union[List[int], Tuple[int]]


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
        super().__init__(
            model_name=model_name,
            group_size=group_size,
            temp=temp,
            end_of_sentence_stop=end_of_sentence_stop,
            answer_length_multiplier=answer_length_multiplier
        )

        if top_k is None and top_p is None:
            self.top_p = 1.0
            self.top_k = self.vocab_size
            self.generation_type = GenerationType.RANDOM
        elif top_p is None and top_k is not None:
            self.top_k = top_k
            if top_k == 1:
                self.generation_type = GenerationType.GREEDY
            elif top_k >= self.vocab_size:
                self.generation_type = GenerationType.RANDOM
            else:
                self.generation_type = GenerationType.TOP_K
        elif top_k is None and top_p is not None:
            self.top_p = top_p
            if top_p == 1.0:
                self.generation_type = GenerationType.RANDOM
            elif top_p == 0.0:
                self.generation_type = GenerationType.GREEDY
            else:
                self.generation_type = GenerationType.TOP_P
        else:
            raise ValueError(
                "Either top_k or top_p \
                should be set.")

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
        token in the sequence"""
        probability = org_prompt_prob
        sequence_length = len(tokens)
        for i in range(sequence_length):
            curr_token = tokens[i]
            probability *= prob_mat[i][curr_token]
        return probability

    @staticmethod
    def flatten(ids: tokenIDS) -> List:
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
         and every completion that have repeated tokens"""
        filtered_completions: Dict[Tuple[int], float]
        filtered_completions = dict()
        for curr_comp, curr_prob in zip(completions, probs):
            if self.end_of_sentence_id in curr_comp:
                end_of_sentence_index = curr_comp.index(self.end_of_sentence_id)
                completion_body = curr_comp[prompt_length:end_of_sentence_index]
            else:
                completion_body = curr_comp[prompt_length:]
            if len(completion_body) == len(set(completion_body)):
                curr_comp_tuple = tuple(curr_comp)
                filtered_completions[curr_comp_tuple] = curr_prob
        return filtered_completions

    def generate_group(
            self, prob_mat: Tensor,
            org_prompt: tokenIDS) -> List[List[int]]:
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
        # theta(prod(len(indices[i])
        # for i in range(group_size)))
        # len(indices[i]) < min(TOP_K, vocab_size)
        # therefore the complexity is
        # O(min(TOP_K, vocab_size) * group_size)

        return new_sequences

    def rec_gen(self, org_prompt: tokenIDS, num_tokens: Optional[int],
                org_prompt_prob: float = 1.0) \
            -> Dict[Tuple[int], float]:
        """Recursively generates the next group of tokens
         in a TREE like behavior"""
        if self.end_of_sentence_id in org_prompt:
            return {tuple(org_prompt): org_prompt_prob}
        num_groups: Optional[int]
        if num_tokens is None:
            num_groups = None
        else:
            num_groups = ceil(num_tokens / self.group_size)
        is_list = isinstance(org_prompt, list)
        is_tuple = isinstance(org_prompt, tuple)
        if is_list or is_tuple:
            tokens_list = TreeGenerator.flatten(org_prompt)
            prob_mat: Tensor = self.get_prob_mat(tokens_list)
        else:
            raise TypeError("org_prompt must be a list or a tuple")
        tokenized_ans_list: List[List[int]] = self.generate_group(
            prob_mat, org_prompt)
        if len(tokenized_ans_list) == 0:
            raise NoCompletionsFound(self)
        prob_list: List[float]
        prob_list = [TreeGenerator.seq_prob(
            seq,
            prob_mat,
            org_prompt_prob
        ) for seq in tokenized_ans_list]
        new_prompts: List[List[int]]
        ans: List[int]
        new_prompts = [tokens_list + TreeGenerator.flatten(ans)
                       for ans in tokenized_ans_list]
        if len(new_prompts) == 0:
            raise NoCompletionsFound(self)

        completion_probs: Dict[Tuple[int], float]
        completion_probs = self.remove_duplicates(
            new_prompts, prob_list, len(org_prompt))
        if len(completion_probs) == 0:
            raise NoCompletionsFound(self)

        if num_groups == 1:
            return completion_probs
        if all([self.end_of_sentence_id in tokenized_ans
                for tokenized_ans in completion_probs.keys()]):
            return completion_probs

        items = completion_probs.items()
        new_completions: Dict[Tuple[int], float] = dict()
        for curr_new_prompt, curr_new_prompt_prob in items:
            curr_new_prompt: List[int]
            curr_completions: Dict[Tuple[int], float]
            if num_tokens is None:
                new_number_tokens = None
            else:
                new_number_tokens = num_tokens - self.group_size
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
            tokenized_prompt: List[int],
            num_new_tokens: Optional[int] = None,
            num_return_sequences: int = 1,
    ) -> List[List[int]]:

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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "group_size": self.group_size,
            "temperature": self.temp,
            "generation_type": self.generation_type,
            "top_p": self.top_p,
            "top_k": self.top_k,

        }
