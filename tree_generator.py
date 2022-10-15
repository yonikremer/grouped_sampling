from math import ceil
from typing import List, Dict, Union, Tuple, Sequence, Any, Optional

from transformers import BatchEncoding

from text_generator import TextGenerator, NoCompletionsFound, GenerationType

tokenIDS = Union[List[int], Tuple[int]]


class TreeGenerator(TextGenerator):
    """A TextGenerator that generates text
     in a TREE like fashion,
     without random sampling."""
    top_p: float
    top_k: int

    def __init__(self, model_name: str, group_size: int,
                 top_k: Optional[int], top_p: Optional[float], temp: float = 1.0,):
        super().__init__(model_name, group_size, temp)

        if top_p is None and top_k is None:
            self.top_p = 0.0
            self.top_k = 1
            self.generation_type = GenerationType.GREEDY
        elif top_k is None:
            self.generation_type = GenerationType.TREE
            self.top_k = self.vocab_size
            self.top_p = top_p
        elif top_p is None:
            self.generation_type = GenerationType.TREE
            self.top_k = top_k
            self.top_p = 1.0
        elif top_k == 1 or top_p == 0.0:
            self.top_p = 0.0
            self.top_k = 1
            self.generation_type = GenerationType.GREEDY
        else:
            self.generation_type = GenerationType.TREE
            self.top_k = top_k
            self.top_p = top_p

    @staticmethod
    def no_duplicates(my_sequence: List[Any]) -> bool:
        """Return if there isn't a repetition in the sequence
        complexity: O(n) where n is the length of the sequence"""
        return len(my_sequence) == len(set(my_sequence))

    @staticmethod
    def combinations(mat: Sequence[Sequence[Any]]) \
            -> List[List[Any]]:
        """Returns all the lists such that list[j] is in mat[j]
        runtime function:
        prod([len(mat[i]) for i in range(len(mat))])"""
        if len(mat) == 1:
            return [[mat[0][i]]
                    for i in range(len(mat[0]))]
        res: List[List[Any]] = []
        for i in mat[0]:
            for j in TreeGenerator.combinations(mat[1:]):
                res.append([i] + j)
        filtered_res: List[List[Any]]
        filtered_res = list(filter(
            TreeGenerator.no_duplicates,
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

    @staticmethod
    def remove_duplicates(
            completions: List[List[int]],
            probs: List[float])\
            -> Dict[Tuple[int], float]:
        """Given a list of tokenized answers
         and the probability of each completion,
        removes every repeated completion
         and every completion that have repeated tokens"""
        filtered_completions: Dict[Tuple[int], float]
        filtered_completions = dict()
        for curr_comp, curr_prob in zip(completions, probs):
            if len(curr_comp) == len(set(curr_comp)):
                curr_comp_tuple = tuple(curr_comp)
                filtered_completions[curr_comp_tuple] = curr_prob
        return filtered_completions

    def tree_grouped_sampling(
            self, prob_mat: List[List[float]],
            org_prompt: tokenIDS) -> List[List[int]]:
        """given a matrix of probabilities,
        returns a list of lists of tokens.
        the matrix is of size group_size x vocab_size
        where matrix[i, j] is
        the probability of token j the i-th token in the group
        samples the tokens such that
        for each place in the group,
        at most TOP_K tokens are sampled
        and at least one token is sampled
        and the added probability of all the tokens is
        less than or equal TOP_P
        returns a list of where every item is
         a tuple of a sequence and probability
        over all complexity of the function is
        O(group_size * vocab_size * log(vocab_size))"""

        # prob_tensor.shape is now
        # (group_size, vocab_size)
        possible_tokens = []

        already_predicted = set(TreeGenerator.flatten(org_prompt))
        for token_prob in prob_mat:  # group_size times
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
            possible_tokens)
        if len(new_sequences) == 0:
            raise NoCompletionsFound(self)
        # theta(prod(len(indices[i])
        # for i in range(group_size)))
        # len(indices[i]) < min(TOP_K, vocab_size)
        # therefore the complexity is
        # O(min(TOP_K, vocab_size) * group_size)

        return new_sequences

    def rec_gen(self, org_prompt: tokenIDS, num_tokens,
                org_prompt_prob: float = 1.0) \
            -> Dict[Tuple[int], float]:
        """Recursively generates the next group of tokens
         in a TREE like behavior"""
        num_groups = ceil(num_tokens / self.group_size)
        is_list = isinstance(org_prompt, list)
        is_tuple = isinstance(org_prompt, tuple)
        if is_list or is_tuple:
            tokens_list: List[int]
            tokens_list = list(TreeGenerator.flatten(org_prompt))
            prob_mat = self.get_prob_mat(None, tokens_list)
        else:
            raise TypeError("org_prompt must be a list or a tuple")
        tokenized_ans_list = self.tree_grouped_sampling(
            prob_mat, org_prompt)
        if len(tokenized_ans_list) == 0:
            raise NoCompletionsFound(self)
        prob_list: List[float]
        prob_list = [TreeGenerator.seq_prob(seq, prob_mat, org_prompt_prob)
                     for seq in tokenized_ans_list]
        new_prompts: List[List[int]]
        new_prompts = [TreeGenerator.flatten(tokens_list + ans)
                       for ans in tokenized_ans_list]
        if len(new_prompts) == 0:
            raise NoCompletionsFound(self)

        completion_probs: Dict[Tuple[int], float]
        completion_probs = TreeGenerator.remove_duplicates(
            new_prompts, prob_list)
        if len(completion_probs) == 0:
            raise NoCompletionsFound(self)

        if num_groups == 1:
            return completion_probs

        items = completion_probs.items()
        new_completions: Dict[Tuple[int], float] = dict()
        for curr_new_prompt, curr_new_prompt_prob in items:
            curr_new_prompt: List[int]
            curr_completions: Dict[Tuple[int], float]
            new_number_tokens = num_tokens - self.group_size
            curr_completions = self.rec_gen(
                curr_new_prompt, new_number_tokens,
                curr_new_prompt_prob)
            tokens: Tuple[int]
            prob: float
            for tokens, prob in curr_completions.items():
                new_completions[tokens] = prob
        return new_completions

    def __call__(self, prompt: str, num_new_tokens: int) -> str:
        """given a func_prompt and number of tokens to generate,
        returns a string of the func_prompt + the generated tokens"""
        if num_new_tokens == 0:
            return prompt

        tokenizer_output = self.tokenizer(prompt,
                                          return_tensors="pt")
        is_dict = isinstance(tokenizer_output, dict)
        is_batch_encoding = isinstance(tokenizer_output,
                                       BatchEncoding)
        if is_dict or is_batch_encoding:
            token_tensor = tokenizer_output["input_ids"]
        else:
            token_tensor = tokenizer_output

        tokenized_prompt: List[int]
        tokenized_prompt = token_tensor.tolist()

        seq_prob_dict: Dict[Tuple[int], float]
        seq_prob_dict = self.rec_gen(tokenized_prompt,
                                     num_new_tokens)

        highest_prob_seq: Tuple[int]
        highest_prob_seq = max(seq_prob_dict,
                               key=seq_prob_dict.get)
        final_token_list: List[int]
        final_token_list = list(highest_prob_seq)
        decoded_prompt = self.tokenizer.decode(
            final_token_list,
            skip_special_tokens=True)
        return decoded_prompt

    def __repr__(self):
        return f"SamplingGenerator: " \
               f"model name: {self.model_name}, " \
               f"group size: {self.group_size}, " \
               f"temperature: {self.temp}, " \
               f"generation type: {self.generation_type}, " \
               f"top_p: {self.top_p}, " \
               f"top_k: {self.top_k}"
