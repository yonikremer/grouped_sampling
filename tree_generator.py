from math import ceil
from typing import List, Dict, Union, Tuple, Sequence, Any

from transformers import BatchEncoding

from text_generator import TextGenerator


class TreeGen(TextGenerator):
    """A TextGenerator that generates text in a tree like fashion, without random sampling."""
        
    top_p: float
    top_k: int
    generation_type = "tree"

    def __init__(self, model_name: str, group_size: int, top_k: int, top_p: float, temp: float = 1.0):
        super().__init__(model_name, group_size, temp)
        self.top_k = top_k
        self.top_p = top_p


    @staticmethod
    def doesnt_have_duplicates(my_sequence: Sequence[Any]) -> bool:
        """Return if there isn't a repetition in the list
        complexity: O(n) where n is the length of the list"""
        return len(my_sequence) == len(set(my_sequence))


    @staticmethod
    def combinations(mat: Sequence[Sequence[Any]]) -> List[List[Any]]:
        """Returns all the lists such that list[j] is in mat[j]
        complexity: prod([len(mat[i]) for i in range(len(mat))])"""
        if len(mat) == 1:
            return [[mat[0][i]] for i in range(len(mat[0]))]
        res: List[List[Any]] = []
        for i in mat[0]:
            for j in TreeGen.combinations(mat[1:]):
                res.append([i] + j)
        filtered_res: List[List[Any]] = list(filter(TreeGen.doesnt_have_duplicates, res))
        return filtered_res


    @staticmethod
    def seq_prob(tokens, prob_mat, org_prompt_prob) -> float:
        """Given the probability matrix and a list of tokens
        returns the probability of the sequence
        prob_mat[a][b] is the probability of the token with id b the a-th token in the sequence"""
        probability = org_prompt_prob
        sequence_length = len(tokens)
        for i in range(sequence_length):
            curr_token = tokens[i]
            probability *= prob_mat[i][curr_token]
        return probability


    @staticmethod
    def flatten(my_list: Union[list, tuple]) -> List:
        """Gets a list where some elements might be lists
        and adds every item in the inner list to the outer list.
        example: [1, [2, 3], 4, [[5]]] -> [1, 2, 3, 4, 5]
        Complexity: O(len(the flatten list) + the number of different lists)"""
        new_list = []
        for item in my_list:
            if isinstance(item, list):
                new_list.extend(TreeGen.flatten(item))
            else:
                new_list.append(item)
        return new_list


    @staticmethod
    def remove_duplicates(completions: List[List[int]], probs: List[float]) -> Dict[Tuple[int], float]:
        """Given a list of tokenized answers and the probability of each completion,
        removes every repeated completion and every completion that have repeated tokens"""
        filtered_completions: Dict[Tuple[int], float] = dict()
        for curr_comp, curr_prob in zip(completions, probs):
            try:
                cond = len(curr_comp) == len(set(curr_comp))
            except TypeError as e:
                print(curr_comp)
                raise e
            if cond:
                curr_comp_tuple = tuple(curr_comp)
                filtered_completions[curr_comp_tuple] = curr_prob
        return filtered_completions


    def tree_grouped_sampling(self, prob_mat: List[List[float]]) -> List[List[int]]:
        """given a matrix of probabilities, returns a list of lists of tokens
        the matrix's is of size group_size x vocab_size
        where matrix[i, j] is the probability of token j the i-th token in the group
        samples the tokens such that for each place in the group,
        at most top_k tokens are sampled and at least one token is sampled
        and the added probability of all the tokens is less than or equal top_p
        returns a list of where every item is a tuple of a sequence and probability
        over all complexity of the function in O(group_size * vocab_size * log(vocab_size))"""

        # prob_tensor.shape is now (group_size, vocab_size)
        possible_tokens = []
        already_predicted = set()
        for token_prob in prob_mat:  # group_size times
            vocab_size = len(token_prob)  # O(1)
            indexed_prob = list(zip(token_prob, range(vocab_size)))  # O(vocab_size)
            sorted_indexed_prob = sorted(indexed_prob, 
                                  TextGenerator.get_second_item,
                                  reverse=True)  
            # O(vocab_size*log(vocab_size))
            curr_k = 0
            total_prob = 0
            curr_indices = []
            for prob, token in sorted_indexed_prob:
                # O(top_k)
                top_p_break = total_prob + prob > self.top_p
                top_k_break = curr_k == self.top_k
                if top_p_break or top_k_break:
                    break
                if token not in already_predicted:
                    already_predicted.add(token)
                    curr_k += 1
                    total_prob += prob
                    curr_indices.append(token)
            possible_tokens.append(curr_indices)  # O(1)
        new_sequences: List[List[int]] = TreeGen.combinations(possible_tokens)
        # theta(prod(len(indices[i]) for i in range(group_size)))
        # len(indices[i]) < min(top_k, vocab_size)
        # therefore the complexity is O(min(top_k, vocab_size) * group_size)
        return new_sequences

    def rec_gen(self, org_prompt, num_tokens,
                org_prompt_prob: float = 1.0) -> Dict[Tuple[int], float]:
        """Recursively generates the next group of tokens in a tree like behavior"""
        num_groups = ceil(num_tokens / self.group_size)
        if isinstance(org_prompt, list) or isinstance(org_prompt, tuple):
            tokenized_prompt_list = TreeGen.flatten(org_prompt)
            str_prompt = self.tokenizer.decode(tokenized_prompt_list)
        else:
            print(org_prompt)
            raise ValueError("org_prompt must be a string or list of integers")

        prob_mat = self.get_prob_mat(str_prompt, self.group_size)
        tokenized_ans_list = TreeGen.tree_grouped_sampling(prob_mat)
        prob_list: List[float]
        prob_list = [TreeGen.seq_prob(seq, prob_mat, org_prompt_prob) for seq in tokenized_ans_list]
        new_prompts: List[List[int]]
        new_prompts = [TreeGen.flatten(tokenized_prompt_list + ans) for ans in tokenized_ans_list]

        completion_probs: Dict[Tuple[int], float]
        completion_probs = TreeGen.remove_duplicates(new_prompts, prob_list)
        items = completion_probs.items()

        if num_groups == 1:
            shorten_completions = {k[:num_tokens]: v for k, v in items}
            return shorten_completions

        new_completions: Dict[Tuple[int], float] = dict()
        for curr_new_prompt, curr_new_prompt_prob in items:
            curr_completions: Dict[Tuple[int], float]
            curr_completions = self.rec_gen(curr_new_prompt, 
                                            num_tokens - self.group_size,
                                            curr_new_prompt_prob)
            tokens: Tuple[int]
            prob: float
            for tokens, prob in curr_completions.items():
                new_completions[tokens] = prob
        return new_completions

    def __call__(self, prompt: str, num_new_tokens: int) -> str:
        tokenized_prompt_ten = self.tokenizer(prompt, return_tensors="pt")
        is_dict = isinstance(tokenized_prompt_ten, dict)
        is_batch_encoding = isinstance(tokenized_prompt_ten, BatchEncoding)
        if is_dict or is_batch_encoding:
            if "input_ids" in tokenized_prompt_ten.keys():
                tokenized_prompt_ten = tokenized_prompt_ten["input_ids"]

        final_num_tokens = tokenized_prompt_ten.shape[1] + num_new_tokens
        
        tokenized_prompt: List[int]
        tokenized_prompt = tokenized_prompt_ten.tolist()

        seq_prob_dict: Dict[Tuple[int], float]
        seq_prob_dict = self.rec_gen(tokenized_prompt, num_new_tokens)
        
        highest_prob_seq: Tuple[int]
        highest_prob_seq = max(seq_prob_dict, key=seq_prob_dict.get)
        decoded_prompt = self.tokenizer.decode(highest_prob_seq[:final_num_tokens])
        return decoded_prompt
