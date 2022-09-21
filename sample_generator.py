from random import choices, seed
from copy import deepcopy
from math import ceil
from typing import Callable, List, Dict, Optional

from transformers import BatchEncoding

from text_generator import TextGenerator


second_item = lambda sliceable: sliceable[1]


class SampleGen(TextGenerator):
    """A TextGenerator that generates text 
    using random sampling with top-k or top-p filtering."""
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    sampling_function: Callable

    generation_type = "sampling"

    def __init__(self, model_name: str, group_size: int, 
                temp: float = 1.0, top_k: Optional[float] = None,
                top_p: Optional[float] = None):
        super().__init__(model_name, group_size, temp)
        seed(0)
        if top_p is None and top_k is not None:
            self.top_k = top_k
            self.sampling_function = self.grouped_top_k_sampling
        elif top_k is None and top_p is not None:
            self.top_p = top_p
            self.sampling_function = self.grouped_top_p_sampling
        else:
            raise ValueError("Either top_k or top_p should be set.")

    def grouped_top_p_sampling(self, prob_mat: List[List[float]],
                               org_used_tokens: List[int]):
        """Generates a group of tokens using top-p sampling."""
        used_tokens = deepcopy(org_used_tokens)
        answer = []
        for curr_token_prob_list in prob_mat:
            for used_token in used_tokens:
                curr_token_prob_list[used_token] = 0.0

            indexed_prob: Dict[int, float]
            indexed_prob = {i: prob for i, prob in enumerate(curr_token_prob_list)}
            # O(vocab_size)
            sorted_probs: Dict[int, float] 
            sorted_probs = sorted(indexed_prob.items(), key=second_item, reverse=True)
            top_p_probs: Dict[int, float] = {}
            prob_sum: float = 0.0
            for i, (curr_token, curr_prob) in enumerate(sorted_probs.items()):
                if i > 0 and prob_sum + curr_prob > self.top_p:
                    break
                prob_sum += curr_prob
                top_p_probs[curr_token] = curr_prob

            weighted_probs = {k: v / prob_sum for k, v in top_p_probs.items()}
            keys_list = list(weighted_probs.keys())
            probs = weighted_probs.values()
            sampled_token: int = choices(keys_list, weights = probs, k=1)[0]
            answer.append(sampled_token)
            used_tokens.append(sampled_token)
        return answer

    def grouped_top_k_sampling(self, prob_mat: List[List[float]],
                               org_used_tokens: List[int]):
        """Generates a group of tokens using top-k sampling."""
        used_tokens = deepcopy(org_used_tokens)
        answer = []
        for curr_token_probs in prob_mat:
            for used_token in used_tokens:
                curr_token_probs[used_token] = 0.0

            indexed_prob: Dict[int, float]
            indexed_prob = {i: prob for i, prob in enumerate(curr_token_probs)}
            # O(vocab_size)
            sorted_items = sorted(indexed_prob.items(), key=second_item, reverse=True)
            sorted_indexed_prob = {key: value for key, value in sorted_items}
            sorted_top_k_keys = list(sorted_indexed_prob.keys())[:self.top_k]
            top_k_probs: Dict[int, float]
            top_k_probs = {k: sorted_indexed_prob[k] for k in sorted_top_k_keys}
            prob_sum: float = sum(top_k_probs.values())
            weighted_probs = {k: v / prob_sum for k, v in top_k_probs.items()}
            keys_list = list(weighted_probs.keys())
            sampled_token: int = choices(keys_list, weighted_probs.values(), k=1)[0]
            answer.append(sampled_token)
            used_tokens.append(sampled_token)
        return answer

    def __call__(self, prompt: str, num_new_tokens: int) -> str:
        num_groups = ceil(num_new_tokens / self.group_size)
        tokenized_prompt_ten = self.tokenizer(prompt, return_tensors="pt")
        if isinstance(tokenized_prompt_ten, BatchEncoding):
            if "input_ids" in tokenized_prompt_ten.keys():
                tokenized_prompt_ten = tokenized_prompt_ten["input_ids"]

        curr_token_list = tokenized_prompt_ten.squeeze().tolist()

        for _ in range(num_groups):
            prob_mat = self.get_prob_mat(None, curr_token_list)
            new_tokens = self.sampling_function(prob_mat, curr_token_list)
            curr_token_list.extend(new_tokens)

        final_num_tokens = tokenized_prompt_ten.shape[1] + num_new_tokens
        shorten_token_list = curr_token_list[:final_num_tokens]
        final_ans = self.tokenizer.decode(shorten_token_list)
        return final_ans
