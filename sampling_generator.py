import heapq
from collections.abc import Iterator
from random import seed
from typing import Callable, List, Dict, Optional, Any

from torch import Tensor, zeros, argmax, multinomial, manual_seed, isclose, tensor

from text_generator import TextGenerator, GenerationType

ProbDict = Dict[int, Tensor]


class ChangingSeed(Iterator):
    """Context manager for changing the seed of the random module.
    How to use:
    with ChangingSeed(first_seed, number_of_different_seeds) as changing_seed:
        for _ in changing_seed:
            # do something with random module"""
    default_seed: int
    curr_seed: int
    max_num_calls: int
    curr_num_calls: int

    def __init__(self, default_seed: int, max_num_calls: int):
        self.default_seed = default_seed
        self.curr_seed = self.default_seed
        self.max_num_calls = max_num_calls
        self.curr_num_calls = 0

    def __enter__(self):
        self.curr_num_calls = 0
        self.curr_seed = self.default_seed
        return self

    def __exit__(self, *args):
        self.curr_seed = self.default_seed
        seed(self.default_seed)
        manual_seed(self.default_seed)

    def __iter__(self):
        self.curr_seed = self.default_seed
        return self

    def __next__(self):
        self.curr_seed += 1
        seed(self.curr_seed)
        manual_seed(self.curr_seed)
        self.curr_num_calls += 1
        if self.curr_num_calls > self.max_num_calls:
            raise StopIteration


class TokenProb:
    """Class for storing the probability of a token and the token itself.
    Used to store the probabilities of the next tokens in the sampling generator.
    Is useful because it supports the < and > operators, which are used in the
    heapq module
    The < and > are the opposite of each other because the heapq module is only supporting minimum heaps
    and I need a maximum heap"""
    __slots__ = ['token_id', 'prob']
    token_id: int
    prob: Tensor  # of shape (1,) and dtype float

    def __init__(self, token_id: int, prob: Tensor):
        self.token_id = token_id
        self.prob = prob

    def __lt__(self, other: "TokenProb"):
        """Overrides the < operator
        Comparison is done by the probability"""
        return self.prob > other.prob

    def __gt__(self, other: "TokenProb"):
        """Overrides the > operator
        Comparison is done by the probability"""
        return self.prob < other.prob


class SamplingGenerator(TextGenerator):
    """A TextGenerator that generates text
    using random sampling
    with top-k or top-p filtering."""
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    default_seed: int = 0
    seed(default_seed)
    manual_seed(default_seed)

    def __init__(self, model_name: str, group_size: int,
                 temp: float = 1.0, top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 end_of_sentence_stop: bool = False,
                 answer_length_multiplier: int = 16, ):
        self.top_p, self.top_k = top_p, top_k
        super().__init__(
            model_name=model_name,
            group_size=group_size,
            temp=temp,
            end_of_sentence_stop=end_of_sentence_stop,
            answer_length_multiplier=answer_length_multiplier,
        )

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if key == "default_seed":
            seed(value)
            manual_seed(value)

    @property
    def generation_type(self) -> GenerationType:
        top_p = self.top_p
        top_k = self.top_k
        if top_k is None and top_p is None:
            return GenerationType.RANDOM
        if top_k == 1 or top_p == 0.0:
            return GenerationType.GREEDY
        if top_k is not None:
            if top_k < self.vocab_size:
                return GenerationType.TOP_K
            return GenerationType.RANDOM
        if top_p is not None:
            if top_p < 1.0:
                return GenerationType.TOP_P
            return GenerationType.RANDOM
        raise RuntimeError("Uncovered case in generation_type property")

    @property
    def sampling_func(self) -> Callable[[Tensor], int]:
        gen_type_to_filter_method: Dict[GenerationType, Callable[[Tensor, ], int]] = {
            GenerationType.TOP_K: self.top_k_sampling,
            GenerationType.TOP_P: self.top_p_sampling,
            GenerationType.GREEDY: SamplingGenerator.highest_prob_token,
            GenerationType.RANDOM: SamplingGenerator.unfiltered_sampling,
        }
        return gen_type_to_filter_method[self.generation_type]

    @staticmethod
    def is_multinomial(prob_vec: Tensor) -> bool:
        """Checks if a probability vector is a valid multinomial distribution.
        A valid multinomial distribution is a probability vector of shape (vocab_size,)
        where the sum of all the probabilities is 1
        and each probability is between 0 and 1"""
        if any(prob < 0 or prob > 1 for prob in prob_vec):
            return False
        prob_sum = prob_vec.sum()
        return bool(isclose(prob_sum, tensor(1.0, dtype=prob_sum.dtype)))

    @staticmethod
    def unfiltered_sampling(prob_vec: Tensor) -> int:
        """A filtering function that doesn't filter any tokens.
        returns a random token id sampled from the probability vector"""
        if not SamplingGenerator.is_multinomial(prob_vec):
            prob_vec = prob_vec / prob_vec.sum()
            assert SamplingGenerator.is_multinomial(prob_vec), \
                f"prob_vec is not a valid multinomial distribution." \
                f"prob_vec is {prob_vec}"
        return multinomial(prob_vec, 1).item()

    @staticmethod
    def highest_prob_token(prob_vec: Tensor) -> int:
        """Gets a probability vector of shape (vocab_size,)
        returns the token id with the highest probability"""
        return argmax(prob_vec).item()

    def top_p_sampling(self, prob_vec: Tensor) -> int:
        """Gets a probability vector of shape (vocab_size,)
        computes a probability vector with the top p tokens
        such that their sum in the original vector is <= self.top_p.
        and samples from that vector.
        If token with the highest probability have a probability higher than top_p, it will be sampled"""
        prob_sum: float = 0.0
        converted_probs = [TokenProb(i, prob) for i, prob in enumerate(prob_vec)]
        heapq.heapify(converted_probs)
        new_probs = zeros(prob_vec.shape, dtype=float)
        while prob_sum < self.top_p and len(converted_probs) > 0:
            curr_prob: TokenProb = heapq.heappop(converted_probs)
            token_id = curr_prob.token_id
            if curr_prob.prob <= 0.0:
                break
            if curr_prob.prob > 1:
                raise ValueError(f"Probability of token {token_id}  in the vector {prob_vec} is {curr_prob.prob}"
                                 f" which is higher than 1")
            prob_sum += curr_prob.prob
            new_probs[token_id] = curr_prob.prob
        if prob_sum == 0.0:
            return converted_probs[0].token_id
        scaled_new_probs = new_probs / prob_sum
        assert SamplingGenerator.is_multinomial(new_probs), \
            f"scaled_new_probs is not a valid multinomial distribution: {scaled_new_probs}"
        return multinomial(scaled_new_probs, 1).item()

    def top_k_sampling(self, prob_vec: Tensor) -> int:
        """Gets a token id: probability mapping
        returns the TOP_K tokens
        with the highest probability.
        this is the bottleneck of the sampling generator."""
        top_k_keys: List[int] = heapq.nlargest(self.top_k, range(prob_vec.shape[0]), key=lambda x: prob_vec[x])
        prob_sum = sum(prob_vec[token_id] for token_id in top_k_keys)
        new_probs = zeros(prob_vec.shape, dtype=float)
        for token_id in top_k_keys:
            new_probs[token_id] = prob_vec[token_id] / prob_sum
        assert SamplingGenerator.is_multinomial(new_probs), \
            f"new_probs is not a valid multinomial distribution: {new_probs}"
        return multinomial(new_probs, 1).item()

    def generate_group(self, prob_mat: Tensor) -> List[int]:
        """Generates a group of tokens
         using the choice_function.
         Complexity: O(group_size * org_used_tokens + group_size ^ 2)"""
        prob_mat.cpu()
        # coping a tensor of size (group_size, vocab_size)
        # so the complexity is O(group_size) (vocab_size is constant)
        new_group: List[int] = []
        for curr_token_probs in prob_mat:  # group_size iterations
            curr_token_probs: Tensor
            # so the complexity of the inner loop is O(group_size + len(org_used_tokens))
            # for tokens with index >= self.vocab_size
            if len(curr_token_probs) > self.vocab_size:
                curr_token_probs = curr_token_probs[:self.vocab_size]
                # coping a tensor of constant size, so it's O(1)
            sampled_token: int = self.sampling_func(curr_token_probs)
            # Constant input size so the complexity is O(1)
            new_group.append(sampled_token)
            # appending to a list is O(1)
            if sampled_token == self.end_of_sentence_id:
                break
        del prob_mat
        return new_group

    def _forward(
            self,
            tokenized_prompt: Tensor,
            num_new_tokens: Optional[int] = None,
            num_return_sequences: int = 1,
    ) -> List[List[int]]:
        """Complexity: O(num_return_sequences * ((n ^ 3) / group_size + (n * l ^ 2) / group_size + group_size))
        where l is the number of tokens in the prompt
        and n is the number of new tokens to generate"""
        # let's define l = len(tokenized_prompt), n = num_new_tokens
        answers: List[List[int]] = []
        curr_token_list: List[int] = tokenized_prompt.tolist()
        # coping a tensor of size lso O(l)
        if num_new_tokens is None:
            raise RuntimeError("num_new_tokens is None")
        for _ in ChangingSeed(
                default_seed=self.default_seed,
                max_num_calls=num_return_sequences):  # num_return_sequences iterations
            for _ in range(num_new_tokens // self.group_size):
                # and each iteration is O(n ^ 2 + l ^ 2 + group_size ^ 2)
                # so the complexity of the loop is O((n ^ 3) / group_size + (n * l ^ 2) / group_size + group_size)
                prob_mat: Tensor = self.get_prob_mat(curr_token_list, len(tokenized_prompt))
                # complexity: O(group_size ^ 2 + len(curr_token_list) ^ 2)
                # len(curr_token_list) <= n + l
                # so the complexity is O(group_size ^ 2 + (n + l) ^ 2) = O(n ^ 2 + nl + l ^ 2 + group_size ^ 2)
                # but nl <= max(n^2, l^2) so the complexity is O(n ^ 2 + l ^ 2 + group_size ^ 2)
                new_tokens = self.generate_group(prob_mat)
                # complexity: O(group_size * len(curr_token_list) + group_size ^ 2)
                # len(curr_token_list) <= n + l
                # so the complexity is O(group_size * (n + l + group_size))
                # len(new_tokens) = group_size
                if self.end_of_sentence_id in new_tokens:  # the check is O(group_size)
                    end_of_sentence_index = new_tokens.index(self.end_of_sentence_id)
                    # O(group_size) because len(new_tokens) <= group_size
                    new_tokens = new_tokens[:end_of_sentence_index]
                    # O(group_size) because end_of_sentence_index < group_size
                curr_token_list.extend(new_tokens)
                # O(group_size) because len(new_tokens) <= group_size
            answers.append(curr_token_list)
            # O(1)
        return answers

    def __repr__(self):
        return f"SamplingGenerator: " \
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
