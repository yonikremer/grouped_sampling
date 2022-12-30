from typing import Dict
from warnings import warn

from torch import cuda, LongTensor, ones, long, Tensor, cat, no_grad, Size
from transformers import PreTrainedModel, AutoModelForCausalLM
from torch.nn import Softmax

from globals import TokenIDS


class ModelWrapper:
    repetition_penalty_theta: float
    padding_id: int
    max_input_len: int
    model: PreTrainedModel
    group_size: int
    vocab_size: int
    end_of_sentence_id: int
    end_of_sentence_stop: bool
    temp: float
    descriptive_attrs = (
        "repetition_penalty_theta",
        "group_size",
        "end_of_sentence_stop",
        "temp",
    )

    def __init__(
            self,
            model_name: str,
            group_size: int,
            max_input_len: int,
            end_of_sentence_id: int,
            padding_id: int,
            vocab_size: int,
            end_of_sentence_stop: bool = True,
            repetition_penalty_theta: float = 1.2,
            temp: float = 1.0,
            use_softmax: bool = True,
            **kwargs
    ):
        """initializes the model wrapper
        Args:
            model_name: the name of the model to be used
            repetition_penalty_theta: the theta parameter of the repetition penalty
            group_size: the number of next tokens to be generated
            max_input_len: the maximum length of the input to the model
            padding_id: the id of the padding token
            vocab_size: the size of the vocabulary
            end_of_sentence_stop: whether to stop when the end of sentence token is generated
            end_of_sentence_id: the id of the end of sequence token
            use_softmax: true if the model should use softmax, false if it should return the logits
            **kwargs: the arguments to be passed to the model
        Complexity: O(1)
        """
        self.use_softmax = use_softmax
        self.end_of_sentence_id = end_of_sentence_id
        self.repetition_penalty_theta = repetition_penalty_theta if repetition_penalty_theta is not None else 1.2
        self.group_size = group_size
        self.max_input_len = max_input_len
        self.padding_id = padding_id
        self.end_of_sentence_stop = end_of_sentence_stop
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.temp = temp
        self.vocab_size = vocab_size
        if cuda.is_available():
            self.model = self.model.cuda()

    @property
    def padding_tokens(self) -> LongTensor:
        cpu_tokens = ones(self.group_size, dtype=long) * self.padding_id
        if cuda.is_available():
            return cpu_tokens.cuda()
        return cpu_tokens

    def prepare_model_kwargs(self, tokens: TokenIDS) -> Dict[str, LongTensor]:
        """preparing the arguments for the model call
        Args:
            tokens: the tokens to be sent to the model
        Returns:
            a dictionary of the arguments for the model call
        Complexity: O(group_size + n) where n is the number of tokens
        """
        if not isinstance(tokens, Tensor):
            tokens = LongTensor(tokens)  # O(n)
        padded_tokens: LongTensor = cat((tokens, self.padding_tokens), dim=0).unsqueeze(0)
        # the length of padded_tokens is n + group_size - 1
        # so creating it is O(n + group_size)
        attention_len = padded_tokens.shape[1]  # n + group_size - 1
        if attention_len > self.max_input_len:
            padded_tokens = padded_tokens[:, -self.max_input_len:]
            # O(self.max_input_len) which is constant so O(1)
            attention_len = self.max_input_len
        attention_mask: LongTensor = ones([1, attention_len], dtype=long)
        # O(attention_len) so O(n + group_size)
        if cuda.is_available():
            padded_tokens = padded_tokens.cuda()  # O(n + group_size)
            attention_mask = attention_mask.cuda()  # O(n + group_size)
        else:
            warn("CUDA is not available, using CPU")
        assert padded_tokens.shape == Size([1, attention_len]), f"padded_tokens shape is {padded_tokens.shape}" \
                                                                f" while attention_len is {attention_len}"
        assert attention_mask.shape == Size([1, attention_len]), f"attention_mask shape is {attention_mask.shape} " \
                                                                 f"while attention_len is {attention_len}"
        return {
            "input_ids": padded_tokens,
            "attention_mask": attention_mask,
        }

    def penalize_logits(self, logits: Tensor, tokens: TokenIDS, generation_start: int) -> Tensor:
        """applies repetition penalty,
        using the repetition_penalty_theta parameter defined in the class
        the formula from the paper is:
        softmax(original_logits, T) = exp(original_logits_i/(T · h(i)) / sum(exp(original_logits_i/(T · h(i)))
        which is equivalent to:
        pi = exp((xi/h(i)) / T / sum(exp(xj/(T · h(j)))
        and to:
        penalized_logits = original_logits / h(i)
        pi = softmax(original_logits / h(i), T)
        where:
            h(i) = θ if i in generated_tokens else 1
            T is the temperature parameter of softmax
        Args:
            logits: the logits matrix of shape (group_size, vocab_size)
            tokens: the sequence of tokes from the prompt and the generated
            generation_start: The index of the first
        Returns:
            original_logits / h(i)
        complexity: O(group_size * n) where n is the number of tokens generated by the algorithm
        """
        if self.repetition_penalty_theta == 1.0:
            # There is no point diving by zero so in this case there is no need for the function.
            return logits
        generated_tokens = set(tokens[generation_start:])  # tokens[generation_start:] has maximum size of n
        for token_id in generated_tokens:  # len(generated_tokens) < max(n, vocab_size)
            logits[:, token_id] /= self.repetition_penalty_theta
            # O(group_size) because we are taking a slice of size group_size
        return logits

    def get_logits_matrix(self, tokens: TokenIDS) -> Tensor:
        """Given a sequence of tokens,
        returns the logits matrix of shape (group_size, vocab_size)
        where logits[i] is the logits vector of the i-th next token
        complexity: O(n^2 + group_size^2) where n is the length of the tokens
        notice that the logits are not divided by the temperature in this function."""
        # define n as the number of tokens in tokens
        model_kwargs = self.prepare_model_kwargs(tokens)  # O(n + group_size)
        with no_grad():
            # The time complexity of causal language model`s __call__ function
            # is O(n^2) where n is the length of the inputs
            outputs = self.model(
                **model_kwargs
            )
            # the length of all the inputs is n + group_size - 1
            # so the complexity of this line is O((n + group_size - 1)^2)
            # which is O(n^2 + group_size^2 + group_size * n)
            # we now that if a > b and a, b > 1 then a^2 > ab so the complexity is O(n^2 + group_size^2)
        unscaled_relevant_logits: Tensor = outputs.logits[0, -self.group_size:, :self.vocab_size]
        # The shape of unscaled_relevant_logits is (group_size, vocab_size)
        # So the complexity of this line should be O(group_size) because we are coping group_size * vocab_size
        # elements from one tensor to the another
        return unscaled_relevant_logits

    def get_prob_mat(self, tokens: TokenIDS, generation_start: int) -> Tensor:
        """Returns the probability matrix
         as a list of lists of floats
         Time complexity: O(n^2 + group_size^2)
         where n is the number of tokens"""
        unscaled_relevant_logits = self.get_logits_matrix(tokens)  # O(n^2 + group_size^2)
        # unscaled_relevant_logits is a tensor of shape (group_size, vocab_size)
        if not self.end_of_sentence_stop:
            unscaled_relevant_logits[:, self.end_of_sentence_id] = -float('inf')
            # setting a vector of size vocab_size so the complexity is O(group_size)
            # setting the logits to -inf so the probability will be 0
        penalized_logits = self.penalize_logits(unscaled_relevant_logits, tokens, generation_start)
        # O(group_size * n) where n is the number of tokens generated by the algorithm
        prob_tensor = self.logits_to_probs(penalized_logits)  # O(group_size)
        # We are doing a softmax operator of group_size different vectors of size vocab_size
        # The complexity of the softmax for each vector is O(1) because the size of the vector is constant
        # the complexity of this line is O(group_size) because we are doing group_size softmax operations
        return prob_tensor

    def logits_to_probs(self, penalized_logits: Tensor) -> Tensor:
        """Gets the logits matrix and returns the probability matrix
        Time complexity: O(group_size)"""
        if self.use_softmax:
            # if the generation type is not greedy then we need to apply softmax
            # to the penalized logits
            if self.temp != 1.0:
                penalized_logits /= self.temp
                # O(group_size * vocab_size) because we are dividing a matrix of size (group_size, vocab_size)
            return Softmax(dim=1)(penalized_logits)  # O(group_size * vocab_size) so the complexity is O(group_size)
        return penalized_logits

    def __str__(self):
        return f"ModelWrapper({self.as_dict()})"

    def __repr__(self):
        return str(self)

    def as_dict(self):
        return {attr_name: getattr(self, attr_name) for attr_name in self.descriptive_attrs}
