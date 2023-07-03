from typing import Dict, Iterable, List
from warnings import warn

from torch import LongTensor, Tensor, cat, cuda, full, long, no_grad, ones, FloatTensor
from torch._dynamo import OptimizedModule
from torch.nn import Softmax

from .repetition_penalty import RepetitionPenaltyStrategy
from .token_ids import TokenIDS


def safe_cat_batch(batch: List[Tensor]) -> Tensor:
    if len(batch) == 0:
        raise ValueError("Cannot cat empty batch")
    if len(batch) == 1:
        return batch[0].unsqueeze(0)
    return cat(batch, dim=0)


class GroupedGenerationUtils:
    descriptive_attrs = (
        "end_of_sentence_stop",
        "temp",
        "repetition_penalty_strategy",
        "use_cuda",
    )

    base_model_args = {
        "output_attentions": False,
        "output_hidden_states": False,
    }

    def __init__(
            self,
            model: OptimizedModule,
            max_input_len: int,
            end_of_sentence_id: int,
            padding_id: int,
            vocab_size: int,
            repetition_penalty_strategy: RepetitionPenaltyStrategy,
            end_of_sentence_stop: bool = True,
            temp: float = 1.0,
            use_softmax: bool = True,
            use_cuda: bool = True,
            **kwargs,
    ):
        """
        initializes the model wrapper
        Args:
            model_name: the name of the model to be used
            repetition_penalty_strategy:
            the strategy to be used for the repetition penalty
            max_input_len: the maximum target_length of the input to the model
            padding_id: the id of the padding token
            vocab_size: the size of the vocabulary
            end_of_sentence_stop:
            whether to stop when the end of sentence token is generated
            end_of_sentence_id:
            the id of the end of sequence token
            use_softmax:
            true if the model should use softmax,
            false if it should return the logits
            use_cuda: whether to use cuda
            **kwargs: the arguments to be passed to the model
        Complexity: O(1)
        """
        self.use_softmax: bool = use_softmax
        self.end_of_sentence_id: int = end_of_sentence_id
        self.repetition_penalty_strategy: RepetitionPenaltyStrategy = (
            repetition_penalty_strategy
        )
        self.max_input_len: int = max_input_len
        self.padding_id: int = padding_id
        self.end_of_sentence_stop: bool = end_of_sentence_stop
        self.model = model
        self.temp: float = temp
        self.vocab_size: int = vocab_size
        self.use_cuda: bool = use_cuda

    def padding_tokens(self, output_length: int) -> LongTensor:
        cpu_tokens = (ones(output_length - 1, dtype=long) * self.padding_id)
        # O(output_length)
        if cuda.is_available() and self.use_cuda:
            return cpu_tokens.cuda()  # pragma: no cover
        return cpu_tokens

    def prepare_model_kwargs(self, tokens: LongTensor, output_length: int) -> Dict[str, Tensor | bool]:
        """
        preparing the arguments for the model call
        Args:
            tokens: the batch to be sent to the model
            output_length: the target_length of the output in tokens.
        Returns:
            a dictionary of the arguments for the model call
        Complexity: O(output_length + n) where n is the number of batches.
        """
        if cuda.is_available() and self.use_cuda:
            tokens = tokens.cuda()  # pragma: no cover
        padded_tokens: LongTensor = cat(
            (tokens, self.padding_tokens(output_length)),
            dim=0
        ).unsqueeze(0)
        # the target_length of padded_tokens is n + output_length - 1
        # so creating it is O(n + output_length)
        attention_len = padded_tokens.shape[1]  # n + output_length - 1
        if attention_len > self.max_input_len:
            padded_tokens = padded_tokens[:, -self.max_input_len:]
            # O(self.max_input_len) which is constant so O(1)
            attention_len = self.max_input_len
        attention_mask: Tensor = ones([1, attention_len], dtype=long)
        # O(attention_len) so O(n + output_length)
        if cuda.is_available() and self.use_cuda:
            padded_tokens = padded_tokens.cuda()  # pragma: no cover
            # O(n + output_length)
            attention_mask = attention_mask.cuda()  # pragma: no cover
            # O(n + output_length)
        else:
            warn("CUDA is not available, using CPU")
        final_kwargs = self.base_model_args
        final_kwargs["input_ids"] = padded_tokens
        final_kwargs["attention_mask"] = attention_mask
        return final_kwargs

    def get_logits_matrix(self, tokens: LongTensor, output_length: int) -> FloatTensor:
        """
        Given a sequence of batch,
        returns the logits matrix of shape (output_length, vocab_size)
        where logits[i] is the logits vector of the i-th next token
        complexity: O(n^2 + output_length^2) where n is the target_length of the batch
        notice that the logits are not divided by the temperature in this function.
        """
        # define n as the number of batch in batch
        model_kwargs = self.prepare_model_kwargs(tokens, output_length)  # O(n + output_length)
        with no_grad():
            # The time complexity of causal language model`s __call__ function
            # is O(n^2) where n is the target_length of the inputs
            outputs = self.model(**model_kwargs)
            # the target_length of all the inputs is n + output_length - 1
            # so the complexity of this line is O((n + output_length - 1)^2)
            # which is O(n^2 + output_length^2 + output_length * n)
            # we now that if a > b and a, b > 1 then a^2 > ab
            # so the complexity is O(n^2 + output_length^2)
        unscaled_relevant_logits = outputs.logits[0, -output_length:, :self.vocab_size]
        # The shape of unscaled_relevant_logits is (output_length, vocab_size)
        # So the complexity of this line should be
        # O(output_length) because we are coping output_length * vocab_size
        # elements from one tensor to the another
        return unscaled_relevant_logits

    def pad_sequence(self, sequence: TokenIDS, target_length: int) -> Tensor:
        """
        pads the sequence with the padding token
        Args:
            sequence: the sequence to be padded
            target_length: the target length of the padded sequence
        Returns:
            the padded sequence
        Complexity: O(n) where n is the length of the sequence
        """
        padding_tensor: Tensor = full(
            (target_length - len(sequence),),
            self.padding_id,
        )
        return cat((LongTensor(sequence), padding_tensor), dim=0)

    def pad_batch(self, batch: LongTensor, length: int) -> Tensor:
        """
        pads a batch of batch
        Args:
            batch: a list of batch
            length: the target_length to pad to
        Returns:
            the padded batch as a LongTensor of size (batch_size, target_length)
        Complexity: O(target_length * batch_size)
        """
        padded_batch: List[LongTensor] = [
            self.pad_sequence(sequence=sequence,
                              target_length=length).unsqueeze(0)
            for sequence in batch
        ]
        answer = safe_cat_batch(padded_batch)
        return answer

    def prepare_model_kwargs_batch(
            self, batch: LongTensor,
            output_length: int
    ) -> Dict[str, LongTensor | bool]:
        """
        preparing the arguments for the model call
        Args:
            batch: the raw batch
            output_length: the target_length of the output in tokens.
        Returns:
            a dictionary of the arguments for the model call
            Complexity: O(batch size *(output_length + n)) where n is the number of batches.
        """
        padding_length: int = (max(len(sequence) for sequence in batch) +
                               output_length - 1)
        cpu_padded_tokens = self.pad_batch(batch, padding_length)
        cpu_attention_mask = ones([len(batch), padding_length], dtype=long)
        if cuda.is_available() and self.use_cuda:
            padded_tokens = cpu_padded_tokens.cuda()  # pragma: no cover
            attention_mask = cpu_attention_mask.cuda()  # pragma: no cover
        else:
            warn("CUDA is not available, using CPU")
            padded_tokens = cpu_padded_tokens
            attention_mask = cpu_attention_mask
        final_kwargs = self.base_model_args
        final_kwargs["input_ids"] = padded_tokens
        final_kwargs["attention_mask"] = attention_mask
        return final_kwargs

    def get_logit_mat_batch(self, batch: LongTensor, output_length: int) -> FloatTensor:
        """
        Given a batch of sequences of tokens,
        returns the logits matrix of shape (output_length, vocab_size)
        where logits[i] is the logits vector of the i-th next token
        complexity: O(n^2 + output_length^2) where n is the target_length of the batch
        notice that the logits are not divided by the temperature in this function.
        """
        # define n as the number of batch in batch
        model_kwargs = self.prepare_model_kwargs_batch(batch, output_length)
        with no_grad():
            all_logits: Tensor = self.model(**model_kwargs).logits
        return all_logits[:, -output_length:, :self.vocab_size]

    def get_prob_mat_batch(
            self,
            tokens: List[TokenIDS],
            generation_start_indexes: Iterable[int],
            output_length: int,
    ) -> Tensor:
        """
        Given a batch of sequences,
        returns the probability matrix of shape (output_length, vocab_size)
        where logits[i] is the logits vector of the i-th next token
        complexity:
         O(batch size * (n^2 + output_length^2)) where n is the target_length of the longest batch sequence.
        notice that the logits are not divided by the temperature in this function.
        """
        unscaled_relevant_logits = self.get_logit_mat_batch(tokens, output_length)
        if not self.end_of_sentence_stop:
            unscaled_relevant_logits[:, :, self.end_of_sentence_id] = -float("inf")
        penalized_logits = self.repetition_penalty_strategy.call_batch(
            unscaled_relevant_logits, tokens, generation_start_indexes
        )
        return self.logits_to_probs(penalized_logits)

    def get_prob_mat(self, tokens: LongTensor, generation_start: int, output_length: int) -> Tensor:
        """
        Returns the probability matrix of shape (output_length, vocab_size)
        Time complexity: O(n^2 + output_length^2)
        where n is the number of tokens
        """
        unscaled_relevant_logits = self.get_logits_matrix(tokens, output_length)
        # O(n^2 + output_length^2)
        # unscaled_relevant_logits is a tensor of shape (output_length, vocab_size)
        if not self.end_of_sentence_stop:
            unscaled_relevant_logits[:, self.end_of_sentence_id] = -float("inf")
            # setting a vector of size vocab_size
            # so the complexity is O(group_size)
            # setting the logits to -inf so the probability will be 0
        penalized_logits = self.repetition_penalty_strategy(
            unscaled_relevant_logits, tokens, generation_start)
        # O(output_length * n)
        # where n is the number of tokens generated by the algorithm
        prob_tensor = self.logits_to_probs(penalized_logits)  # O(output_length)
        # We are doing a softmax operator
        # of output_length different vectors of size vocab_size
        # The complexity of the softmax for each vector is
        # O(1) because the size of the vector size is constant
        # the complexity of this line is O(output_length)
        # because we are doing output_length softmax operations
        return prob_tensor

    def logits_to_probs(self, penalized_logits: Tensor) -> Tensor:
        """
        Gets the logits matrix and returns the probability matrix
        Time complexity: O(output_length)
        """
        if self.use_softmax:
            # if the generation type
            # is not greedy, then we need to apply softmax
            # to the penalized logits
            if self.temp != 1.0:
                penalized_logits /= self.temp
                # O(output_length * vocab_size)
                # because we are dividing a matrix of size (output_length, vocab_size)
            return Softmax(dim=-1)(penalized_logits)
            # O(output_length * vocab_size) so the complexity is O(output_length)
        return penalized_logits

    def __str__(self):
        return f"GroupedGenerationUtils({self.as_dict()})"

    def __repr__(self):
        return str(self)

    def as_dict(self):
        return {
            attr_name: getattr(self, attr_name)
            for attr_name in self.descriptive_attrs
        }
