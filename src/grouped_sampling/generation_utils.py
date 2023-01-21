from typing import Dict, List, Iterable
from warnings import warn

from torch import cuda, LongTensor, ones, long, Tensor, cat, no_grad, full
from transformers import AutoModelForCausalLM
from torch.nn import Softmax

from .token_ids import TokenIDS
from .repetition_penalty import RepetitionPenaltyStrategy


def safe_cat_batch(batch: List[Tensor]) -> Tensor:
    if len(batch) == 0:
        raise ValueError("Cannot cat empty batch")
    if len(batch) == 1:
        return batch[0].unsqueeze(0)
    return cat(batch, dim=0)


class GroupedGenerationUtils:
    descriptive_attrs = (
        "group_size",
        "end_of_sentence_stop",
        "temp",
        "repetition_penalty_strategy",
    )

    def __init__(
            self,
            model_name: str,
            group_size: int,
            max_input_len: int,
            end_of_sentence_id: int,
            padding_id: int,
            vocab_size: int,
            repetition_penalty_strategy: RepetitionPenaltyStrategy,
            end_of_sentence_stop: bool = True,
            temp: float = 1.0,
            use_softmax: bool = True,
            **kwargs
    ):
        """initializes the model wrapper
        Args:
            model_name: the name of the model to be used
            repetition_penalty_strategy:
            the strategy to be used for the repetition penalty
            group_size: the number of next batch to be generated
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
            **kwargs: the arguments to be passed to the model
        Complexity: O(1)
        """
        self.use_softmax: bool = use_softmax
        self.end_of_sentence_id: int = end_of_sentence_id
        self.repetition_penalty_strategy: RepetitionPenaltyStrategy = repetition_penalty_strategy
        self.group_size: int = group_size
        self.max_input_len: int = max_input_len
        self.padding_id: int = padding_id
        self.end_of_sentence_stop: bool = end_of_sentence_stop
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, **kwargs
        )
        self.temp: float = temp
        self.vocab_size: int = vocab_size
        if cuda.is_available():
            self.model = self.model.cuda()

    @property
    def padding_tokens(self) -> LongTensor:
        cpu_tokens = ones(self.group_size - 1, dtype=long) * self.padding_id  # O(group_size)
        if cuda.is_available():
            return cpu_tokens.cuda()
        return cpu_tokens

    def prepare_model_kwargs(
            self, tokens: TokenIDS
    ) -> Dict[str, LongTensor]:
        """preparing the arguments for the model call
        Args:
            tokens: the batch to be sent to the model
        Returns:
            a dictionary of the arguments for the model call
        Complexity: O(group_size + n) where n is the number of batch
        """
        if not isinstance(tokens, Tensor):
            tokens = LongTensor(tokens)  # O(n)
        if cuda.is_available():
            tokens = tokens.cuda()
        padded_tokens: LongTensor = cat(
            (tokens, self.padding_tokens), dim=0
        ).unsqueeze(0)
        # the target_length of padded_tokens is n + group_size - 1
        # so creating it is O(n + group_size)
        attention_len = padded_tokens.shape[1]  # n + group_size - 1
        if attention_len > self.max_input_len:
            padded_tokens = padded_tokens[:, -self.max_input_len:]
            # O(self.max_input_len) which is constant so O(1)
            attention_len = self.max_input_len
        attention_mask: LongTensor = ones(
            [1, attention_len], dtype=long
        )
        # O(attention_len) so O(n + group_size)
        if cuda.is_available():
            padded_tokens = padded_tokens.cuda()  # O(n + group_size)
            attention_mask = attention_mask.cuda()  # O(n + group_size)
        else:
            warn("CUDA is not available, using CPU")
        return {
            "input_ids": padded_tokens,
            "attention_mask": attention_mask,
        }

    def get_logits_matrix(self, tokens: TokenIDS) -> Tensor:
        """Given a sequence of batch,
        returns the logits matrix of shape (group_size, vocab_size)
        where logits[i] is the logits vector of the i-th next token
        complexity: O(n^2 + group_size^2) where n is the target_length of the batch
        notice that the logits are not divided by the temperature in this function."""
        # define n as the number of batch in batch
        model_kwargs = self.prepare_model_kwargs(tokens)  # O(n + group_size)
        with no_grad():
            # The time complexity of causal language model`s __call__ function
            # is O(n^2) where n is the target_length of the inputs
            outputs = self.model(
                **model_kwargs
            )
            # the target_length of all the inputs is n + group_size - 1
            # so the complexity of this line is O((n + group_size - 1)^2)
            # which is O(n^2 + group_size^2 + group_size * n)
            # we now that if a > b and a, b > 1 then a^2 > ab
            # so the complexity is O(n^2 + group_size^2)
        unscaled_relevant_logits: Tensor
        unscaled_relevant_logits = outputs.logits[0, -self.group_size:, :self.vocab_size]
        # The shape of unscaled_relevant_logits is (group_size, vocab_size)
        # So the complexity of this line should be
        # O(group_size) because we are coping group_size * vocab_size
        # elements from one tensor to the another
        return unscaled_relevant_logits

    def pad_sequence(self, sequence: TokenIDS, target_length: int) -> LongTensor:
        """pads the sequence with the padding token
        Args:
            sequence: the sequence to be padded
            target_length: the target length of the padded sequence
        Returns:
            the padded sequence
        Complexity: O(n) where n is the length of the sequence
        """
        padding_tensor: LongTensor = full(
            (target_length - len(sequence),),
            self.padding_id,
        )
        return cat((LongTensor(sequence), padding_tensor), dim=0)

    def pad_batch(self, batch: List[TokenIDS], length: int) -> LongTensor:
        """pads a batch of batch
        Args:
            batch: a list of batch
            length: the target_length to pad to
        Returns:
            the padded batch as a LongTensor of size (batch_size, target_length)
        Complexity: O(target_length * batch_size)
        """
        padded_batch: List[LongTensor] = [
            self.pad_sequence(sequence=sequence, target_length=length).unsqueeze(0)
            for sequence in batch
        ]
        answer = safe_cat_batch(padded_batch)
        # assert answer.shape == (len(batch), length), f"{answer.shape} != {(len(batch), length)}"
        return answer

    def prepare_model_kwargs_batch(self, batch: List[TokenIDS]) -> Dict[str, LongTensor]:
        """preparing the arguments for the model call
        Args:
            batch: the raw batch
        Returns:
            a dictionary of the arguments for the model call
            Complexity: O(batch size *(group_size + n)) where n is the number of batch
        """
        # assert len(batch) > 0, "batch is empty"
        # assert len(batch) > 1, "use prepare_model_kwargs instead"
        padding_length = max([len(sequence) for sequence in batch]) + self.group_size - 1
        cpu_padded_tokens = self.pad_batch(batch, padding_length)
        # assert cpu_padded_tokens.shape[1] == padding_length, f"{cpu_padded_tokens.shape[1]} != {padding_length}"
        # assert cpu_padded_tokens.shape[0] == len(batch), f"{cpu_padded_tokens.shape[0]} != {len(batch)}"
        cpu_attention_mask = ones([len(batch), padding_length], dtype=long)
        if cuda.is_available():
            padded_tokens = cpu_padded_tokens.cuda()
            attention_mask = cpu_attention_mask.cuda()
        else:
            warn("CUDA is not available, using CPU")
            padded_tokens = cpu_padded_tokens
            attention_mask = cpu_attention_mask
        # assert padded_tokens.shape[1] == padding_length, f"{padded_tokens.shape[1]} != {padding_length}"
        # assert padded_tokens.shape[0] == len(batch), f"{padded_tokens.shape[0]} != {len(batch)}"
        return {
            "input_ids": padded_tokens,
            "attention_mask": attention_mask
        }

    def get_logit_mat_batch(self, batch: List[TokenIDS]) -> Tensor:
        """Given a batch of sequences of tokens,
        returns the logits matrix of shape (group_size, vocab_size)
        where logits[i] is the logits vector of the i-th next token
        complexity: O(n^2 + group_size^2) where n is the target_length of the batch
        notice that the logits are not divided by the temperature in this function."""
        # define n as the number of batch in batch
        # assert len(batch) > 0, "batch is empty"
        # assert len(batch) > 1, "use get_logit_mat instead"
        model_kwargs = self.prepare_model_kwargs_batch(batch)
        # assert model_kwargs["input_ids"].shape[0] == len(batch)
        # assert model_kwargs["attention_mask"].shape[0] == len(batch)
        # max_len = max([len(sequence) for sequence in batch])
        # assert model_kwargs["input_ids"].shape[1] == max_len + self.group_size - 1, \
        #     f"{model_kwargs['input_ids'].shape[1]} != {max_len + self.group_size - 1}"
        with no_grad():
            all_logits: Tensor = self.model(
                **model_kwargs
            ).logits
        # assert all_logits.shape[0] == len(batch), f"The shape of the logits is not correct, " \
        #                                            f"expected {len(batch)} but got {all_logits.shape[0]}"
        # assert all_logits.shape[2] == self.vocab_size, f"The shape of the logits is not correct, " \
        #                                                f"expected {self.vocab_size} but got {all_logits.shape[2]}"
        unscaled_relevant_logits: List[Tensor] = []
        for i, sequence in enumerate(batch):
            # chose stop_index such that the target_length of the sequence is group_size
            curr_relevant_logits = all_logits[
                                   i, len(sequence) - 1:len(sequence) + self.group_size, :self.vocab_size
                                   ]
            # assert curr_relevant_logits.shape == (self.group_size, self.vocab_size), \
            #     f"curr_relevant_logits.shape = {curr_relevant_logits.shape} != " \
            #     f"(self.group_size, self.vocab_size) = ({self.group_size}, {self.vocab_size})"
            un_squeezed_logits = curr_relevant_logits.unsqueeze(0)
            # assert un_squeezed_logits.shape == (1, self.group_size, self.vocab_size), \
            #     f"un_squeezed_logits.shape = {un_squeezed_logits.shape} != " \
            #     f"(1, self.group_size, self.vocab_size) = (1, {self.group_size}, {self.vocab_size})"
            unscaled_relevant_logits.append(un_squeezed_logits)

        answer = cat(unscaled_relevant_logits)
        # assert answer.shape == (len(batch), self.group_size, self.vocab_size)
        return answer

    def get_prob_mat_batch(self, tokens: List[TokenIDS], generation_start_indexes: Iterable[int]) -> Tensor:
        """Given a batch of sequences of batch,
        returns the probability matrix of shape (group_size, vocab_size)
        where logits[i] is the logits vector of the i-th next token
        complexity: O(batch size * (n^2 + group_size^2)) where n is the target_length of the longest batch sequence.
        notice that the logits are not divided by the temperature in this function."""
        unscaled_relevant_logits = self.get_logit_mat_batch(tokens)
        # assert unscaled_relevant_logits.shape[0] == len(tokens)
        # assert unscaled_relevant_logits.shape[1] == self.group_size
        # assert unscaled_relevant_logits.shape[2] == self.vocab_size, \
        #     f"{unscaled_relevant_logits.shape[2]} != {self.vocab_size}"
        if not self.end_of_sentence_stop:
            try:
                unscaled_relevant_logits[:, :, self.end_of_sentence_id] = -float('inf')
            except IndexError as e:
                print("unscaled_relevant_logits", unscaled_relevant_logits)
                print("self.end_of_sentence_id", self.end_of_sentence_id)
                raise e
        penalized_logits = self.repetition_penalty_strategy.call_batch(
            unscaled_relevant_logits, tokens, generation_start_indexes
        )
        return self.logits_to_probs(penalized_logits)

    def get_prob_mat(
            self, tokens: TokenIDS, generation_start: int
    ) -> Tensor:
        """Returns the probability matrix
         as a list of lists of floats
         Time complexity: O(n^2 + group_size^2)
         where n is the number of tokens"""
        unscaled_relevant_logits = self.get_logits_matrix(tokens)
        # O(n^2 + group_size^2)
        # unscaled_relevant_logits is a tensor of shape (group_size, vocab_size)
        if not self.end_of_sentence_stop:
            unscaled_relevant_logits[:, self.end_of_sentence_id] = -float('inf')
            # setting a vector of size vocab_size
            # so the complexity is O(group_size)
            # setting the logits to -inf so the probability will be 0
        penalized_logits = self.repetition_penalty_strategy(
            unscaled_relevant_logits, tokens, generation_start
        )
        # O(group_size * n)
        # where n is the number of tokens generated by the algorithm
        prob_tensor = self.logits_to_probs(penalized_logits)  # O(group_size)
        # We are doing a softmax operator
        # of group_size different vectors of size vocab_size
        # The complexity of the softmax for each vector is
        # O(1) because the size of the vector size is constant
        # the complexity of this line is O(group_size)
        # because we are doing group_size softmax operations
        return prob_tensor

    def logits_to_probs(self, penalized_logits: Tensor) -> Tensor:
        """Gets the logits matrix and returns the probability matrix
        Time complexity: O(group_size)"""
        if self.use_softmax:
            # if the generation type
            # is not greedy then we need to apply softmax
            # to the penalized logits
            if self.temp != 1.0:
                penalized_logits /= self.temp
                # O(group_size * vocab_size)
                # because we are dividing a matrix of size (group_size, vocab_size)
            return Softmax(dim=-1)(penalized_logits)
            # O(group_size * vocab_size) so the complexity is O(group_size)
        return penalized_logits

    def __str__(self):
        return f"GroupedGenerationUtils({self.as_dict()})"

    def __repr__(self):
        return str(self)

    def as_dict(self):
        return {attr_name: getattr(self, attr_name)
                for attr_name in self.descriptive_attrs}
