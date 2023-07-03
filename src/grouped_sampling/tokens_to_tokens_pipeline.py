from torch import LongTensor

from src.grouped_sampling import GroupedGenerationUtils
from src.grouped_sampling.logits_vec_to_token import LogitVectorToTokenPipeLine


class TokensToTokensPipeLine:
    """A pipeline that gets the input as a LongTensor of tokens and returns a LongTensor of tokens."""

    def __init__(
            self,
            logits_vec_to_token_pipeline: LogitVectorToTokenPipeLine,
            wrapped_model: GroupedGenerationUtils
    ):
        self.logits_vec_to_token_pipeline = logits_vec_to_token_pipeline
        self.wrapper_model = wrapped_model

    def call_single_sequence(
            self,
            token_ids: LongTensor,
            output_length: int,
    ) -> LongTensor:
        """A single call to the model with a single sequence."""
        logits_matrix = self.wrapper_model.get_logits_matrix(token_ids, output_length)
        return self.logits_vec_to_token_pipeline.logit_matrix_to_tokens(token_ids, logits_matrix)

    def call_batch(
            self,
            token_ids: LongTensor,
            output_length: int
    ):
        """
        A single call to the model with a batch of sequences.
        Args:
            token_ids: LongTensor of shape (batch_size, input_seq_len) with the input sequences.
            output_length: The length of the output sequences. A positive integer.
        """
        if token_ids.dim() != 2 or min(token_ids.shape) == 0:
            raise ValueError(f"token_ids should be a 2D long tensor"
                             f"Got input ids with shape {token_ids.shape} and dimention {token_ids.dim()}")
        if not isinstance(output_length, int):
            raise TypeError(f"output_length should be an int, got {type(output_length)}")
        if output_length <= 0:
            raise ValueError(f"output_length should be a positive int, got {output_length}")
        logits_matrix = self.wrapper_model.get_logit_mat_batch(token_ids, output_length)
        return self.logits_vec_to_token_pipeline.batch_to_tokens(token_ids, logits_matrix)
