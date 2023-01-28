from typing import Tuple

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.python.ops.linalg.linalg_impl import band_part
from keras.backend import floatx


def create_masks(
        inp: Tensor,
        pad_int: int) \
        -> Tuple[Tensor, Tensor]:
    """Creates all the masks needed for the model
    input:
    inp: Tensor of shape
    (batch_size, seq_len),
    pad_int: int
    Returns:
    tuple of (padding_mask, look_ahead_mask)
    padding_mask, look_ahead_mask:
    Tensors of shape
    (batch_size, 1, 1, seq_len)"""
    seq_len = inp.shape[1]

    def create_padding_mask(
            seq: Tensor) -> Tensor:
        """Returns a padding mask for the given sequence.
        Input:
            seq:
                tf.Tensor of shape (batch_size, seq_len)
        Returns:
            tf.Tensor of shape
            (batch_size, 1, 1, seq_len)"""
        bool_mask = tf.math.equal(seq, pad_int)
        mask = tf.cast(bool_mask, floatx())
        # For every item in the sequence, 1 if it is a padding token, 0 if it is not
        # add extra dimensions to add the padding
        return mask[:, tf.newaxis, tf.newaxis, :]
        # (batch_size, 1, 1, seq_len)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.

    def create_look_ahead_mask() -> tf.Tensor:
        """Returns a look ahead mask
         for the given sequence length."""
        ones = tf.ones(
            (seq_len, seq_len),
            dtype=floatx())
        mask = 1 - band_part(ones, -1, 0)
        # (seq_len, seq_len)
        return mask

    look_ahead_mask = create_look_ahead_mask()
    # (seq_len, seq_len)
    padding_mask = create_padding_mask(inp)
    # (batch_size, 1, 1, seq_len)
    look_ahead_mask = tf.maximum(
        padding_mask,
        look_ahead_mask)
    # (batch_size, 1, 1, seq_len)

    return padding_mask, look_ahead_mask
