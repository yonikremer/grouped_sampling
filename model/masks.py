import tensorflow as tf
from tensorflow import Tensor
from typing import Tuple


def create_masks(inp: Tensor, tar: Tensor, pad_ten: Tensor) -> Tuple[Tensor, Tensor]:
    """Creates all the masks needed for the model
    input: inp: tf.Tensor of shape (batch_size, seq_len), tar: tf.Tensor of shape (batch_size, set_size)
    Returns: tuple of (padding_mask, look_ahead_mask)
    padding_mask, look_ahead_mask: tf.Tensor of shape (batch_size, 1, 1, seq_len)"""

    def create_padding_mask(seq: Tensor) -> Tensor:
            """Returns a padding mask for the given sequence.
            input: seq: tf.Tensor of shape (batch_size, seq_len)
            Returns: tf.Tensor of shape (batch_size, 1, 1, seq_len)"""
            seq = tf.cast(tf.math.equal(seq, pad_ten), tf.keras.backend.floatx())
            # For every item in the sequence, 1 if it is a padding token, 0 if it is not
            # add extra dimensions to add the padding
            return seq[:, tf.newaxis, tf.newaxis, :]
            # (batch_size, 1, 1, seq_len)

    # Encoder padding mask
    padding_mask: tf.Tensor = create_padding_mask(inp)  # (batch_size, 1, 1, seq_len)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    set_size: int = tar.shape[1]

    def create_look_ahead_mask(set_size: int) -> tf.Tensor:
        mask = 1 - tf.linalg.band_part(tf.ones((set_size, set_size)), -1, 0)
        mask = tf.cast(mask, dtype=tf.keras.backend.floatx())
        return mask
        # (seq_len, seq_len)

    look_ahead_mask = create_look_ahead_mask(set_size)
    # (seq_len, seq_len)
    dec_target_padding_mask = create_padding_mask(tar)
    # (batch_size, 1, 1, seq_len)
    look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    # (batch_size, 1, 1, seq_len)

    return padding_mask, look_ahead_mask
