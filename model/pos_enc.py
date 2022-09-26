import tensorflow as tf
import numpy as np
from tensorflow import Tensor
from keras.backend import floatx


def create_positional_encoding(
        max_len: int,
        d_model: int) -> Tensor:
    """Returns the positional encoding
     for a given a maximal sequence length
      and model dimension.
    inputs: max_len: int, d_model: int
    returns: Tensor of shape
    (1, max_len, d_model)"""

    def get_angles(
            positions: np.ndarray,
            timestamps: np.ndarray) \
            -> np.ndarray:
        """Returns the angle in radians for given positions,
        timestamps and the dimension of the model
        input:
        positions: np.ndarray of shape (max_len, 1),
        timestamps: np.ndarray of shape (1, d_model),
        d_model: int
        output: np.ndarray of shape (max_len, d_model)"""
        if floatx() == "float32":
            float_d_model = np.float32(d_model)
        else:
            float_d_model = np.float16(d_model)
        is_even = (2 * (timestamps // 2))
        exp1 = (is_even / float_d_model)
        angle_rates = 1 / np.power(10000, exp1)

        return positions * angle_rates

    angle_rads = get_angles(
        np.arange(max_len)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :])
    # (max_len, d_model)

    # apply sin to even indices in the array;
    # 2i for i in range(d_model // 2)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # (max_len, d_model)

    # apply cos to odd indices in the array;
    # 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    # (max_len, d_model)

    pos_encode = angle_rads[np.newaxis, ...]
    # (1, max_len, d_model)

    return tf.cast(
        pos_encode,
        dtype=floatx())
