import tensorflow as tf
import numpy as np


def create_positional_encoding(max_len: int, d_model: int) -> tf.Tensor:
    """Returns the positional encoding for a given a maximal sequence length and model dimension.
    inputs: max_len: int, d_model: int
    returns: tf.Tensor of shape (1, max_len, d_model) and dtype tf.keras.backend.floatx()
    The 1 is for the batch dimension, the place in the batch dimension does not matter"""

    def get_angles(positions: np.ndarray, timestamps: np.ndarray, d_model: int) -> np.ndarray:
        """Returns the angle in radians for given positions, timestamps and the dimension of the model
        input: positions: np.ndarray of shape (max_len, 1), timestamps: np.ndarray of shape (1, d_model), d_model: int
        output: np.ndarray of shape (max_len, d_model)"""
        if tf.keras.backend.floatx() == "float32":
            angle_rates = 1 / np.power(10000, ((2 * (timestamps//2)) / np.float32(d_model)))
        else:
            angle_rates = 1 / np.power(10000, ((2 * (timestamps//2)) / np.float16(d_model)))

        return positions * angle_rates
    
    angle_rads = get_angles(np.arange(max_len)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)  # (max_len, d_model)

    # apply sin to even indices in the array; 2i for i in range(d_model // 2)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])  # (max_len, d_model)

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])  # (max_len, d_model)

    pos_encode = angle_rads[np.newaxis, ...]  # (1, max_len, d_model)

    return tf.cast(pos_encode, dtype=tf.keras.backend.floatx())
