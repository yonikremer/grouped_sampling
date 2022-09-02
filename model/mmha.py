from typing import Optional

import tensorflow as tf
from tensorflow import keras, Tensor, TensorSpec
from keras.backend import floatx as float_type
from keras.layers import Layer, Softmax


class ScaledDotProductAttention(Layer):
    def __init__(self, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.scale: TensorSpec(shape=(), dtype=float_type())
        # scale = 1 / sqrt(d_model) = d_model ^ -0.5
        self.scale = tf.math.pow(tf.cast(d_model, float_type()), -0.5)
        self.softmax = Softmax(axis=-1)

    def call(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Scaled Dot-Product Attention
        input:
        q: Tensor of shape (batch_size, seq_len, d_model),
        k: Tensor of shape (batch_size, seq_len, d_model),
        v: Tensor of shape (batch_size, seq_len, d_model),
        mask: Optional[Tensor] of shape (batch_size, 1, 1, seq_len)
        output: Tensor of shape (batch_size, seq_len, d_model)"""
        matmul_qk: Tensor = tf.matmul(q, k, transpose_b=True)  # (batch_size, seq_len, seq_len)

        # Scaled Dot-Product Attention
        scaled_attention_logits: Tensor = matmul_qk * self.scale  # (batch_size, seq_len, seq_len)
        # matmul_qk / sqrt(d_model)

        # Masking
        if mask is not None:
            # noinspection PyTypeChecker
            if float_type() == 'float16':
                # tf.float16.min is minus infinity
                scaled_attention_logits += (mask * tf.float16.min)
                # changed from -1e9 to inf to avoid overflow
            else:
                scaled_attention_logits += (mask * -1e9) 

        # Normalize
        attention_weights = self.softmax(scaled_attention_logits)
        # (..., seq_len_q, seq_len_k)

        # Output
        output = tf.matmul(attention_weights, v)

        return output


class MyMultiHeadAttention(Layer):
    """U can use the built-in keras.layers.multihead_attention but is caused a bug for me"""
    def __init__(self, num_heads: int, d_model: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // self.num_heads

        self.wq = keras.layers.Dense(d_model)
        self.wk = keras.layers.Dense(d_model)
        self.wv = keras.layers.Dense(d_model)

        self.dense = keras.layers.Dense(d_model)
        self.sdpa = ScaledDotProductAttention(d_model)

    def split_heads(self, x: Tensor) -> Tensor:
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v_k: Tensor, q: Tensor, mask: Tensor) -> Tensor:
        """inputs:
        v_k: Tensor of shape (batch_size, seq_len, d_model) in self attention keys and values are the same
        q: Tensor of shape (batch_size, seq_len, d_model)
        mask: Optional[Tensor] of shape (batch_size, seq_len)"""
        batch_size = tf.shape(q)[0]

        q: Tensor = self.wq(q)  # (batch_size, seq_len, d_model)
        k: Tensor = self.wk(v_k)  # (batch_size, seq_len, d_model)
        v: Tensor = self.wv(v_k)  # (batch_size, seq_len, d_model)

        q: Tensor = self.split_heads(q)  # (batch_size, num_heads, seq_len_q, depth)
        k: Tensor = self.split_heads(k)  # (batch_size, num_heads, seq_len_k, depth)
        v: Tensor = self.split_heads(v)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape should be (batch_size, num_heads, seq_len_q, depth)
        scaled_attention = self.sdpa(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3]) 
         # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
          # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output
