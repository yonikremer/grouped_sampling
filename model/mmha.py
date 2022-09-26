from typing import Optional

import tensorflow as tf
from tensorflow import keras, Tensor, TensorSpec
from keras.backend import floatx
from keras.layers import Layer, Softmax, Dense


class ScaledDotProductAttention(Layer):
    scale: TensorSpec(shape=(),
                      dtype=floatx())
    softmax: Softmax

    def __init__(self,
                 d_model: int,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # scale = 1 / sqrt(d_model)
        # scale = d_model ^ -0.5
        float_scale = d_model ** -0.5
        self.scale = tf.cast(float_scale, floatx())
        self.softmax = Softmax(axis=-1)

    def call(self,
             q: Tensor,
             k: Tensor,
             v: Tensor,
             mask: Optional[Tensor] = None)\
            -> Tensor:
        """Scaled Dot-Product Attention
        input:
        q: Tensor (batch_size, seq_len, d_model),
        k: Tensor (batch_size, seq_len, d_model),
        v: Tensor (batch_size, seq_len, d_model),
        mask: Optional[Tensor (batch_size, 1, 1, seq_len)]
        output: Tensor (batch_size, seq_len, d_model)"""
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        # (batch_size, seq_len, seq_len)

        # Scaled Dot-Product Attention
        scaled_attention_logits = matmul_qk * self.scale
        # (batch_size, seq_len, seq_len)
        # matmul_qk / sqrt(d_model)

        # Masking
        if mask is not None:
            # noinspection PyTypeChecker
            if floatx() == 'float16':
                # tf.float16.min is minus infinity
                scaled_attention_logits += (mask * tf.float16.min)
                # changed from -1e9 to inf to avoid overflow
            else:
                scaled_attention_logits += (mask * -1e9) 

        # Normalize
        attention_weights = self.softmax(
            scaled_attention_logits)
        # (..., seq_len_q, seq_len_k)

        # Output
        output = tf.matmul(attention_weights, v)

        return output


class MyMultiHeadAttention(Layer):
    d_model: int
    num_heads: int
    depth: int
    wq: Dense
    wk: Dense
    wv: Dense
    dense: Dense
    sdpa: ScaledDotProductAttention

    def __init__(self,
                 num_heads: int,
                 d_model: int,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) "
                             f" must be divisible "
                             f"by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // self.num_heads

        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)

        self.dense = keras.layers.Dense(d_model)
        self.sdpa = ScaledDotProductAttention(d_model)

    def split_heads(self,
                    x: Tensor) -> Tensor:
        """Split the last dimension into
         (num_heads, depth).
         Transpose the result such that the
         shape is
         (batch_size, num_heads, seq_len, depth)
        """
        batch_size = tf.shape(x)[0]
        new_shape = (batch_size, -1, self.num_heads, self.depth)
        x = tf.reshape(x, new_shape)
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self,
             v_k: Tensor,
             q: Tensor,
             mask: Tensor) \
            -> Tensor:
        """inputs:
        v_k: Tensor of shape
        (batch_size, seq_len, d_model)
        in self attention,
        keys and values are the same
        q: Tensor of shape
        (batch_size, seq_len, d_model)
        mask: Optional[Tensor]
        of shape (batch_size, seq_len)"""
        batch_size = tf.shape(q)[0]

        q: Tensor = self.wq(q)
        # (batch_size, seq_len, d_model)
        k: Tensor = self.wk(v_k)
        # (batch_size, seq_len, d_model)
        v: Tensor = self.wv(v_k)
        # (batch_size, seq_len, d_model)

        q: Tensor = self.split_heads(q)
        # (batch_size, num_heads, seq_len_q, depth)
        k: Tensor = self.split_heads(k)
        # (batch_size, num_heads, seq_len_k, depth)
        v: Tensor = self.split_heads(v)
        # (batch_size, num_heads, seq_len_v, depth)

        scaled_attention = self.sdpa(q, k, v, mask)
        # (batch_size, num_heads, seq_len_q, depth)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3]) 
        # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(
            scaled_attention,
            (batch_size, -1, self.d_model))
        # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)
        # (batch_size, seq_len_q, d_model)

        return output
