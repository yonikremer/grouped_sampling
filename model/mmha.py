import tensorflow as tf
from typing import Optional


class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.scale: tf.TensorSpec(shape=(), dtype=tf.keras.backend.floatx())
        # scale = 1 / sqrt(d_model)
        self.scale = tf.math.pow(tf.cast(d_model, tf.keras.backend.floatx()), -0.5)
        self.softmax = tf.keras.layers.Softmax(axis=-1)

    def call(self, q: tf.Tensor, k: tf.Tensor, v: tf.Tensor, mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        """Scaled Dot-Product Attention
        input: 
        q: tf.Tensor of shape (batch_size, seq_len, d_model), 
        k: tf.Tensor of shape (batch_size, seq_len, d_model), 
        v: tf.Tensor of shape (batch_size, seq_len, d_model), 
        mask: Optional[tf.Tensor] of shape (batch_size, 1, 1, seq_len)
        output: tf.Tensor of shape (batch_size, seq_len, d_model)"""
        matmul_qk: tf.Tensor = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

        # Scaled Dot-Product Attention
        scaled_attention_logits: tf.Tensor = matmul_qk * self.scale  # (..., seq_len_q, seq_len_k)
        # matmul_qk / sqrt(d_model)

        # Masking
        if mask is not None:
            # noinspection PyTypeChecker
            if tf.keras.backend.floatx() == 'float16':
                # tf.float16.min is minus infinity
                scaled_attention_logits += (mask * tf.float16.min)  # changed from -1e9 to prevent nan's
            else:
                scaled_attention_logits += (mask * -1e9) 

        # Normalize
        attention_weights = self.softmax(scaled_attention_logits)
        # (..., seq_len_q, seq_len_k)

        # Output
        output = tf.matmul(attention_weights, v)

        return output


class MyMultiHeadAttention(tf.keras.layers.Layer):
    """U can use the built-in tf.keras.layers.multihead_attention but is caused a bug for me"""
    def __init__(self, num_heads: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)
        self.sdpa = ScaledDotProductAttention(d_model)

    def split_heads(self, x: tf.Tensor, batch_size: int) -> tf.Tensor:
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v_k: tf.Tensor, q: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """inputs:
        v_k: tf.Tensor of shape (batch_size, seq_len, d_model) in self attention keys and values are the same
        q: tf.Tensor of shape (batch_size, seq_len, d_model)
        mask: Optional[tf.Tensor] of shape (batch_size, seq_len)"""
        batch_size = tf.shape(q)[0]

        q: tf.Tensor = self.wq(q)  # (batch_size, seq_len, d_model)
        k: tf.Tensor = self.wk(v_k)  # (batch_size, seq_len, d_model)
        v: tf.Tensor = self.wv(v_k)  # (batch_size, seq_len, d_model)

        q: tf.Tensor = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k: tf.Tensor = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v: tf.Tensor = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape should be (batch_size, num_heads, seq_len_q, depth)
        scaled_attention = self.sdpa(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3]) 
         # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
          # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output
