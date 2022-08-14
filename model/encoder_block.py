import tensorflow as tf
from keras.layers import Layer, Dropout, LayerNormalization

from mmha import MyMultiHeadAttention
from pwff import PointWiseFeedForwardNetwork


class EncoderBlock(Layer):
    def __init__(self, d_model: int, num_heads: int, dff: int, drop_out_rate: float, **kwargs):
        super().__init__(**kwargs)

        self.mha = MyMultiHeadAttention(num_heads = num_heads, d_model = d_model)
        self.ffn = PointWiseFeedForwardNetwork(d_model, dff)

        self.layer_norm = LayerNormalization(epsilon=1e-6)

        self.dropout = Dropout(drop_out_rate)


    def call(self, x: tf.Tensor, training: bool, mask: tf.Tensor) -> tf.Tensor:
        # all tensors are of shape (batch_size, input_seq_len, d_model)
        attn_output = self.mha(x, x, mask)
        attn_output = self.dropout(attn_output, training=training)
        # might be data leak
        out1 = self.layer_norm(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout(ffn_output, training=training)
        out2 = self.layer_norm(out1 + ffn_output)

        return out2
