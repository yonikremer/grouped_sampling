from tensorflow import Tensor
from keras.layers import Layer, LayerNormalization, Dropout

from mmha import MyMultiHeadAttention
from pwff import PointWiseFeedForwardNetwork


class DecoderBlock(Layer):
    def __init__(self, d_model: int, num_heads: int, dff: int, rate: float, **kwargs):
        super().__init__(**kwargs)

        self.mha = MyMultiHeadAttention(num_heads = num_heads, d_model = d_model)

        self.ffn = PointWiseFeedForwardNetwork(d_model, dff)

        self.layer_norm = LayerNormalization(epsilon=1e-6)

        self.dropout = Dropout(rate)

    def call(self, x: Tensor, enc_output: Tensor, look_ahead_mask: Tensor, padding_mask: Tensor, training):
        # enc_output.shape should be (batch_size, input_seq_len, d_model)

        attn1 = self.mha(x, x, look_ahead_mask)  # (batch_size, set_size, d_model)
        attn1 = self.dropout(attn1, training=training)  # (batch_size, set_size, d_model)
        # out1 = self.layer_norm(attn1 + x)
        # might be data leak
        out1 = self.layer_norm(attn1)  # (batch_size, set_size, d_model)

        attn2 = self.mha(enc_output, out1, padding_mask)  # (batch_size, set_size, d_model)
        attn2 = self.dropout(attn2, training=training)  # (batch_size, set_size, d_model)
        out2 = self.layer_norm(attn2 + out1)  # (batch_size, set_size, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, set_size, d_model)
        ffn_output = self.dropout(ffn_output, training=training)
        out3 = self.layer_norm(ffn_output + out2)  # (batch_size, set_size, d_model)

        return out3
