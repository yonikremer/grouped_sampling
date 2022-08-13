import tensorflow as tf
from tensorflow import Tensor
from keras import backend
from keras.layers import Layer, LayerNormalization, Dropout

from encoder_block import EncoderBlock

class Encoder(Layer):
    """The encoder part of the model"""
    def __init__(self, pos_encoding: Tensor, num_blocks: int, d_model: int, num_heads: int, dff: int, rate=0.1, **kwargs):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.num_blocks = num_blocks
        self.pos_encoding = pos_encoding

        self.enc_blocks = [EncoderBlock(d_model, num_heads, dff, rate) for _ in range(num_blocks)]
        self.dropout = Dropout(rate)
        self.scale = tf.math.sqrt(tf.cast(self.d_model, backend.floatx()))

    def call(self, x: Tensor, training, mask: Tensor) -> Tensor:
        seq_len = tf.shape(x)[1]

        # adding position encoding.
        x *= self.scale

        x += self.pos_encoding[:, :seq_len, :]  # (batch_size, input_seq_len, d_model)
        x = self.dropout(x, training=training)  # (batch_size, input_seq_len, d_model)

        for i in range(self.num_blocks):
            x = self.enc_blocks[i](x, training, mask)  # (batch_size, input_seq_len, d_model)

        return x  # (batch_size, input_seq_len, d_model)
