import tensorflow as tf
from keras.layers import Layer
from tensorflow import Tensor

from decoder_block import DecoderBlock


class Decoder(Layer):
    def __init__(self, pos_encoding, num_blocks: int, d_model: int, num_heads: int, dff: int,
                 vocab_size: int, rate: float, **kwargs):
        super().__init__(**kwargs)

        self.scale = tf.math.sqrt(tf.cast(d_model, tf.keras.backend.floatx()))
        self.num_blocks = num_blocks
        self.pos_encoding = pos_encoding

        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.dec_layers = [DecoderBlock(d_model, num_heads, dff, rate) for _ in range(num_blocks)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, tar: Tensor, enc_output: Tensor, training: bool,
             look_ahead_mask: Tensor, padding_mask: Tensor) -> tf.Tensor:

        seq_len = tf.shape(tar)[1]

        x = self.embedding(tar)  # (batch_size, set_size, d_model)
        x *= self.scale
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_blocks):
            x = self.dec_layers[i](x=x, enc_output=enc_output, look_ahead_mask=look_ahead_mask, 
                                   padding_mask=padding_mask, training=training)

        # x.shape should be (batch_size, set_size, d_model)
        return x