from typing import List

import tensorflow as tf
from decoder_block import DecoderBlock
from keras.layers import Dropout, Layer
from tensorflow import Tensor


class Decoder(Layer):
    scale: Tensor
    num_blocks: int
    pos_encoding: Tensor
    decoder_blocks: List[DecoderBlock]
    dropout: Dropout

    def __init__(self, pos_encoding, num_blocks: int, d_model: int,
                 num_heads: int, dff: int, rate: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d_model_tensor = tf.cast(d_model, tf.keras.backend.floatx())
        self.scale = tf.math.sqrt(d_model_tensor)
        self.num_blocks = num_blocks
        self.pos_encoding = pos_encoding

        self.dec_blocks = [
            DecoderBlock(d_model, num_heads, dff, rate)
            for _ in range(num_blocks)
        ]
        self.dropout = Dropout(rate)

    def call(self, x: Tensor, training: bool, look_ahead_mask: Tensor,
             padding_mask: Tensor) -> tf.Tensor:
        # x.shape [batch_size, seq len, d_model]
        seq_len = tf.shape(x)[1]

        x += self.pos_encoding[:, :seq_len, :]

        for decoder_block in self.dec_blocks:
            x = decoder_block(
                x=x,
                look_ahead_mask=look_ahead_mask,
                padding_mask=padding_mask,
                training=training,
            )
        # x.shape should be (batch_size, set_size, d_model)
        return x
