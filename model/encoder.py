import tensorflow as tf

from encoder_block import EncoderBlock

class Encoder(tf.keras.layers.Layer):
    def __init__(self, pos_encoding: tf.Tensor, num_blocks: int, d_model: int, num_heads: int, dff: int, rate=0.1, **kwargs):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.num_blocks = num_blocks
        self.pos_encoding = pos_encoding

        self.enc_blocks = [EncoderBlock(d_model, num_heads, dff, rate) for _ in range(num_blocks)]
        # the encoder 
        self.dropout = tf.keras.layers.Dropout(rate)
        self.scale = tf.math.sqrt(tf.cast(self.d_model, tf.keras.backend.floatx()))

    def call(self, x: tf.Tensor, training, mask: tf.Tensor) -> tf.Tensor:

        seq_len = tf.shape(x)[1]

        # adding position encoding.
        # assert not tf.math.is_nan(x[0][0][0])
        x *= self.scale
        # assert not tf.math.is_nan(x[0][0][0])
        
        x += self.pos_encoding[:, :seq_len, :]  # (batch_size, input_seq_len, d_model)
        # assert not tf.math.is_nan(x[0][0][0])
        x = self.dropout(x, training=training)  # (batch_size, input_seq_len, d_model)
        # assert not tf.math.is_nan(x[0][0][0])

        for i in range(self.num_blocks):
            x = self.enc_blocks[i](x, training, mask)  # (batch_size, input_seq_len, d_model)
            # assert not tf.math.is_nan(x[0][0][0])

        return x  # (batch_size, input_seq_len, d_model) 