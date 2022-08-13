import tensorflow as tf
from decoder import Decoder
from encoder import Encoder
from emb_trans import EmbeddingTransposed
from pos_enc import create_positional_encoding
from masks import create_masks


class SeTransformer(tf.keras.Model):
    """The base architecture of my models in this project."""
    def __init__(self, num_blocks: int, d_model: int, num_heads: int, dff: int,
                 vocab_size: int, max_len: int, rate: float, pad_int: int, using_tpu: bool, **kwargs):
        super().__init__(**kwargs)  # calls tf.keras.Model's __init__ method
        self.pad_int = pad_int
        self.vocab_size = vocab_size
        pos_encoding = create_positional_encoding(max_len, d_model)
        self.d_model = d_model

        self.encoder = Encoder(pos_encoding, num_blocks, d_model, num_heads, dff, rate)
        self.decoder = Decoder(pos_encoding, num_blocks, d_model, num_heads, dff, vocab_size, rate)

        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=d_model)
        self.emb_trans = EmbeddingTransposed(self.embedding, "softmax")
        # self.dense = tf.keras.layers.Dense(vocab_size, activation="softmax")


    def count_params(self) -> int:
        """counts trainable parameters
        Raises an error if caleed before building the model"""
        param_count: int = self.encoder.count_params() + self.decoder.count_params() + self.embedding.count_params() + self.emb_trans.count_params()
        assert isinstance(param_count, int)
        return param_count


    def call(self, inputs: list, training: bool) -> tf.Tensor:
        inp, tar = inputs
        # inp.shape should be (batch_size, max_seq_len)
        # tar.shape should be (batch_size, set_size)
        x = self.embedding(inp)  # (batch_size, max_seq_len, d_model)
        # for d0 in range(x.shape[0]):
        #     for d1 in range(x.shape[1]):
        #         for d2 in range(x.shape[2]):
        #             assert not tf.math.is_nan(x[d0][d1][d2])
        # if len(x.shape) == 3:
        #     assert not tf.math.is_nan(x[0][0][0])
        # elif len(x.shape) == 2:
        #     assert not tf.math.is_nan(x[0][0])
        # else: raise ValueError('embedding output should by 3 dim tensor')
        padding_mask, look_ahead_mask = create_masks(inp, tar, self.pad_int)
        
        enc_output = self.encoder(x, training, padding_mask)  # (batch_size, max_seq_len, d_model)
        # assert not tf.math.is_nan(enc_output[0][0][0])

        # dec_output.shape should be (batch_size, set_size, d_model)
        dec_output = self.decoder(tar, enc_output, training, look_ahead_mask, padding_mask)
        # assert not tf.math.is_nan(dec_output[0][0][0])

        final_output = self.emb_trans(dec_output)
        # assert not tf.math.is_nan(final_output[0][0][0])
        return final_output