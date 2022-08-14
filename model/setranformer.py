"""Contains the class SeTransformer"""

from tensorflow import Tensor
from keras import Model
from keras.layers import Embedding

from decoder import Decoder
from encoder import Encoder
from emb_trans import EmbeddingTransposed
from pos_enc import create_positional_encoding
from masks import create_masks


class SeTransformer(Model):
    """The base architecture of my models in this project."""
    def __init__(self, num_blocks: int, d_model: int, num_heads: int, dff: int,
                 vocab_size: int, max_len: int, rate: float, pad_int: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad_int = pad_int
        self.vocab_size = vocab_size
        pos_encoding = create_positional_encoding(max_len, d_model)
        self.d_model = d_model

        self.encoder = Encoder(pos_encoding, num_blocks, d_model, num_heads, dff, rate)
        self.decoder = Decoder(pos_encoding, num_blocks, d_model, num_heads, dff, vocab_size, rate)

        self.embedding = Embedding(input_dim=vocab_size, output_dim=d_model)
        self.emb_trans = EmbeddingTransposed(self.embedding, "softmax")


    def count_params(self):
        """counts parameters
        Raises an error if called before building the model"""
        sub_layers = (self.encoder, self.decoder, self.embedding)
        # Ignoring the embedding transposed layer
        # because it is sharing parameters with the embedding layer
        param_count = sum(map(lambda x: x.count_params(), sub_layers))
        assert isinstance(param_count, int)
        return param_count


    def call(self, inputs: list) -> Tensor:
        """The forward pass of the model"""
        inp, tar, training = inputs
        # inp.shape should be (batch_size, max_seq_len)
        # tar.shape should be (batch_size, set_size)
        input_embeddings = self.embedding(inp)  # (batch_size, max_seq_len, d_model)
        padding_mask, look_ahead_mask = create_masks(inp, tar, self.pad_int)

        enc_output = self.encoder(input_embeddings, training, padding_mask)
        # (batch_size, max_seq_len, d_model)

        # dec_output.shape should be (batch_size, set_size, d_model)
        dec_output = self.decoder(tar, enc_output, training, look_ahead_mask, padding_mask)

        final_output = self.emb_trans(dec_output)
        return final_output
