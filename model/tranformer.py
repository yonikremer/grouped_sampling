"""Contains the class Transformer"""

from tensorflow import Tensor
from keras import Model
from keras.layers import Embedding

from decoder import Decoder
from emb_trans import EmbeddingTransposed
from pos_enc import create_positional_encoding
from masks import create_masks


class Transformer(Model):
    """
    The base architecture
     of my models in the project.
     This is an implementation of the
     decoder only causal language model.
     """
    
    pad_int: int
    embedding: Embedding
    decoder: Decoder
    emb_trans: EmbeddingTransposed

    def __init__(self,
                 num_blocks: int,
                 d_model: int,
                 num_heads: int,
                 dff: int,
                 vocab_size: int,
                 max_len: int,
                 rate: float,
                 pad_int: int,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.pad_int = pad_int
        pos_encoding = create_positional_encoding(
            max_len, d_model)

        self.decoder = Decoder(pos_encoding,
                               num_blocks,
                               d_model,
                               num_heads,
                               dff,
                               rate)

        self.embedding = Embedding(
            input_dim=vocab_size,
            output_dim=d_model)
        self.emb_trans = EmbeddingTransposed(
            self.embedding,
            "softmax")

    def count_params(self) -> int:
        """
        counts parameters
        Raises an error if:
        called before building the model
        """
        sub_layers = (self.encoder,
                      self.decoder,
                      self.embedding)
        # Ignoring the
        # embedding transposed layer
        # because it is sharing
        # parameters with the embedding layer
        return sum(layer.count_params() for layer in sub_layers)

    def call(self, x: Tensor, training: bool) \
            -> Tensor:
        """The forward pass of the model"""
        # x.shape should be (batch_size, max_seq_len)
        embeddings = self.embedding(x)
        # (batch_size, max_seq_len, d_model)
        padding_mask, look_ahead_mask = create_masks(
            x, self.pad_int)

        dec_output = self.decoder(
            embeddings,
            training,
            look_ahead_mask,
            padding_mask)

        final_output = self.emb_trans(dec_output)
        # dec_output.shape should be
        # (batch_size, set_size, vocab_size)
        return final_output
