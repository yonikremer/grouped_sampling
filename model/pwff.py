from keras.layers import Dense, Layer
from tensorflow import Tensor


class PointWiseFeedForwardNetwork(Layer):
    layer1: Dense
    layer2: Dense

    def __init__(self, d_model: int, dff: int, **kwargs):
        super().__init__(**kwargs)
        self.layer1 = Dense(dff, activation="relu")
        self.layer2 = Dense(d_model)

    def call(self, x: Tensor, **kwargs) -> Tensor:
        """x's shape:
        (batch_size, seq_len, d_model)
        Returns tensor of shape
        (batch_size, seq_len, d_model)"""
        x = self.layer1(x)
        # (batch_size, seq_len, dff)
        x = self.layer2(x)
        # (batch_size, seq_len, d_model)
        return x
