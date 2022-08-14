from tensorflow import Tensor
from keras.layers import Layer, Dense


class PointWiseFeedForwardNetwork(Layer):
    def __init__(self, d_model: int, dff: int, **kwargs): 
        super().__init__(**kwargs)
        self.layer1 = Dense(dff, activation='relu')  # (batch_size, seq_len, dff)
        self.layer2 = Dense(d_model)  # (batch_size, seq_len, d_model)
    
    def call(self, x: Tensor) -> Tensor:
        """Gets tensor of shape (batch_size, seq_len, d_model) and dtype tf.keras.beckend.floatx()
        Returns tensor of shape (batch_size, seq_len, d_model) and dtype tf.keras.beckend.floatx()"""
        x = self.layer1(x)
        x = self.layer2(x)
        return x
