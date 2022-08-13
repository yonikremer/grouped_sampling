import tensorflow as tf


class PointWiseFeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self, d_model: int, dff: int, **kwargs): 
        super().__init__(**kwargs)
        self.layer1 = tf.keras.layers.Dense(dff, activation='relu')  # (batch_size, seq_len, dff)
        self.layer2 = tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Gets tensor of shape (batch_size, seq_len, d_model) and dtype tf.keras.beckend.floatx()
        Returns tensor of shape (batch_size, seq_len, d_model) and dtype tf.keras.beckend.floatx()"""
        x = self.layer1(x)
        x = self.layer2(x)
        return x
