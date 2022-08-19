import tensorflow as tf
from keras.layers import Layer, Embedding


class EmbeddingTransposed(Layer):
    def __init__(self, tied_to: Embedding, activation: str, **kwargs):
        super(EmbeddingTransposed, self).__init__(**kwargs)
        self.tied_to = tied_to
        self.activation = tf.keras.activations.get(activation)

    def build(self):
        self.custom_weights = self.tied_to.weights[0]
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape[0], tf.keras.backend.int_shape(self.tied_to.weights[0])[0]

    def call(self, inputs):
        output = tf.keras.backend.dot(inputs, tf.keras.backend.transpose(self.custom_weights))
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = {"activation": tf.keras.activations.serialize(self.activation)}
        base_config = super(EmbeddingTransposed, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))