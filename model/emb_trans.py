from typing import List

import tensorflow as tf
from keras.layers import Layer, Embedding
from keras import activations
from keras.backend import int_shape
from tensorflow import Tensor


class EmbeddingTransposed(Layer):
    custom_weights: Tensor
    tied_to: Embedding
    activation: Layer

    def __init__(self,
                 tied_to: Embedding,
                 activation: str,
                 *args,
                 **kwargs):
        super(EmbeddingTransposed,
              self).__init__(
            *args, **kwargs)
        self.tied_to = tied_to
        self.activation = activations.get(activation)

    def build(self, *args, **kwargs):
        super(EmbeddingTransposed, self).build(*args, **kwargs)
        self.custom_weights = self.tied_to.weights[0]
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape[0], int_shape(self.tied_to.weights[0])[0]

    def call(self, inputs, **kwargs):
        # return self.activation()
        output = tf.matmul(
            inputs,
            self.custom_weights,
            transpose_b=True)
        if self.activation is not None:
            return self.activation(output)
        return output

    def get_config(self):
        config_list = [("activation",
                        activations.serialize(self.activation))]
        base_object = super(EmbeddingTransposed, self)
        base_config = base_object.get_config()
        config_list: List
        config_list = list(base_config.items()) + config_list
        return dict(config_list)
