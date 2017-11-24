from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf

class PadZeros(Layer):
    def __init__(self, diff, **kwargs):
        super(PadZeros, self).__init__(**kwargs)
        self.diff = diff
        self.trainable = False

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(PadZeros, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return tf.pad(x, ((0, 0), (0, 0), (0, 0), (0, self.diff)), mode='CONSTANT')

    def compute_output_shape(self, input_shape):
        batch, b_width, b_height, b_channels = input_shape
        return batch, b_width, b_height, b_channels + self.diff

    def get_config(self):
        config = {'diff': self.diff}
        base_config = super(PadZeros, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

