import tensorflow as tf
import keras


class Pad1D(keras.layers.Layer):
    def __init__(self, padding=(1, 1), mode="reflect", **kwargs):
        self.padding = tuple(padding)
        self.mode = mode
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 3, "Input shape should be [batch, N, channel]"
        return input_shape[0], input_shape[1] + self.padding[0] + self.padding[1], input_shape[2]

    def call(self, inputs, *args, **kwargs):
        padding_left, padding_right = self.padding
        return tf.pad(inputs, [[0, 0], [padding_left, padding_right], [0, 0]], mode=self.mode)
