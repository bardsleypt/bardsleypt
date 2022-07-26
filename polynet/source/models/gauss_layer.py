import tensorflow as tf
import keras
from source.utils.logger import Logger

logger = Logger().get_logger()


class GaussFilter(keras.layers.Layer):
    def __init__(self, kernel_width=16, order=1, stride=1, eps=1e-3,
                 center_kernel=True, **kwargs):
        super().__init__(**kwargs)
        self.kernel_width = kernel_width
        self.order = order
        self.stride = stride
        self.eps = eps
        self.center_kernel = center_kernel
        self.sigma = self.add_weight(shape=(1,), dtype=tf.float32,
                                     initializer='ones',
                                     trainable=True, name='sigma')
        self.kernel = self.add_weight(shape=(self.kernel_width, 1, 1), dtype=tf.float32,
                                      initializer=self.gen_kernel, trainable=False, name='kernel')
        self.bias = self.add_weight(shape=(1,), initializer='zero', dtype=tf.float32,
                                    trainable=True, name='bias')

    @staticmethod
    def _assign_new_value(variable, value):
        with keras.backend.name_scope('UpdateValue') as scope:
            if tf.compat.v1.executing_eagerly_outside_functions():
                return variable.assign(value, name=scope)
            else:
                with tf.compat.v1.colocate_with(variable):  # pylint: disable=protected-access
                    return tf.compat.v1.assign(variable, value, name=scope)

    def gen_kernel(self, shape=(-1, 1, 1), dtype=tf.float32):
        gauss = [tf.math.exp(-(x - self.kernel_width / 2) ** 2 / (2 * self.sigma ** 2 + self.eps))
                 for x in range(self.kernel_width)]

        # Additional processing for derivative orders
        if self.order == 0:
            pass
        elif self.order == 1:
            gauss = [(x - self.kernel_width / 2) * val for x, val in enumerate(gauss)]
        elif self.order == 2:
            gauss = [(self.sigma ** 2 - (x - self.kernel_width / 2) ** 2) * val
                     for x, val in enumerate(gauss)]
        else:
            msg = 'Order {} is not defined'.format(self.order)
            logger.error(msg)
            raise Exception(msg)

        # Concatenate two halves of Gaussian kernel, normalize, and reshape
        gauss /= tf.reduce_max(tf.abs(gauss))
        gauss = tf.cast(gauss, dtype)
        if self.center_kernel:
            gauss -= tf.reduce_mean(gauss)
        return tf.reshape(gauss, shape)

    def call(self, inputs, training=False):
        if training:
            kernel = self.gen_kernel()
            self._assign_new_value(self.kernel, kernel)
        else:
            kernel = self.kernel
        outs = tf.nn.convolution(inputs, kernel, self.stride, padding='VALID') + self.bias
        return outs

    def get_config(self):
        return {'kernel': self.kernel, 'sigma': self.sigma, 'bias': self.bias}
