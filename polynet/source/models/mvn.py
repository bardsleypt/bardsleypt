import tensorflow as tf
import keras


class MVN(keras.layers.Layer):
    def __init__(self, mean_norm=True, var_norm=True,
                 update=True, eps=1e-3, name='mvn', **kwargs):
        self.mean_norm = mean_norm
        self.var_norm = var_norm
        self.update = tf.constant(update)
        self.eps = eps
        self.mean_accum, self.std_accum, self.mean, self.std, self.num_samples = [None] * 5
        super(MVN, self).__init__(name=name, **kwargs)

    def get_config(self):
        return {'mean': self.mean, 'mean_norm': self.mean_norm,
                'std': self.std, 'var_norm': self.var_norm,
                'num_samples': self.num_samples}

    def build(self, input_shape):
        self.update = self.add_weight(shape=[],
                                      initializer=keras.initializers.Constant(
                                          self.update),
                                      dtype=tf.bool, name='update', trainable=False)
        self.num_samples = self.add_weight(shape=[], initializer='zero', dtype=tf.float32,
                                           name='num_samples', trainable=False)
        self.mean = self.add_weight(shape=input_shape[1:], initializer='zero',
                                    dtype=tf.float32, name='mean', trainable=False)
        self.mean_accum = self.add_weight(shape=input_shape[1:], initializer='zero',
                                          dtype=tf.float32, name='mean_acc',
                                          trainable=False)
        self.std = self.add_weight(shape=input_shape[1:], initializer='ones',
                                   dtype=tf.float32, name='std', trainable=False)
        self.std_accum = self.add_weight(shape=input_shape[1:], initializer='ones',
                                         dtype=tf.float32, name='std_acc',
                                         trainable=False)

    def update_stats(self, data):
        self.num_samples.assign_add(tf.cast(tf.shape(data)[0], tf.float32))
        if self.mean_norm or self.var_norm:
            self.mean_accum.assign_add(tf.reduce_sum(data, axis=0))
            mean = self.mean_accum / self.num_samples
        else:
            mean = self.mean_accum

        if self.var_norm:
            self.std_accum.assign_add(tf.reduce_sum(data * data, axis=0))
            std = self.std_accum / self.num_samples - mean * mean
            std = tf.clip_by_value(std, self.eps, float('inf'))
            std = tf.sqrt(std)
        else:
            std = self.std_accum

        return mean, std

    def call(self, input_tensor, training=None):
        if self.update:
            mean, std = self.update_stats(input_tensor)
            self._assign_new_value(self.mean, mean)
            self._assign_new_value(self.std, std)
        else:
            mean, std = self.mean, self.std
        out = input_tensor - mean if self.mean_norm else input_tensor
        out = out / std if self.var_norm else out
        return out

    @staticmethod
    def _assign_new_value(variable, value):
        with keras.backend.name_scope('UpdateValue') as scope:
            if tf.compat.v1.executing_eagerly_outside_functions():
                return variable.assign(value, name=scope)
            else:
                with tf.compat.v1.colocate_with(variable):
                    return tf.compat.v1.assign(variable, value, name=scope)
