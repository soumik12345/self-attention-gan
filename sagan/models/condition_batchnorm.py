import tensorflow as tf
from tensorflow import keras


class ConditionBatchNorm(keras.layers.Layer):
    def __init__(self, n_class=2, decay_rate=0.999, eps=1e-7):
        super(ConditionBatchNorm, self).__init__()
        self.n_class = n_class
        self.decay = decay_rate
        self.eps = 1e-5

    def build(self, input_shape):
        self.input_size = input_shape
        batch_size, height, width, num_channels = input_shape

        self.gamma = self.add_weight(
            shape=[self.n_class, num_channels],
            initializer="ones",
            trainable=True,
            name="gamma",
        )

        self.beta = self.add_weight(
            shape=[self.n_class, num_channels],
            initializer="zeros",
            trainable=True,
            name="beta",
        )

        self.moving_mean = self.add_weight(
            shape=[1, 1, 1, num_channels],
            initializer="zeros",
            trainable=False,
            name="moving_mean",
        )

        self.moving_var = self.add_weight(
            shape=[1, 1, 1, num_channels],
            initializer="ones",
            trainable=False,
            name="moving_var",
        )

    def call(self, x, labels, training=False):

        beta = tf.gather(self.beta, labels)
        beta = tf.expand_dims(beta, 1)
        gamma = tf.gather(self.gamma, labels)
        gamma = tf.expand_dims(gamma, 1)

        if training:
            mean, var = tf.nn.moments(x, axes=(0, 1, 2), keepdims=True)
            self.moving_mean.assign(
                self.decay * self.moving_mean + (1 - self.decay) * mean
            )
            self.moving_var.assign(
                self.decay * self.moving_var + (1 - self.decay) * var
            )
            output = tf.nn.batch_normalization(x, mean, var, beta, gamma, self.eps)

        else:
            output = tf.nn.batch_normalization(
                x, self.moving_mean, self.moving_var, beta, gamma, self.eps
            )

        return output
