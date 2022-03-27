import tensorflow as tf
from tensorflow.keras import layers

from sagan.models.condition_batchnorm import ConditionBatchNorm
from sagan.models.spectral_norm import SpectralNorm


class Resblock(layers.Layer):
    def __init__(self, filters, n_class):
        super(Resblock, self).__init__(name=f"g_resblock_{filters}x{filters}")
        self.filters = filters
        self.n_class = n_class

    def build(self, input_shape):
        input_filter = input_shape[-1]
        self.conv_1 = layers.Conv2D(
            filters=self.filters,
            kernel_size=3,
            padding="same",
            name="conv2d_1",
            kernel_constraint=SpectralNorm(),
        )
        self.conv_2 = layers.Conv2D(
            filters=self.filters,
            kernel_size=3,
            padding="same",
            name="conv2d_2",
            kernel_constraint=SpectralNorm(),
        )
        self.cbn_1 = ConditionBatchNorm(self.n_class)
        self.cbn_2 = ConditionBatchNorm(self.n_class)
        self.learned_skip = False

        if self.filters != input_filter:
            self.learned_skip = True
            self.conv_3 = layers.Conv2D(
                filters=self.filters,
                kernel_size=1,
                padding="same",
                name="conv2d_3",
                kernel_constraint=SpectralNorm(),
            )
            self.cbn_3 = ConditionBatchNorm(self.n_class)

    def call(self, input_tensor, labels):
        x = self.conv_1(input_tensor)
        x = self.cbn_1(x, labels)
        x = tf.nn.leaky_relu(x, 0.2)

        x = self.conv_2(x)
        x = self.cbn_2(x, labels)
        x = tf.nn.leaky_relu(x, 0.2)

        if self.learned_skip:
            skip = self.conv_3(input_tensor)
            skip = self.cbn_3(skip, labels)
            skip = tf.nn.leaky_relu(skip, 0.2)
        else:
            skip = input_tensor

        output = skip + x
        return output
