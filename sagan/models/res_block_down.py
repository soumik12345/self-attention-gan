import tensorflow as tf
from tensorflow.keras import layers

import tensorflow_addons as tfa


class ResblockDown(layers.Layer):
    def __init__(self, filters, downsample=True):
        super(ResblockDown, self).__init__()
        self.filters = filters
        self.downsample = downsample

    def build(self, input_shape):
        input_filter = input_shape[-1]
        self.conv_1 = tfa.layers.SpectralNormalization(
            layers.Conv2D(filters=self.filters, kernel_size=3, padding="same")
        )
        self.conv_2 = tfa.layers.SpectralNormalization(
            layers.Conv2D(filters=self.filters, kernel_size=3, padding="same")
        )
        self.learned_skip = False

        if self.filters != input_filter:
            self.learned_skip = True
            self.conv_3 = tfa.layers.SpectralNormalization(
                layers.Conv2D(filters=self.filters, kernel_size=1, padding="same")
            )

    def down(self, x):
        return tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")

    def call(self, input_tensor):
        x = self.conv_1(input_tensor)
        x = tf.nn.leaky_relu(x, 0.2)

        x = self.conv_2(x)
        x = tf.nn.leaky_relu(x, 0.2)

        if self.downsample:
            x = self.down(x)

        if self.learned_skip:
            skip = self.conv_3(input_tensor)
            skip = tf.nn.leaky_relu(skip, 0.2)
            if self.downsample:
                skip = self.down(skip)
        else:
            skip = input_tensor
        output = skip + x
        return output
