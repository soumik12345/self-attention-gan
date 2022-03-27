import tensorflow as tf
from tensorflow.keras import layers

from sagan.models.spectral_norm import SpectralNorm


class SelfAttention(layers.Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        batch_size, height, width, num_channels = input_shape
        self.n_feats = height * width
        self.query_conv = layers.Conv2D(
            filters=num_channels // 8,
            kernel_size=1,
            padding="same",
            kernel_constraint=SpectralNorm(),
            name="Conv_Theta",
        )
        self.key_conv = layers.Conv2D(
            filters=num_channels // 8,
            kernel_size=1,
            padding="same",
            kernel_constraint=SpectralNorm(),
            name="Conv_Phi",
        )
        self.conv_g = layers.Conv2D(
            filters=num_channels // 2,
            kernel_size=1,
            padding="same",
            kernel_constraint=SpectralNorm(),
            name="Conv_G",
        )
        self.value_conv = layers.Conv2D(
            filters=num_channels,
            kernel_size=1,
            padding="same",
            kernel_constraint=SpectralNorm(),
            name="Conv_AttnG",
        )
        self.sigma = self.add_weight(
            shape=[1], initializer="zeros", trainable=True, name="sigma"
        )

    def call(self, x):
        batch_size, height, width, num_channels = x.shape
        theta = self.query_conv(x)
        theta = tf.reshape(theta, (-1, self.n_feats, theta.shape[-1]))

        phi = self.key_conv(x)
        phi = tf.nn.max_pool2d(phi, ksize=2, strides=2, padding="VALID")
        phi = tf.reshape(phi, (-1, self.n_feats // 4, phi.shape[-1]))

        attn = tf.matmul(theta, phi, transpose_b=True)
        attn = tf.nn.softmax(attn)

        g = self.conv_g(x)
        g = tf.nn.max_pool2d(g, ksize=2, strides=2, padding="VALID")
        g = tf.reshape(g, (-1, self.n_feats // 4, g.shape[-1]))

        attn_g = tf.matmul(attn, g)
        attn_g = tf.reshape(attn_g, (-1, height, width, attn_g.shape[-1]))
        attn_g = self.value_conv(attn_g)

        output = x + self.sigma * attn_g

        return output
