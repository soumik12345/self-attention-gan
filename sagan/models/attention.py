import tensorflow as tf
from tensorflow.keras import layers

from sagan.models.spectral_norm import SpectralNorm


class SelfAttention(layers.Layer):
    def __init__(self):
        super(SelfAttention, self).__init__()

    def build(self, input_shape):
        n, h, w, c = input_shape
        self.n_feats = h * w
        self.conv_theta = layers.Conv2D(
            c // 8,
            1,
            padding="same",
            kernel_constraint=SpectralNorm(),
            name="Conv_Theta",
        )
        self.conv_phi = layers.Conv2D(
            c // 8, 1, padding="same", kernel_constraint=SpectralNorm(), name="Conv_Phi"
        )
        self.conv_g = layers.Conv2D(
            c // 2, 1, padding="same", kernel_constraint=SpectralNorm(), name="Conv_G"
        )
        self.conv_attn_g = layers.Conv2D(
            c, 1, padding="same", kernel_constraint=SpectralNorm(), name="Conv_AttnG"
        )
        self.sigma = self.add_weight(
            shape=[1], initializer="zeros", trainable=True, name="sigma"
        )

    def call(self, x):
        n, h, w, c = x.shape
        theta = self.conv_theta(x)
        theta = tf.reshape(theta, (-1, self.n_feats, theta.shape[-1]))

        phi = self.conv_phi(x)
        phi = tf.nn.max_pool2d(phi, ksize=2, strides=2, padding="VALID")
        phi = tf.reshape(phi, (-1, self.n_feats // 4, phi.shape[-1]))

        attn = tf.matmul(theta, phi, transpose_b=True)
        attn = tf.nn.softmax(attn)

        g = self.conv_g(x)
        g = tf.nn.max_pool2d(g, ksize=2, strides=2, padding="VALID")
        g = tf.reshape(g, (-1, self.n_feats // 4, g.shape[-1]))

        attn_g = tf.matmul(attn, g)
        attn_g = tf.reshape(attn_g, (-1, h, w, attn_g.shape[-1]))
        attn_g = self.conv_attn_g(attn_g)

        output = x + self.sigma * attn_g

        return output
