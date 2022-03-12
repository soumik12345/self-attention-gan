import numpy as np

from tensorflow import keras
from tensorflow.keras import layers

from .res_block import Resblock
from .spectral_norm import SpectralNorm
from .attention import SelfAttention


def build_generator(z_dim, n_class):

    DIM = 64

    z = layers.Input(shape=(z_dim))
    labels = layers.Input(shape=(1), dtype="int32")

    x = layers.Dense(4 * 4 * 4 * DIM, kernel_constraint=SpectralNorm())(z)
    x = layers.Reshape((4, 4, 4 * DIM))(x)

    x = layers.UpSampling2D((2, 2))(x)
    x = Resblock(4 * DIM, n_class)(x, labels)

    x = layers.UpSampling2D((2, 2))(x)
    x = Resblock(2 * DIM, n_class)(x, labels)

    x = SelfAttention()(x)

    x = layers.UpSampling2D((2, 2))(x)
    x = Resblock(DIM, n_class)(x, labels)

    output_image = keras.activations.tanh(
        layers.Conv2D(3, 3, padding="same", kernel_constraint=SpectralNorm())(x)
    )

    return keras.models.Model([z, labels], output_image, name="generator")
