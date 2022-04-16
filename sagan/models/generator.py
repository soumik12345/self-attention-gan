import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers

from .attention import SelfAttention
from .res_block import Resblock


def build_generator(z_dim, n_class):

    DIM = 64

    z = layers.Input(shape=(z_dim))
    labels = layers.Input(shape=(1), dtype="int32")

    x = tfa.layers.SpectralNormalization(layers.Dense(units=4 * 4 * 4 * DIM))(z)
    x = layers.Reshape((4, 4, 4 * DIM))(x)

    x = layers.UpSampling2D(size=(2, 2))(x)
    x = Resblock(4 * DIM, n_class)(x, labels)

    x = layers.UpSampling2D(size=(2, 2))(x)
    x = Resblock(2 * DIM, n_class)(x, labels)

    x = SelfAttention()(x)

    x = layers.UpSampling2D(size=(2, 2))(x)
    x = Resblock(DIM, n_class)(x, labels)

    output_image = tfa.layers.SpectralNormalization(
        keras.activations.tanh(
            layers.Conv2D(filters=3, kernel_size=(3, 3), padding="same")
        )
    )(x)

    return keras.models.Model([z, labels], output_image, name="generator")
