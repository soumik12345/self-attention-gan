import numpy as np

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_addons as tfa

from .attention import SelfAttention


def build_generator(
    image_size: int = 64,
    latent_dim: int = 100,
    filters: int = 64,
    kernel_size: int = 4
):
    input_tensor = keras.Input(shape=(latent_dim,))
    x = layers.Reshape((1, 1, latent_dim))(input_tensor)
    repeat_num = int(np.log2(image_size)) - 1
    factor = 2 ** (repeat_num - 1)
    current_filters = filters * factor
    for i in range(3):
        current_filters = current_filters // 2
        strides = 4 if i == 0 else 2
        x = tfa.layers.SpectralNormalization(
            layers.Conv2DTranspose(
                filters=current_filters,
                kernel_size=kernel_size,
                strides=strides,
                padding="same"
            )
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    x, attention_1 = SelfAttention(current_filters)(x)
    for i in range(repeat_num - 4):
        current_filters = current_filters // 2
        x = tfa.layers.SpectralNormalization(
            layers.Conv2DTranspose(
                filters=current_filters,
                kernel_size=kernel_size,
                strides=2,
                padding="same"
            )
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    x, attention_2 = SelfAttention(current_filters)(x)
    x = tfa.layers.SpectralNormalization(
        layers.Conv2DTranspose(
            filters=3,
            kernel_size=kernel_size,
            strides=2,
            padding="same"
        )
    )(x)
    output_tensor = layers.Activation("tanh")(x)
    return keras.Model(input_tensor, [output_tensor, attention_1, attention_2])
