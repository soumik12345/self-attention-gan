import numpy as np

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_addons as tfa

from .attention import SelfAttention


def build_discriminator(image_size: int = 64, filters: int = 64, kernel_size: int = 4):
    input_tensor = keras.Input(shape=(image_size, image_size, 3))
    current_filters = filters
    x = input_tensor
    for i in range(3):
        current_filters = current_filters * 2
        x = tfa.layers.SpectralNormalization(
            layers.Conv2D(
                filters=current_filters,
                kernel_size=kernel_size,
                strides=2,
                padding="same",
            )
        )(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
    x, attention_1 = SelfAttention(current_filters)(x)
    for i in range(int(np.log2(image_size)) - 5):
        current_filters = current_filters * 2
        x = tfa.layers.SpectralNormalization(
            layers.Conv2D(
                filters=current_filters,
                kernel_size=kernel_size,
                strides=2,
                padding="same",
            )
        )(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
    x, attention_2 = SelfAttention(current_filters)(x)
    x = tfa.layers.SpectralNormalization(layers.Conv2D(filters=1, kernel_size=4))(x)
    output_tensor = layers.Flatten()(x)
    return keras.Model(
        input_tensor, [output_tensor, attention_1, attention_2], name="discriminator"
    )
