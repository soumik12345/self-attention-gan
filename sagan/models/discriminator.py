import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_addons as tfa

from .res_block_down import ResblockDown
from .attention import SelfAttention


def build_discriminator(n_class):
    DIM = 64
    IMAGE_SHAPE = (64, 64, 3)
    input_image = layers.Input(shape=IMAGE_SHAPE)
    input_labels = layers.Input(shape=(1))

    embedding = layers.Embedding(n_class, 4 * DIM)(input_labels)

    embedding = layers.Flatten()(embedding)

    x = ResblockDown(DIM)(input_image)  # 64

    x = ResblockDown(2 * DIM)(x)  # 32

    x = SelfAttention()(x)

    x = ResblockDown(4 * DIM)(x)  # 16

    x = ResblockDown(4 * DIM, False)(x)  # 4

    x = tf.reduce_sum(x, (1, 2))

    embedded_x = tf.reduce_sum(x * embedding, axis=1, keepdims=True)

    output = layers.Dense(1)(x)

    output += embedded_x

    return keras.models.Model([input_image, input_labels], output, name="discriminator")
