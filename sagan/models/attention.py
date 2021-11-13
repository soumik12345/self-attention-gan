import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers


class SelfAttention(layers.Layer):
    def __init__(
        self, input_dims, trainable=True, name=None, dtype=None, dynamic=False, **kwargs
    ):
        super().__init__(
            trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs
        )
        self.query_convolution = layers.Conv2D(filters=input_dims // 8, kernel_size=1)
        self.key_convolution = layers.Conv2D(filters=input_dims // 8, kernel_size=1)
        self.value_convolution = layers.Conv2D(filters=input_dims // 8, kernel_size=1)

    def build(self, input_shape):
        self.gamma = self.add_weight(
            self.name + "_gamma", shape=(), initializer=initializers.Zeros()
        )

    def call(self, inputs, *args, **kwargs):
        query = self.query_convolution(inputs)
        key = self.key_convolution(inputs)
        value = self.value_convolution(inputs)
        query_dim = tf.shape(query)
        batch_size, height, width = query_dim[0], query_dim[1], query_dim[2]
        proj_query = tf.reshape(query, (batch_size, height * width, -1))
        proj_key = tf.transpose(
            tf.reshape(key, (batch_size, height * width, -1)), (0, 2, 1)
        )
        proj_value = tf.transpose(
            tf.reshape(value, (batch_size, height * width, -1)), (0, 2, 1)
        )
        energy = tf.matmul(proj_query, proj_key)
        attention = tf.nn.softmax(energy)
        out = tf.matmul(proj_value, tf.transpose(attention, (0, 2, 1)))
        out = tf.reshape(tf.transpose(out, (0, 2, 1)), (batch_size, height, width, -1))
        return tf.add(tf.multiply(out, self.gamma), inputs), attention
