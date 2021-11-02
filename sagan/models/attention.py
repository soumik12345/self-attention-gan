import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers


class AttentionLayer(layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(
            trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs
        )

    def build(self, input_shape):
        self.gamma = self.add_weight(
            self.name + "_gamma", shape=(), initializer=initializers.Zeros()
        )

    def call(self, inputs, *args, **kwargs):
        if len(inputs) != 4:
            raise ValueError("Attention layer should have 4 inputs")
        query_tensor = inputs[0]
        key_tensor = inputs[1]
        value_tensor = inputs[2]
        origin_input = inputs[3]
        input_shape = tf.shape(query_tensor)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        proj_query = tf.reshape(query_tensor, (batch_size, height * width, -1))
        proj_key = tf.transpose(
            tf.reshape(key_tensor, (batch_size, height * width, -1)), (0, 2, 1)
        )
        proj_value = tf.transpose(
            tf.reshape(value_tensor, (batch_size, height * width, -1)), (0, 2, 1)
        )
        energy = tf.matmul(proj_query, proj_key)
        attention = tf.nn.softmax(energy)
        out = tf.matmul(proj_value, tf.transpose(attention, (0, 2, 1)))
        out = tf.reshape(tf.transpose(out, (0, 2, 1)), (batch_size, height, width, -1))
        return tf.add(tf.multiply(out, self.gamma), origin_input), attention


class SelfAttention(keras.Model):
    def __init__(self, input_dims, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attention = AttentionLayer()
        self.query_convolution = layers.Conv2D(filters=input_dims // 8, kernel_size=1)
        self.key_convolution = layers.Conv2D(filters=input_dims // 8, kernel_size=1)
        self.value_convolution = layers.Conv2D(filters=input_dims, kernel_size=1)

    def call(self, inputs, training=False):
        query = self.query_convolution(inputs)
        key = self.key_convolution(inputs)
        value = self.value_convolution(inputs)
        return self.attn([query, key, value, inputs])
