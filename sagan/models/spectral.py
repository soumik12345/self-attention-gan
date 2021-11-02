import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers, backend, Model

from .initializers import L2RandomNormal


class SpectralNorm(layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        power_iterations=1,
        epsilon=1e-12,
        stddev=2e-2,
        *args,
        **kwargs,
    ):
        super().__init__(
            filters,
            kernel_size,
            kernel_initializer=initializers.TruncatedNormal,
            *args,
            **kwargs,
        )
        self.power_iterations = power_iterations
        self.epsilon = epsilon
        self._stddev = stddev

    def build(self, input_shape):
        super().build(input_shape)
        input_shape = tf.TensorSpec(input_shape)
        self.input_spec = layers.InputSpec(shape=[None] + input_shape[1:])
        if hasattr(self.layer, "kernel"):
            self.w = self.layer.kernel
        elif hasattr(self.layer, "embeddings"):
            self.w = self.layer.embeddings
        else:
            raise AttributeError(
                f"Attributes 'kernel' or 'embeddings' not find in {type(self.layer).__name__}"
            )
        self.w_shape = self.w.shape.as_list()
        self.u = self.add_weight(
            shape=(1, self.w_shape[-1]),
            # initializer=initializers.TruncatedNormal(stddev=self._stddev),
            initializer=L2RandomNormal(epsilon=self.epsilon, stddev=self._stddev),
            dtype=self.w.dtype,
            trainable=False,
        )

    @tf.function
    def normalize_weights(self):
        w = tf.reshape(self.w, [-1, self.w_shape[-1]])
        u = self.u
        for _ in range(self.power_iterations):
            v = tf.math.l2_normalize(tf.matmul(u, w, transpose_b=True))
            u = tf.math.l2_normalize(tf.matmul(v, w))
        sigma = tf.matmul(tf.matmul(v, w), u, transpose_b=True)
        self.w.assign(self.w / sigma)
        self.u.assign(u)

    def get_config(self):
        config = {"power_iterations": self.power_iterations}
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(self.layer.compute_output_shape(input_shape).as_list())

    def call(self, inputs, training=None):
        training = backend.learning_phase() if training is None else training
        if training:
            self.normalize_weights()
        return self.layer(inputs)
