import tensorflow as tf
from tensorflow import keras


class SpectralNorm(keras.constraints.Constraint):
    def __init__(self, n_iter=5):
        self.n_iter = n_iter

    def call(self, input_weights):
        w = tf.reshape(input_weights, (-1, input_weights.shape[-1]))
        u = tf.random.normal((w.shape[0], 1))
        for _ in range(self.n_iter):
            v = tf.matmul(w, u, transpose_a=True)
            v /= tf.norm(v)

            u = tf.matmul(w, v)
            u /= tf.norm(u)

        spec_norm = tf.matmul(u, tf.matmul(w, v), transpose_a=True)
        return input_weights / spec_norm
