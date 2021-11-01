import tensorflow as tf
from tensorflow.keras import initializers


class L2RandomNormal(initializers.TruncatedNormal):
    def __init__(self, epsilon=1e-2, mean=0.0, stddev=1.0, seed=None):
        super(L2RandomNormal, self).__init__(mean=mean, stddev=stddev, seed=seed)
        self.epsilon = epsilon

    def __call__(self, shape, dtype=tf.float32):
        initial_values = super(L2RandomNormal, self).__call__(shape, dtype=dtype)
        return initial_values / (tf.norm(initial_values) + self.epsilon)
