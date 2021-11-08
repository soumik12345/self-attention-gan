from absl import app
from absl import flags

from ml_collections.config_flags import config_flags

import tensorflow as tf
from tensorflow import keras

def w_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)