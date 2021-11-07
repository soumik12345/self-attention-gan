from absl import app
from absl import flags

from ml_collections.config_flags import config_flags

import tensorflow as tf
from tensorflow import keras

class GANMonitor(keras.callbacks.Callback):
    """Reference https://keras.io/examples/generative/wgan_gp/"""
    def __init__(self, num_img=6, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images = (generated_images * 127.5) + 127.5

        for i in range(self.num_img):
            img = generated_images[i].numpy()
            img = keras.preprocessing.image.array_to_img(img)
            img.save("generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch))

def w_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)