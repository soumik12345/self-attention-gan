import ml_collections

import tensorflow as tf
from tensorflow import keras

from .generator import build_generator
from .discriminator import build_discriminator


def w_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)


class SelfAttentionGAN(keras.Model):
    def __init__(self, configs: ml_collections.ConfigDict, **kwargs):
        super(SelfAttentionGAN, self).__init__(**kwargs)
        self.configs = configs
        self.latent_dim = configs.latent_dim
        # self.gp_weight = configs.gp_weight
        self.generator = build_generator(configs.image_size, configs.latent_dim)
        self.discriminator = build_discriminator(configs.image_size)

    def compile(self, g_optimizer, d_optimizer, **kwargs):
        super(SelfAttentionGAN, self).compile(**kwargs)
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """Reference: https://keras.io/examples/generative/wgan_gp/#create-the-wgangp-model"""
        alpha = tf.random.normal([batch_size, 1, 1, 1], minavl=0.0, maxval=1.0)
        interpolated_image = real_images + alpha * (fake_images - real_images)
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_image)
            predictions = self.discriminator(interpolated_image, training=True)
        gradients = gp_tape.gradient(predictions, [interpolated_image])[0]
        gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean((gradient_norm - 1.0) ** 2)
        return gradient_penalty

    def train_discriminator_step(self, random_latent_vectors, real_images):
        with tf.GradientTape() as d_tape:
            (
                generated_images,
                generator_attention_1,
                generator_attention_2,
            ) = self.generator(random_latent_vectors, training=False)
            real_prediction, real_attention_1, real_attention_2 = self.discriminator(
                real_images, training=True
            )
            fake_prediction, fake_attention_1, fake_attention_2 = self.discriminator(
                generated_images, training=True
            )

            real_labels = tf.ones(shape=tf.shape(real_prediction), dtype=tf.float32)
            real_loss = w_loss(-real_labels, real_prediction)
            generated_loss = w_loss(real_labels, fake_prediction)

            gradient_penalty = self.gradient_penalty(real_images, generated_images)

            total_loss = real_loss + generated_loss + gradient_penalty

        gradients = d_tape.gradient(total_loss, self.d.trainable_variables)
        self.d_optimizer.apply_gradients(
            zip(gradients, self.discriminator.trainable_variables)
        )

        return total_loss, gradient_penalty

    def train_generator_step(self, random_latent_vectors):
        with tf.GradientTape() as g_tape:
            generated_images, g_attention_1, g_attention_2 = self.generator(
                random_latent_vectors, training=True
            )
            fake_prediction, d_attention_1, d_attention_2 = self.discriminator(
                generated_images, training=False
            )
            misleading_labels = tf.ones(
                shape=tf.shape(fake_prediction), dtype=tf.float32
            )
            generator_loss = w_loss(fake_prediction, -misleading_labels)

        gradients = g_tape.gradient(generator_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(gradients, self.generator.trainable_variables)
        )

        return generator_loss

    def train_step(self, real_images):
        random_latent_vectors = tf.random.truncated_normal(
            shape=(self.configs.batch_size, self.latent_dim)
        )
        discriminator_loss, gradient_penalty = self.train_discriminator_step(
            random_latent_vectors, real_images
        )
        generator_loss = self.train_generator_step(random_latent_vectors)
        return {
            "generator_loss": generator_loss,
            "discriminator_loss": discriminator_loss,
            "gradient_penalty": gradient_penalty,
        }

    def summary(
        self, line_length=None, positions=None, print_fn=None, expand_nested=False
    ):
        self.generator.summary(
            line_length=line_length,
            positions=positions,
            print_fn=print_fn,
            expand_nested=expand_nested,
        )
        self.discriminator.summary(
            line_length=line_length,
            positions=positions,
            print_fn=print_fn,
            expand_nested=expand_nested,
        )

    def save(
        self,
        filepath,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None,
        save_traces=True,
    ):
        self.generator.save(
            filepath + "_generator.h5",
            overwrite=overwrite,
            include_optimizer=include_optimizer,
            save_format=save_format,
            signatures=signatures,
            options=options,
            save_traces=save_traces,
        )
        self.discriminator.save(
            filepath + "_discriminator.h5",
            overwrite=overwrite,
            include_optimizer=include_optimizer,
            save_format=save_format,
            signatures=signatures,
            options=options,
            save_traces=save_traces,
        )

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        self.generator.save_weights(
            filepath + "_generator.h5",
            overwrite=overwrite,
            save_format=save_format,
            options=options,
        )
        self.discriminator.save_weights(
            filepath + "_discriminator.h5",
            overwrite=overwrite,
            save_format=save_format,
            options=options,
        )
