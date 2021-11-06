import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from models.generator import build_generator
from models.discriminator import build_discriminator
from utils import save_plot


## Define the hyperparameters
BATCH_SIZE = 64
LATENT_DIM = 128
EPOCHS = 100
generator_save_path = "./saved_models/generator.h5"
discriminator_save_path = "./saved_models/discriminator.h5"

# Build the SAGAN module
class SAGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim, gp_weight=10.0, **kwargs):
        super(SAGAN, self).__init__(**kwargs)
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(SAGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    # Calculate the gradient_penalty.
    def gradient_penalty(self, batch_size, real_images, fake_images):
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]

        for _ in range(2):
            # Sample random points in the latent space
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            generated_images = self.generator(random_latent_vectors)
            generated_labels = tf.zeros((batch_size, 1))

            # Train the discriminator
            with tf.GradientTape() as ftape:
                predictions = self.discriminator(generated_images)
                d1_loss = self.loss_fn(generated_labels, predictions)
            grads = ftape.gradient(d1_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_weights)
            )

            labels = tf.ones((batch_size, 1))

            with tf.GradientTape() as rtape:
                predictions = self.discriminator(real_images)
                d_cost = self.loss_fn(labels, predictions)

                # Calculate the gradient penalty
                gp = self.gradient_penalty(
                    batch_size, real_images, random_latent_vectors
                )
                # Add the gradient penalty to the original discriminator loss
                d2_loss = d_cost + gp * self.gp_weight

            grads = rtape.gradient(d2_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_weights)
            )

        ## Train the generator
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        misleading_labels = tf.ones((batch_size, 1))

        with tf.GradientTape() as gtape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = gtape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return {"d1_loss": d1_loss, "d2_loss": d2_loss, "g_loss": g_loss}


generator = build_generator()
discriminator = build_discriminator()

sagan = SAGAN(discriminator, generator, LATENT_DIM)

## Compile the model

### Define the optimizers
generator_optimizer = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0, beta_2=0.9)
discriminator_optimizer = keras.optimizers.Adam(
    learning_rate=0.0004, beta_1=0, beta_2=0.9
)

### Define the loss function
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1)

sagan = sagan.compile(discriminator_optimizer, generator_optimizer, loss_fn)

## Train the model
for epoch in range(EPOCHS):
    sagan.fit(dataset, epochs=25)
    generator.save(generator_save_path)
    discriminator.save(discriminator_save_path)

    n_samples = 25
    noise = np.random.normal(size=(n_samples, BATCH_SIZE))
    examples = generator.predict(noise)
    save_plot(examples, epoch, int(np.sqrt(n_samples)))
