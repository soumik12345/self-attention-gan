import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from models.generator import build_generator
from models.discriminator import build_discriminator


## Define the hyperparameters
BATCH_SIZE = 64
LATENT_DIM = 128
EPOCHS = 100
generator_save_path = "./saved_models/generator.h5"
discriminator_save_path = "./saved_models/discriminator.h5"

## Save the images
def save_plot(examples, epoch, n):
    examples = (examples + 1) / 2.0
    for i in range(n * n):
        plt.subplot(n, n, i + 1)
        plt.axis("off")
        plt.imshow(examples[i])
    filename = f"/samples/demo-{epoch+1}.png"
    plt.savefig(filename)
    plt.close()


# Build the SAGAN module
class SAGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim, **kwargs):
        super(SAGAN, self).__init__(**kwargs)
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(SAGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

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
                d2_loss = self.loss_fn(labels, predictions)
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
