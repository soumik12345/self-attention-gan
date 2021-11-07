from absl import app
from absl import flags

from ml_collections.config_flags import config_flags

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from sagan.models.generator import build_generator
from sagan.models.discriminator import build_discriminator
from sagan.models.sagan import SelfAttentionGAN
from sagan.dataloader import DataLoader
from sagan.utils import save_plot

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("experiment_configs")


## Define the hyperparameters
generator_save_path = FLAGS.experiment_configs.save_generator_model_path
discriminator_save_path = FLAGS.experiment_configs.save_discriminator_model_path

generator = build_generator()
discriminator = build_discriminator()
dataset = DataLoader.get_dataset()

sagan = SelfAttentionGAN(discriminator, generator, FLAGS.experiment_configs.latent_dim) # Will come from the config file

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
for epoch in range(FLAGS.experiment_configs.epochs): 
    sagan.fit(dataset, epochs=25)
    generator.save(generator_save_path)
    discriminator.save(discriminator_save_path)

    n_samples = 25
    noise = np.random.normal(size=(n_samples, FLAGS.experiment_configs.batch_size))
    examples = generator.predict(noise)
    save_plot(examples, epoch, int(np.sqrt(n_samples)))
