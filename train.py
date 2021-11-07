from absl import app
from absl import flags

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from sagan.models.generator import build_generator
from sagan.models.discriminator import build_discriminator
from sagan.models.sagan import SelfAttentionGAN
from sagan.dataloader import DataLoader
from sagan.utils import GANMonitor

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("experiment_configs")


## Define the hyperparameters
generator_save_path = configs.save_generator_model_path
discriminator_save_path = configs.save_discriminator_model_path

generator = build_generator()
discriminator = build_discriminator()
dataset = DataLoader.get_dataset()

sagan = SelfAttentionGAN(discriminator, generator, configs.latent_dim) # Will come from the config file

## Compile the model

### Define the optimizers
generator_optimizer = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0, beta_2=0.9)
discriminator_optimizer = keras.optimizers.Adam(
    learning_rate=0.0004, beta_1=0, beta_2=0.9
)

## Compile the Self-Attention GAN model
sagan = sagan.compile(discriminator_optimizer, generator_optimizer)

## Setting up Callback
cbk = GANMonitor(num_img=3, latent_dim=FLAGS.experiment_configs.lentent_dim)

## Train the model
sagan.fit(dataset, batch_size=FLAGS.experiment_configs.batch_size, epochs=FLAGS.experiment_configs.epochs, callbacks=[cbk])
