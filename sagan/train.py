import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from models.generator import build_generator
from models.discriminator import build_discriminator
from models.sagan import SelfAttentionGAN
from .dataloader import DataLoader
from utils import save_plot


## Define the hyperparameters
generator_save_path = "./saved_models/generator.h5"
discriminator_save_path = "./saved_models/discriminator.h5"

generator = build_generator()
discriminator = build_discriminator()
dataset = DataLoader.get_dataset()

sagan = SelfAttentionGAN(discriminator, generator, LATENT_DIM) # Will come from the config file

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
