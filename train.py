import os
from datetime import datetime

import wandb
from absl import app, flags, logging
from ml_collections.config_flags import config_flags
from tensorflow import keras

from sagan.dataloader import DataLoader
from sagan.models.sagan import SelfAttentionGAN
from sagan.utils import init_wandb, initialize_device

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("experiment_configs")


def main(_):

    # Get the current timestamp
    timestamp = datetime.utcnow().strftime("%y%m%d-%H%M%S")

    # Iinitialising Weights and Biases, if wandb is used
    if FLAGS.experiment_configs.use_wandb:
        wandb.login()
        init_wandb(
            FLAGS.experiment_configs.project_name,
            f"{FLAGS.experiment_configs.project_name}_{timestamp}",
            FLAGS.experiment_configs.to_dict(),
        )

    # Detect strategy
    strategy = initialize_device()
    batch_size = FLAGS.experiment_configs.batch_size * strategy.num_replicas_in_sync
    FLAGS.experiment_configs["batch_size"] = batch_size
    logging.info(f"Preparing data loader with a batch size of {batch_size}.")

    ## Load the data
    pipeline_train, pipeline_test = DataLoader.get_dataset(
        batch_size=FLAGS.experiment_configs.batch_size
    )
    logging.info(f"Number of training examples: {pipeline_train.cardinality()}")
    logging.info(f"Number of validation examples: {pipeline_test.cardinality()}")

    # Setting up callback
    logging.info("Initializing callbacks.")

    # Model Checkpoint Callback
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=f"{FLAGS.experiment_configs.checkpoint_filepath}_{timestamp}",
        save_best_only=True,
        save_weights_only=True,
    )

    logger_path = os.path.join(
        "training_logs",
        f"{FLAGS.experiment_configs.project_name}_{timestamp}",
    )
    logger_callback = keras.callbacks.CSVLogger(filepath=logger_path)

    # Pack the callbacks as a list.
    train_callbacks = [checkpoint_callback, logger_callback]

    if FLAGS.experiment_configs.use_wandb:
        train_callbacks.append(wandb.keras.WandbCallback())

    ## Load the model
    sagan = SelfAttentionGAN(FLAGS.experiment_configs.latent_dim)

    ### Define the optimizers
    generator_optimizer = keras.optimizers.Adam(
        learning_rate=FLAGS.experiment_configs.generator_learning_rate,
        beta_1=FLAGS.experiment_configs.beta_one,
        beta_2=FLAGS.experiment_configs.beta_two,
    )
    discriminator_optimizer = keras.optimizers.Adam(
        learning_rate=FLAGS.experiment_configs.discriminator_learning_rate,
        beta_1=FLAGS.experiment_configs.beta_one,
        beta_2=FLAGS.experiment_configs.beta_two,
    )

    ## Compile the Self-Attention GAN model
    sagan = sagan.compile(discriminator_optimizer, generator_optimizer)

    ## Train the model
    sagan.fit(
        pipeline_train,
        batch_size=FLAGS.experiment_configs.batch_size,
        epochs=FLAGS.experiment_configs.epochs,
        callbacks=train_callbacks,
    )


if __name__ == "__main__":
    app.run(main)
