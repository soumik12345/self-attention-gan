from datetime import datetime

import wandb
from absl import app, flags, logging
from ml_collections.config_flags import config_flags
from tensorflow import keras

from sagan.callbacks import CheckpointArtifactCallback, GanMonitor
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

    # Pack the callbacks as a list.
    train_callbacks = []

    if FLAGS.experiment_configs.use_wandb:
        train_callbacks.append(wandb.keras.WandbCallback())
        train_callbacks.append(
            CheckpointArtifactCallback(
                experiment_name=FLAGS.experiment_configs.project_name,
                model_name=f"{FLAGS.experiment_configs.project_name}_{timestamp}",
                wandb_run=wandb,
            )
        )

    train_callbacks.append(
        gan_monitor_callback=GanMonitor(
            pipeline_test,
            FLAGS.experiment_configs.batch_size,
            epoch_interval=FLAGS.experiment_configs.epoch_interval,
            use_wandb=True if FLAGS.experiment_configs.use_wandb else False,
            plot_save_dir=FLAGS.experiment_configs.plot_save_dir,  # Change `FLAGS.experiment_configs.plot_save_dir` to control this.
        )
    )

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
