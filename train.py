from absl import app
from absl import flags

from tensorflow import keras

from ml_collections.config_flags import config_flags

from sagan.dataloader import DataLoader
from sagan.models.sagan import SelfAttentionGAN

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("experiment_configs")


def main(_):
    ## Load the data
    dataset = DataLoader.get_dataset()

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

    ## Setting up Callback
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        FLAGS.experiment_configs.checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    ## Train the model
    sagan.fit(
        dataset,
        batch_size=FLAGS.experiment_configs.batch_size,
        epochs=FLAGS.experiment_configs.epochs,
        callbacks=[checkpoint_callback],
    )


if __name__ == "__main__":
    app.run(main)
