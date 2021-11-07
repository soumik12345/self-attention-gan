import ml_collections


def get_config() -> ml_collections.ConfigDicts:
    config = ml_collections.ConfigDict()
    config.batch_size = 64  # Batch Size
    config.epochs = 100  # Number of Epochs
    config.image_size = 128  # Image Size
    config.latent_dim = 128  # Latent Dimension

    config.generator_learning_rate = 1e-4  # Generator Learning Rate
    config.discriminator_learning_rate = 4e-4  # Discriminator Learning Rate
    config.beta_one = 0.1  # Beta One
    config.beta_two = 0.9  # Beta Two

    config.save_generator_model_path = "./saved_models/generator.h5"  # Generator Model Path
    config.save_discriminator_model_path = "./saved_models/discriminator.h5"  # Discriminator Model Path

    return config
