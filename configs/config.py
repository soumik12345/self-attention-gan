import ml_collections


def get_config() -> ml_collections.ConfigDicts:
    config = ml_collections.ConfigDict()
    config.batch_size = 64  # Batch Size
    config.epochs = 100  # Number of Epochs
    config.image_size = 128  # Image Size

    config.generator_learning_rate = 0.0001  # Generator Learning Rate
    config.discriminator_learning_rate = 0.0004  # Discriminator Learning Rate
    config.beta_one = 0.1  # Beta One
    config.beta_two = 0.9  # Beta Two

    return config
