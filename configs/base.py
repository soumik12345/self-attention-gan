import ml_collections


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.use_wandb = False  # For using weights and biases
    config.project_name = "Self-Attention-GAN"  # wandb project name

    config.batch_size = 64  # Batch Size
    config.epoch_interval = 5  # Epoch interval for monitoring
    config.epochs = 100  # Number of Epochs
    config.image_size = 128  # Image Size
    config.latent_dim = 128  # Latent Dimension

    config.generator_learning_rate = 1e-4  # Generator Learning Rate
    config.discriminator_learning_rate = 4e-4  # Discriminator Learning Rate
    config.beta_one = 0.1  # Beta One
    config.beta_two = 0.9  # Beta Two

    config.checkpoint_filepath = "./checkpoint/sagan"  # Checkpoint File-Path
    config.plot_save_dir = "checkpoints/plots"  # Plot Save-Directory

    return config
