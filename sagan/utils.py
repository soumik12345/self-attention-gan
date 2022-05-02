import os

import matplotlib.pyplot as plt
import tensorflow as tf
import wandb


def make_dir(dir_name: str):
    """
    To create a directory, if it does not exsists.

    Args:
        dir_name (str): The name of the directory

    Returns:
        None
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def init_wandb(project_name: str, experiment_name: str, config: Dict):
    """Initialize WandB.

    Args:
        project_name (str): project name on WandB
        experiment_name (str): experiment name on WandB
        config: Experiment configurations.

    Returns:
        None
    """
    if project_name is not None and experiment_name is not None:
        wandb.init(
            project=project_name,
            name=experiment_name,
            config=config,
            entity="mobilevit",
        )


def initialize_device():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
    except ValueError:
        if len(tf.config.list_physical_devices("GPU")) > 1:
            strategy = tf.distribute.MirroredStrategy()
        else:
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    return strategy


def plot_results(images, titles, figure_size=(12, 12)):
    fig = plt.figure(figsize=figure_size)
    for i in range(len(images)):
        fig.add_subplot(1, len(images), i + 1).set_title(titles[i])
        _ = plt.imshow(images[i])
        plt.axis("off")
    return fig
