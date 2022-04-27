import os
import wandb

import tensorflow as tf
from tensorflow import keras


class CheckpointArtifactCallback(keras.callbacks.Callback):
    def __init__(self, experiment_name: str, model_name: str, wandb_run):
        super().__init__()
        self.model_name = model_name
        self.wandb_run = wandb_run
        self.experiment_name = experiment_name

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.model_name)
        artifact = wandb.Artifact(name=self.experiment_name, type="model")
        artifact.add_dir(self.model_name)
        self.wandb_run.log_artifact(artifact)
