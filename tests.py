import unittest

import tensorflow as tf

from sagan import DataLoader
from sagan.models.sagan import SelfAttentionGAN


class DataLoaderTester(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.data_loader = DataLoader(dataset_name="tf_flowers")

    def test_train_dataset(self):
        train_dataset = self.data_loader.get_dataset(split="train", batch_size=16)
        data = next(iter(train_dataset))
        assert data.shape == (16, 64, 64, 3)

    # def test_train_dataset(self):
    #     test_dataset = self.data_loader.get_dataset(split="test", batch_size=16)
    #     data = next(iter(test_dataset))
    #     assert data.shape == (16, 64, 64, 3)


# class ModelTester(unittest.TestCase):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.model = SelfAttentionGAN(128)

#     def test_compile(self):
#         self.model.compile(
#             discriminator_optimizer=tf.keras.optimizers.Adam(
#                 learning_rate=1e-4, beta_1=0.1, beta_2=0.9
#             ),
#             generator_optimizer=tf.keras.optimizers.Adam(
#                 learning_rate=1e-4, beta_1=0.1, beta_2=0.9
#             ),
#         )
