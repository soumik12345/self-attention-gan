import unittest

import tensorflow as tf

from sagan import DataLoader


class DataLoaderTester(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.data_loader = DataLoader(dataset_name="tf_flowers")

    def test_train_dataset(self):
        train_dataset, _ = self.data_loader.get_dataset(batch_size=16)
        data = next(iter(train_dataset))
        print(data.shape)
        assert data.shape == (16, 64, 64, 3)

    def test_train_dataset(self):
        _, test_dataset = self.data_loader.get_dataset(batch_size=16)
        data = next(iter(test_dataset))
        print(data.shape)
        assert data.shape == (16, 64, 64, 3)
