import unittest
import tensorflow as tf

from sagan import DataLoader


class DataLoaderTester(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.data_loader = DataLoader(dataset_name="caltech_birds2011")

    def test_train_dataset(self):
        train_dataset = self.data_loader.get_dataset(split="train", batch_size=16)
        data = next(iter(train_dataset))
        assert data.shape == (16, 64, 64, 3)

    def test_train_dataset(self):
        test_dataset = self.data_loader.get_dataset(split="test", batch_size=16)
        data = next(iter(test_dataset))
        assert data.shape == (16, 64, 64, 3)
