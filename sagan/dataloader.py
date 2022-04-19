import tensorflow as tf
import tensorflow_datasets as tfds


class DataLoader:
    def __init__(self, dataset_name: str, image_size: int = 64) -> None:
        self.dataset_name = dataset_name
        self.image_size = image_size

    def _load_data(self, data):
        image = tf.cast(data["image"], dtype=tf.float32)
        image = tf.image.resize(images=image, size=[self.image_size, self.image_size])
        image = (image / 127.0) - 1.0
        return image

    def _prepare_dataset(self, dataset, batch_size: int = 16, is_training: bool = True):
        if is_training:
            dataset = dataset.shuffle(10 * batch_size)

        dataset = dataset.map(self._load_data, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    def get_dataset(self, batch_size: int = 16):
        train_dataset, test_dataset = tfds.load(
            "tf_flowers", split=["train[:85%]", "train[85%:]"], shuffle_files=True
        )

        train_dataset = self._prepare_dataset(train_dataset, batch_size=batch_size)
        test_dataset = self._prepare_dataset(
            test_dataset, batch_size=batch_size, is_training=False
        )

        return train_dataset, test_dataset
