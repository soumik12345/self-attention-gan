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

    def get_dataset(self, split: str = "train", batch_size: int = 16):
        dataset = tfds.load(name=self.dataset_name, split=split, shuffle_files=True)
        dataset = dataset.map(
            self._load_data, num_parallel_calls=tf.data.AUTOTUNE
        ).cache()
        dataset = dataset.shuffle(10 * batch_size).batch(
            batch_size, drop_remainder=True
        )
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset
