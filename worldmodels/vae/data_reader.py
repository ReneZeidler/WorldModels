import os
from pathlib import Path

import numpy as np
import tensorflow as tf

# Original paper uses only 10,000 episodes, but there's no reason not to use all that are available.
MAX_EPISODES_PER_EPOCH = None


def ds_gen(dirname: Path, z_size: int, normalize: bool = False, mean: float = 0.0, var: float = 0.0):
    def gen():
        files = list(dirname.glob("*.npz"))[:MAX_EPISODES_PER_EPOCH]
        indices = np.arange(len(files))
        np.random.shuffle(indices)  # Read in random order each time

        for i in indices:
            with np.load(str(files[i])) as data:
                for img in data["image"]:
                    if normalize:
                        # Zero mean, unit variance
                        img_i = (img - mean) / np.sqrt(var)
                    else:
                        # Images between 0 and 1
                        img_i = img / 255.0
                    zeroed_outputs = np.zeros([2 * z_size])
                    yield img_i, (img_i, zeroed_outputs)  # (input, (reconstruction, z))

    return gen


def create_tf_dataset(train_dir: Path, z_size: int, normalize: bool = False, mean: float = 0.0, var: float = 0.0):
    val_dir = train_dir / "validation"

    output_types = (tf.float32, (tf.float32, tf.float32))  # (input, (reconstruction, z))
    output_shapes = ((64, 64, 3), ((64, 64, 3), (z_size * 2)))  # (input, (reconstruction, z))

    train_gen = ds_gen(train_dir, z_size, normalize, mean, var)
    val_gen = ds_gen(val_dir, z_size, normalize, mean, var)

    train_data = tf.data.Dataset.from_generator(train_gen, output_types=output_types, output_shapes=output_shapes)
    val_data = tf.data.Dataset.from_generator(val_gen, output_types=output_types, output_shapes=output_shapes)

    return train_data, val_data
