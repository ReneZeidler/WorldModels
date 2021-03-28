from pathlib import Path
from typing import Callable

import numpy as np
import tensorflow as tf


def load_from_npz(data_path: Path, context_size: int, batch_size: int,
                  mode='train', consecutive_frames: bool = False) -> tf.data.Dataset:
    if mode not in ('train', 'test'):
        raise ValueError(f"Unsupported mode {mode} requested. Supported modes are train and test")

    if mode == 'test':
        data_path = data_path / "validation"

    # Number of views in the context + target view
    example_size = context_size + 1

    if consecutive_frames:
        dataset = read_consecutive_frames(data_path, example_size)
    else:
        dataset = read_random_frames(data_path, example_size)
    if mode == 'train':
        dataset = dataset.repeat().shuffle(1000, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    def _format_batch(frames, cameras):
        """Reads batch_size (query, target) pairs."""
        context_frames  = frames [:,   :-1]
        context_cameras = cameras[:,   :-1]
        target          = frames [:, -1   ]
        query_camera    = cameras[:, -1   ]

        inputs = {
            "context_frames" : context_frames,
            "context_cameras": context_cameras,
            "target"         : target,
            "query_camera"   : query_camera
        }
        return inputs, np.zeros(frames.shape[0])  # Dummy labels, since we don't use them

    dataset = dataset.map(_format_batch).prefetch(5)

    return dataset


def load_from_tfrecord(data_path: Path, context_size: int, batch_size: int,
                       mode='train', filename: str = "data.tfrecord") -> tf.data.Dataset:
    if mode not in ('train', 'test'):
        raise ValueError(f"Unsupported mode {mode} requested. Supported modes are train and test")

    if mode == 'test':
        data_path = data_path / "validation"

    data_file = str(data_path / filename)
    print(f"Loading data from {data_file}")
    dataset = tf.data.TFRecordDataset(data_file)

    # Create a description of the features.
    feature_description = {
        "context_frames" : tf.io.FixedLenFeature([], tf.string),
        "context_cameras": tf.io.FixedLenFeature([], tf.string),
        "target"         : tf.io.FixedLenFeature([], tf.string),
        "query_camera"   : tf.io.FixedLenFeature([], tf.string),
    }

    def _with_shape(serialized, shape, out_type=tf.float32) -> tf.Tensor:
        return tf.reshape(tf.io.parse_tensor(serialized, out_type), shape)

    def _parse_function(example_proto):
        # Parse the input `tf.train.Example` proto using the dictionary above.
        features = tf.io.parse_single_example(example_proto, feature_description)
        features["context_frames"]  = _with_shape(features["context_frames"] , (context_size, 64, 64, 3))
        features["context_cameras"] = _with_shape(features["context_cameras"], (context_size, 7        ))
        features["target"]          = _with_shape(features["target"]         , (              64, 64, 3))
        features["query_camera"]    = _with_shape(features["query_camera"]   , (              7,       ))
        return features, np.zeros(())  # Dummy label, since we don't use it

    dataset = dataset.map(_parse_function)
    if mode == 'train':
        dataset = dataset.repeat().shuffle(1000, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(5)

    return dataset


def to_tf_record(data_path: Path, context_size: int, mode='train', consecutive_frames: bool = False):
    if mode not in ('train', 'test'):
        raise ValueError(f"Unsupported mode {mode} requested. Supported modes are train and test")

    if mode == 'test':
        data_path = data_path / "validation"

    # Number of views in the context + target view
    example_size = context_size + 1

    if consecutive_frames:
        dataset = read_consecutive_frames(data_path, example_size)
    else:
        dataset = read_random_frames(data_path, example_size)

    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _serialize_example(frames, cameras):
        context_frames  = frames [  :-1]
        context_cameras = cameras[  :-1]
        target          = frames [-1   ]
        query_camera    = cameras[-1   ]

        features = {
            "context_frames" : _bytes_feature(tf.io.serialize_tensor(context_frames)),
            "context_cameras": _bytes_feature(tf.io.serialize_tensor(context_cameras)),
            "target"         : _bytes_feature(tf.io.serialize_tensor(target)),
            "query_camera"   : _bytes_feature(tf.io.serialize_tensor(query_camera)),
        }

        example_proto = tf.train.Example(features=tf.train.Features(feature=features))
        return example_proto.SerializeToString()

    def _tf_serialize_example(frames, cameras):
        tf_string = tf.py_function(
            _serialize_example,
            (frames, cameras),  # pass these args to the above function.
            tf.string)  # the return type is `tf.string`.
        return tf.reshape(tf_string, ())  # The result is a scalar

    dataset = dataset.map(_tf_serialize_example)

    filename = str(data_path / "data.tfrecord")
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(dataset)


def read_frames(basepath: Path, num_frames: int, index_choosing_function: Callable[[int], np.ndarray]):
    def gen():
        filepaths = list(basepath.glob("*.npz"))
        num_files = len(filepaths)
        indices = np.arange(num_files)
        np.random.shuffle(indices)  # Read in random order each time

        print(f"Found {num_files} records")
        c = 0

        for i in indices:
            c += 1
            if c % 100 == 0:
                print(f"Reading file {c:5d}/{num_files:5d}")
            with np.load(str(filepaths[i])) as data:
                total_frames = len(data["image"])
                indices = index_choosing_function(total_frames)

                frames = tf.gather(data["image"], indices)
                frames = tf.cast(frames, tf.float32) / 255

                raw_pose_params = tf.gather(data["camera"], indices)
                raw_pose_params = tf.cast(raw_pose_params, tf.float32)
                # Convert (x,y,z,pitch,yaw) into (x,y,z,sin(yaw),cos(yaw),sin(pitch),cos(pitch))
                pos   = raw_pose_params[:, 0:3]
                pitch = raw_pose_params[:, 3:4]
                yaw   = raw_pose_params[:, 4:5]
                cameras = tf.concat([pos, tf.sin(yaw), tf.cos(yaw), tf.sin(pitch), tf.cos(pitch)], axis=1)

                yield frames, cameras

    output_types = (tf.float32, tf.float32)  # (frames, cameras)
    output_shapes = ((num_frames, 64, 64, 3), (num_frames, 7))  # (frames, cameras)

    dataset = tf.data.Dataset.from_generator(gen, output_types=output_types, output_shapes=output_shapes)
    return dataset


def select_random_frames(num_frames: int = 3):
    def select(total_frames: int):
        return np.random.randint(total_frames, size=num_frames)

    return select


def select_consecutive_frames(num_frames: int = 3):
    def select(total_frames: int):
        start_index = np.random.randint(total_frames - num_frames)
        return np.arange(start_index, start_index + num_frames)

    return select


def read_random_frames(basepath: Path, num_frames: int = 3):
    return read_frames(basepath, num_frames, select_random_frames(num_frames))


def read_consecutive_frames(basepath: Path, num_frames: int = 3):
    return read_frames(basepath, num_frames, select_consecutive_frames(num_frames))


# path = Path(r'D:\Rene\Programming\Master\WorldModels\results\WorldModels\MiniGrid-Empty-Random-6x6-v0\record')
# to_tf_record(path, 3)

# dataset = load_from_tfrecord(str(path / "data.tfrecord"), 10)
# dataset = get_dataset(path, 3, 10)

# for parsed_record in dataset.take(1):
#    print(repr(parsed_record))
