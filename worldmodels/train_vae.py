import io
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from data_reader import analyse_dataset, ensure_validation_split
from utils import PARSER, get_path
from vae.data_reader import create_tf_dataset
from vae.vae import CVAE


class LogImage(tf.keras.callbacks.Callback):
    def __init__(self, log_dir: str, predict_images: tf.data.Dataset, num_images: int = 5):
        super().__init__()
        self.num_images = num_images
        self.predict_images = predict_images.unbatch().shuffle(5000).take(num_images)
        self.writer = tf.summary.create_file_writer(os.path.join(log_dir, "images"))

    def on_epoch_end(self, epoch, logs=None):
        figure = plt.figure(figsize=(4.5, self.num_images * 2))  # type: plt.Figure

        for i, (x, _y) in enumerate(self.predict_images):
            x = tf.reshape(x, (1, 64, 64, 3))
            y = self.model(x)[0]  # reconstruction

            x = tf.reshape(x, (64, 64, 3))
            y = tf.reshape(y, (64, 64, 3))

            plt.subplot(self.num_images, 2, 2 * i + 1, title="Ground truth")
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(x)

            plt.subplot(self.num_images, 2, 2 * i + 2, title="Reconstruction")
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(y)

        # figure.show()

        with self.writer.as_default():
            # Save the plot to a PNG in memory.
            buf = io.BytesIO()
            figure.savefig(buf, format='png')
            buf.seek(0)
            # Convert PNG buffer to TF image
            image = tf.image.decode_png(buf.getvalue(), channels=4)
            # Add the batch dimension
            image = tf.expand_dims(image, 0)

            # For some reason this fails most of the time an no image shows up in TensorBoard, but figure.show()
            # shows all images correctly.
            tf.summary.image("Reconstruction Examples", image, step=epoch)
            self.writer.flush()

    def on_train_end(self, logs=None):
        self.writer.close()


def main():
    args = PARSER.parse_args()

    data_path = get_path(args, "record")
    model_save_path = get_path(args, "tf_vae", create=True)

    ensure_validation_split(data_path)
    _n_train, _avg_frames, mean, var = analyse_dataset(data_path)
    if args.normalize_images:
        train_data, val_data = create_tf_dataset(data_path, args.z_size, True, mean, var)
    else:
        train_data, val_data = create_tf_dataset(data_path, args.z_size)

    shuffle_size = 5 * 1000  # Roughly 20 full episodes for shuffle windows, more increases RAM usage
    train_data = train_data.shuffle(shuffle_size, reshuffle_each_iteration=True).batch(args.vae_batch_size).prefetch(2)
    val_data = val_data.batch(args.vae_batch_size).prefetch(2)

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_dir = model_save_path / "tensorboard" / current_time

    vae = CVAE(args=args)
    vae.compile(optimizer=vae.optimizer, loss=vae.get_loss())
    vae.fit(train_data, validation_data=val_data, epochs=args.vae_num_epoch, callbacks=[
        tf.keras.callbacks.TensorBoard(log_dir=str(tensorboard_dir), update_freq=50, histogram_freq=1),
        LogImage(str(tensorboard_dir), val_data),
        tf.keras.callbacks.ModelCheckpoint(str(model_save_path / "ckpt-e{epoch:02d}"), verbose=1),
    ])
    vae.save(str(model_save_path))


if __name__ == "__main__":
    np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpu_devices:
        tf.config.experimental.set_memory_growth(gpu, True)

    main()
