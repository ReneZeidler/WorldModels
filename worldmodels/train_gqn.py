from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

from data_reader import ensure_validation_split
from gqn.data_reader import load_from_tfrecord
from gqn.gqn import GenerativeQueryNetwork
from utils import PARSER, get_path

# Training hyper-parameters

#                       S_max      mu_i   LR_sched    enc    h_dim  z_dim    L
# Minigrid simple 1:    400,000   0.005     exp     pyramid    64     32    12  (diverged)
# Vizdoom 1:            400,000   0.005     exp     pyramid    64     32    12  (only learned gray-brown-gray)

# Minigrid simple 2:  1,000,000   0.0015    exp     pyramid    64     32     8  (only learned gray)
# Vizdoom 2:          2,000,000   0.0005    lin     pyramid    64     32     8  (only learned brown)

# Minigrid simple 3:  1,000,000   0.0005    lin      pool     128     64    12  (only learned gray)
# Vizdoom 3:          1,000,000   0.0005    lin      pool     128     64    12  (only learned gray-brown-gray,
#                                                                                slight variation in outputs)


S_max = 1000000  # Original: 2,000,000
num_epochs = 200
S_epoch = S_max // num_epochs
checkpoint_every = S_max // 20

mu_i = 0.0005   # Initial LR. Original: 0.0005  == 5e-4
mu_f = 0.00005  # Final LR.   Original: 0.00005 == 5e-5
mu_n = int(0.8 * S_max)  # Original: 1,600,000

sigma_i = 2.0  # Initial sigma (for log-likelihood calculation). Original: 2.0
sigma_f = 0.7  # Final sigma. Original: 0.7
sigma_n = S_max // 10  # Original: 200,000

# Following values are loaded from config file:

# batch_size = 36   # Original: 36
# context_size = 3  # Original:  5

# consecutive_frames = True  # Whether to take consecutive or random frames from a training sequence

# Model hyper-parameters
# x_dim = 3    # Image channels (RGB)
# r_dim = 256  # Encoder features (Pyramid/Pool -> 1x1xr_dim, Tower -> 16x16xr_dim), Original: 256
# h_dim = 128  # LSTM output size, Original: Unknown
# z_dim = 64   # Latent vector size, Original: Unknown
# L = 12  # Generator model size (number of iterations), Original: 12


def cast_im(x):
    return tf.cast(tf.clip_by_value(x * 255, 0, 255), tf.uint8)


class LogImages(tf.keras.callbacks.Callback):
    def __init__(self, log_dir: Path, test_data: tf.data.Dataset, num_images: int = 5):
        super().__init__()
        self.num_images = num_images
        self.test_batch = [b for b in test_data.take(1)][0][0]
        self.writer = tf.summary.create_file_writer(str(log_dir / "images"))

    def on_epoch_end(self, epoch, logs=None):
        with self.writer.as_default():
            x_mu, x_q, r, kl = self.model(self.test_batch, training=True)
            tf.summary.image("inference output",  cast_im(x_mu[:self.num_images]), step=epoch)

            x_mu, x_q, r = self.model.sample(self.test_batch)
            tf.summary.image("generation output", cast_im(x_mu[:self.num_images]), step=epoch)
            tf.summary.image("generation target", cast_im(x_q [:self.num_images]), step=epoch)
            self.writer.flush()

    def on_train_end(self, logs=None):
        self.writer.close()


def main():
    args = PARSER.parse_args()

    data_path = get_path(args, "record")
    model_save_path = get_path(args, "tf_gqn", create=True)

    ensure_validation_split(data_path)
    train_data = load_from_tfrecord(data_path, args.gqn_context_size, args.gqn_batch_size, mode='train')
    test_data  = load_from_tfrecord(data_path, args.gqn_context_size, args.gqn_batch_size, mode='test' )

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_dir = model_save_path / "tensorboard" / current_time

    # lr  = tf.optimizers.schedules.ExponentialDecay(mu_i, mu_n, mu_f / mu_i,   name="lr_schedule"   )
    lr    = tf.optimizers.schedules.PolynomialDecay (mu_i,    mu_n,    mu_f,    name="lr_schedule"   )
    sigma = tf.optimizers.schedules.PolynomialDecay (sigma_i, sigma_n, sigma_f, name="sigma_schedule")
    optimizer = tf.optimizers.Adam(learning_rate=lr)

    model = GenerativeQueryNetwork(args.gqn_x_dim, args.gqn_r_dim,
                                   args.gqn_h_dim, args.gqn_z_dim, args.gqn_l, name="gqn")
    model.compile(optimizer, sigma, const_sigma=sigma_f)
    model.fit(
        train_data, validation_data=test_data, validation_steps=5,
        steps_per_epoch=S_epoch, epochs=num_epochs,
        callbacks=[
            tf.keras.callbacks.TensorBoard(
                log_dir=str(tensorboard_dir), update_freq=20, histogram_freq=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                str(model_save_path / "ckpt-e{epoch:02d}"), save_freq=checkpoint_every, verbose=1
            ),
            LogImages(tensorboard_dir, test_data),
        ]
    )
    # model.save(str(model_save_path))


if __name__ == "__main__":
    np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpu_devices:
        tf.config.experimental.set_memory_growth(gpu, True)

    main()
