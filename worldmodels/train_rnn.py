"""
train mdn-rnn from pre-processed data.
also save 1000 initial mu and logvar, for generative experiments (not related to training).
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

from rnn.rnn import MDNRNN, sample_vae
from utils import PARSER, get_path


def load_data(path: Path):
    print(f"Loading data from {path}...")

    raw_data = np.load(str(path))

    # Load preprocessed data
    data_mu     = raw_data["mu"]     if "mu"     in raw_data else None
    data_logvar = raw_data["logvar"] if "logvar" in raw_data else None

    data_r      = raw_data["r"] if "r" in raw_data else None
    data_v      = raw_data["v"] if "v" in raw_data else None

    data_action = raw_data["action"]
    data_reward = raw_data["reward"]
    data_done   = raw_data["done"  ]

    # Turn (num_samples, sequence_len) into (num_samples, sequence_len, 1)
    if len(data_reward.shape) == 2:
        data_reward = np.expand_dims(data_reward, axis=2)
    if len(data_done  .shape) == 2:
        data_done   = np.expand_dims(data_done  , axis=2)

    print(f"Loaded {len(data_done)} samples from {path}")

    return data_mu, data_logvar, data_r, data_v, data_action, data_reward, data_done


def create_initial_z(basedir: Path, train_data: tuple, num_samples: int = 1000):
    data_mu, data_logvar, data_r, data_v, _action, _reward, _done = train_data
    use_gqn = data_r is not None

    # Save 1000 initial mu and logvars. Used for sampling when training in dreams
    print(f"Generating {num_samples} initial z samples...", end="")

    if use_gqn:
        initial_z = []
        for i in range(num_samples):
            # Multiply values by 10,000 and save as int
            r = np.copy(data_r[i, 0, :])
            v = np.copy(data_v[i, 0, :])
            z = (np.concatenate([r, v]) * 10000).astype(np.int).tolist()
            initial_z.append(z)

        path = basedir / "initial_z_gqn.json"
        data = initial_z
    else:
        initial_mu = []
        initial_logvar = []
        for i in range(num_samples):
            # Multiply values by 10,000 and save as int
            mu     = np.copy(data_mu    [i, 0, :] * 10000).astype(np.int).tolist()
            logvar = np.copy(data_logvar[i, 0, :] * 10000).astype(np.int).tolist()
            initial_mu.append(mu)
            initial_logvar.append(logvar)

        path = basedir / "initial_z_vae.json"
        data = [initial_mu, initial_logvar]

    print(" Done.")

    with open(str(path), 'wt') as outfile:
        json.dump(data, outfile, sort_keys=True, indent=0, separators=(',', ': '))

    print(f"Saved samples to {path}")


def data_generator(data_mu, data_logvar, data_r, data_v, data_action, data_reward, data_done,
                   batch_size: int, sequence_length: int,
                   predict_done: bool, predict_reward: bool,
                   shuffle_and_repeat: bool = True):
    n_data = len(data_done)

    print("Using data with shapes:")
    print("data_mu    ", data_mu    .shape if data_mu     is not None else None)
    print("data_logvar", data_logvar.shape if data_logvar is not None else None)
    print("data_r     ", data_r     .shape if data_r      is not None else None)
    print("data_v     ", data_v     .shape if data_v      is not None else None)
    print("data_action", data_action.shape)
    print("data_reward", data_reward.shape)
    print("data_done  ", data_done  .shape)

    use_gqn = data_r is not None

    def gen():
        while True:
            # Shuffle all indices before each loop (skipped for validation dataset)
            perm = np.random.permutation(n_data) if shuffle_and_repeat else np.arange(n_data)
            for i in range(0, n_data, batch_size):
                indices = perm[i:i + batch_size]
                if len(indices) < batch_size:
                    break

                # suboptimal b/c we are always only taking first set of steps
                action = tf.cast(data_action[indices, :sequence_length, :], tf.float16)
                reward = tf.cast(data_reward[indices, :sequence_length, :], tf.float16)
                done   = tf.cast(data_done  [indices, :sequence_length, :], tf.float16)

                if use_gqn:
                    r = tf.cast(data_r[indices, :sequence_length, :], tf.float16)
                    v = tf.cast(data_v[indices, :sequence_length, :], tf.float16)
                    z = tf.concat([r, v], axis=2)
                else:
                    mu     = tf.cast(data_mu    [indices, :sequence_length, :], tf.float16)
                    logvar = tf.cast(data_logvar[indices, :sequence_length, :], tf.float16)
                    z = sample_vae(mu, logvar)

                # Inputs for RNN: z and action
                inputs = tf.concat([z, action], axis=2)

                # Target z value is shifted one to the "left" on the timestep axis
                # (and the new last position is just padded with 0)
                dummy_zero = tf.zeros([z.shape[0], 1, z.shape[2]], dtype=tf.float16)
                z_target = tf.concat([z[:, 1:, :], dummy_zero], axis=1)

                # Mask is inverse of done flag: Only propagate loss while simulation is not done.
                # The mask is just stitched at the end as a new z dimension and separated again in the loss function.
                # This is hacky, but the intended way of propagating masks in Keras is difficult to use.
                mask = 1.0 - done
                z_target = tf.concat([z_target, mask], axis=2)  # use a signal to not pass grad

                # TODO: Named outputs with dict
                outputs = [z_target]

                if predict_done:
                    # Mask for reward is shifted one to the "right" on the time axis so that the first done=True is
                    # still unmasked and can be learned.
                    done_mask = tf.concat([
                        tf.ones([batch_size, 1, 1], dtype=tf.float16),
                        mask[:, :-1, :]
                    ], axis=1)
                    done_target = tf.concat([done, done_mask], axis=2)
                    outputs.append(done_target)

                if predict_reward:
                    reward_target = tf.concat([reward, mask], axis=2)
                    outputs.append(reward_target)

                outputs = tuple(outputs) if len(outputs) > 1 else outputs[0]

                yield inputs, outputs

            if not shuffle_and_repeat:
                break  # If not repeating (for validation dataset), don't loop

    return gen


def create_dataset(train_data: tuple, validation_data: tuple,
                   batch_size: int, sequence_length: int,
                   z_size: int, input_width: int,
                   predict_done: bool, predict_reward: bool):
    # noinspection PyTypeChecker
    train_gen = data_generator(*train_data, batch_size, sequence_length,
                               predict_done, predict_reward)
    # noinspection PyTypeChecker
    val_gen = data_generator(*validation_data, batch_size, sequence_length,
                             predict_done, predict_reward, shuffle_and_repeat=False)

    input_type = tf.float16
    input_shape = (batch_size, sequence_length, input_width)

    output_types = [tf.float16]
    output_shapes = [(batch_size, sequence_length, z_size + 1)]
    if predict_done:
        output_types.append(tf.float16)
        output_shapes.append((batch_size, sequence_length, 1 + 1))
    if predict_reward:
        output_types.append(tf.float16)
        output_shapes.append((batch_size, sequence_length, 1 + 1))
    output_types = tuple(output_types) if len(output_types) > 1 else output_types[0]
    output_shapes = tuple(output_shapes) if len(output_shapes) > 1 else output_shapes[0]

    gen_types = (input_type, output_types)
    gen_shapes = (input_shape, output_shapes)

    train_dataset = tf.data.Dataset.from_generator(train_gen, gen_types, gen_shapes)
    train_dataset = train_dataset.prefetch(8)

    validation_dataset = tf.data.Dataset.from_generator(val_gen, gen_types, gen_shapes)
    validation_dataset = validation_dataset.prefetch(8)

    return train_dataset, validation_dataset


def train_rnn(args, train_dataset, validation_dataset):
    model_save_path = get_path(args, "tf_rnn", create=True)

    rnn = MDNRNN(args=args)
    rnn.compile(optimizer=rnn.optimizer, loss=rnn.loss_fn, metrics=rnn.get_metrics())

    print("Start training")

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_dir = model_save_path / "tensorboard" / current_time

    rnn.fit(
        train_dataset, validation_data=validation_dataset,
        steps_per_epoch=args.rnn_epoch_steps, epochs=args.rnn_num_steps // args.rnn_epoch_steps,
        callbacks=[
            tf.keras.callbacks.TensorBoard(
                log_dir=str(tensorboard_dir), update_freq=20, histogram_freq=1, profile_batch=0
            ),
            tf.keras.callbacks.ModelCheckpoint(
                str(model_save_path / "ckpt-e{epoch:03d}"), verbose=1
            ),
        ]
    )

    rnn.save(str(model_save_path))
    print(f"Model saved to {model_save_path}")


def main():
    args = PARSER.parse_args()

    data_dir = get_path(args, "series")
    train_data_path = data_dir / "series.npz"
    val_data_path   = data_dir / "series_validation.npz"

    train_data      = load_data(train_data_path)
    validation_data = load_data(val_data_path  )

    initial_z_dir = get_path(args, "tf_initial_z", create=True)
    create_initial_z(initial_z_dir, train_data)

    train_dataset, validation_dataset = create_dataset(
        train_data, validation_data,
        args.rnn_batch_size, args.rnn_max_seq_len,
        args.z_size, args.rnn_input_seq_width,
        args.rnn_predict_done, args.rnn_predict_reward
    )
    train_rnn(args, train_dataset, validation_dataset)


if __name__ == '__main__':
    np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpu_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    # tf.config.experimental_run_functions_eagerly(True)  # used for debugging

    main()
