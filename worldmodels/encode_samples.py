import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from gqn.gqn import GenerativeQueryNetwork
from utils import PARSER, get_path
from vae.vae import CVAE

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

args = PARSER.parse_args()
data_dir = get_path(args, "record")
val_dir = data_dir / "validation"
series_dir = get_path(args, "series", create=True)
use_gqn = args.use_gqn  # type: bool

if use_gqn:
    model_ckpt = "ckpt-e200"
    model_dir = "tf_gqn"
else:
    model_ckpt = "ckpt-e10"
    model_dir = "tf_vae"
model_path = get_path(args, model_dir) / model_ckpt

# Original paper uses only 10,000 episodes, but there's no reason not to use all that are available.
MAX_EPISODES = 10000


def ds_gen(dirname: Path):
    def gen():
        filenames = list(dirname.glob("*.npz"))[:MAX_EPISODES]
        for filename in filenames:
            data = np.load(str(filename))

            img    = data['image' ]
            camera = data['camera'] if 'camera' in data else np.zeros((img.shape[0], 5), dtype=np.float16)
            action = np.reshape(data['action'], newshape=[-1, args.a_width])
            reward = data['reward']
            done   = data['done'  ]

            n_pad = args.max_frames - img.shape[0]  # pad so all episodes are the same length

            img    = tf.pad(img,    [[0, n_pad], [0, 0], [0, 0], [0, 0]])
            camera = tf.pad(camera, [[0, n_pad], [0, 0]                ])
            action = tf.pad(action, [[0, n_pad], [0, 0]                ])
            reward = tf.pad(reward, [[0, n_pad]                        ])
            done   = tf.pad(done,   [[0, n_pad]                        ], constant_values=done[-1])

            yield img, camera, action, reward, done

    return gen


def create_tf_dataset(dirname: Path):
    output_types = (tf.float32, tf.float32, tf.float32, tf.float32, tf.bool)
    output_shapes = (
        (args.max_frames, 64, 64, 3),
        (args.max_frames, 5),
        (args.max_frames, args.a_width),
        (args.max_frames,),
        (args.max_frames,)
    )
    dataset = tf.data.Dataset.from_generator(ds_gen(dirname), output_types=output_types, output_shapes=output_shapes)
    dataset = dataset.prefetch(32)
    return dataset


@tf.function
def encode_batch_vae(vae: CVAE, batch_img):
    batch_img = batch_img / 255.0
    return vae.encode_mu_logvar(batch_img)


@tf.function
def encode_batch_gqn(gqn: GenerativeQueryNetwork, batch_img, batch_camera):
    batch_img = batch_img / 255.0
    return gqn.encode(batch_img, batch_camera)


# def decode_batch(batch_z):
#     # decode the latent vector
#     batch_img = vae.decode(z.reshape(batch_size, z_size)) * 255.
#     batch_img = np.round(batch_img).astype(np.uint8)
#     batch_img = batch_img.reshape(batch_size, 64, 64, 3)
#     return batch_img


def save_dataset(dataset, filepath: Path):
    if use_gqn:
        print("Loading GQN model...")
        gqn = GenerativeQueryNetwork(args.gqn_x_dim, args.gqn_r_dim,
                                     args.gqn_h_dim, args.gqn_z_dim, args.gqn_l, name="gqn")
        gqn.load_weights(str(model_path))
    else:
        print("Loading VAE model...")
        vae = CVAE(args=args)
        vae.load_weights(str(model_path))
    print(f"Weights loaded from checkpoint {model_path}")

    mu_data     = []  # VAE mu (for z distribution)
    logvar_data = []  # VAE logvar (for z distribution)

    r_data      = []  # GQN representation
    v_data      = []  # GQN viewpoint

    action_data = []
    reward_data = []
    done_data   = []

    i = 0
    for i, batch in enumerate(dataset):
        image, camera, action, r, d = batch
        # shape = (sequence_len, *data_shape)

        if use_gqn:
            # Convert (x,y,z,pitch,yaw) into (x,y,z,sin(yaw),cos(yaw),sin(pitch),cos(pitch))
            pos   = camera[:, 0:3]
            pitch = camera[:, 3:4]
            yaw   = camera[:, 4:5]
            camera = tf.concat([pos, tf.sin(yaw), tf.cos(yaw), tf.sin(pitch), tf.cos(pitch)], axis=1)

            # noinspection PyUnboundLocalVariable
            r = encode_batch_gqn(gqn, image, camera)
            r_data.append(r     .numpy().astype(np.float32))
            v_data.append(camera.numpy().astype(np.float32))
        else:
            # noinspection PyUnboundLocalVariable
            mu, logvar = encode_batch_vae(vae, image)
            mu_data    .append(mu    .numpy().astype(np.float32))
            logvar_data.append(logvar.numpy().astype(np.float32))

        action_data.append(action.numpy()                   )
        reward_data.append(r     .numpy().astype(np.float32))
        done_data  .append(d     .numpy().astype(np.bool   ))

        print("\r{:5d}".format(i), end="")
    print(" Done!".format(i))

    data = {
        "action": np.array(action_data),
        "reward": np.array(reward_data),
        "done"  : np.array(done_data),
    }

    if use_gqn:
        data["r"] = np.array(r_data)
        data["v"] = np.array(v_data)
    else:
        data["mu"]     = np.array(mu_data)
        data["logvar"] = np.array(logvar_data)

    np.savez_compressed(str(filepath), **data)
    print(f"Encoded samples saved to {filepath}")


def main():
    print("Setting niceness to 19")
    os.nice(19)

    print("Encoding training dataset")
    dataset = create_tf_dataset(data_dir)
    filepath = series_dir / "series.npz"
    save_dataset(dataset, filepath)

    print("Encoding validation dataset")
    dataset = create_tf_dataset(val_dir)
    filepath = series_dir / "series_validation.npz"
    save_dataset(dataset, filepath)


if __name__ == '__main__':
    main()
