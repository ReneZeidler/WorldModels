import os
from pathlib import Path

import numpy as np


def ensure_validation_split(data_dir: Path, validation_fraction: float = 0.1):
    val_dir = data_dir / "validation"
    # Create directory for validation data (if it doesn't exist)
    val_dir.mkdir(exist_ok=True)

    # List training and validation files
    train_files = list(data_dir.glob("*.npz"))
    val_files   = list(val_dir .glob("*.npz"))
    n_train = len(train_files)
    n_val   = len(val_files  )

    indices = np.arange(n_train)
    np.random.shuffle(indices)

    # Move subset of data to validation dir
    if n_val / (n_train + n_val) < validation_fraction:
        num_to_move = int(np.ceil((n_train + n_val) * validation_fraction) - n_val)
        print(f"Moving {num_to_move} episodes into validation set")

        for i in indices[-num_to_move:]:
            train_files[i].rename(val_dir / train_files[i].name)

        n_train -= num_to_move
        n_val   += num_to_move

    print(f"Training set: {n_train} episodes, Validation set: {n_val} episodes")


def analyse_dataset(train_dir: Path, analyse_num_episodes: int = 20) -> (int, float, float, float):
    # List training files
    train_files = list(train_dir.glob("*.npz"))
    n_train = len(train_files)

    indices = np.arange(n_train)
    np.random.shuffle(indices)

    print("Analysing random subset of {:d}/{:d} episodes".format(analyse_num_episodes, n_train))
    all_obs = []
    for i in indices[:analyse_num_episodes]:
        with np.load(str(train_files[i])) as data:
            obs = data["image"]
            print("Episode {:4d}: {:4d} frames, mean: {:6.1f}, var: {:6.1f}".format(
                i, obs.shape[0], obs.mean(), obs.var()
            ))
            all_obs.append(obs)

    all_obs = np.concatenate(all_obs)
    mean, var = all_obs.mean(), all_obs.var()
    avg_frames = all_obs.shape[0] / analyse_num_episodes
    print("Total: {:4d} frames ({:6.1f} avg), mean: {:6.1f}, var: {:6.1f}".format(
        all_obs.shape[0], avg_frames, mean, var
    ))

    return n_train, avg_frames, mean, var
