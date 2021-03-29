"""
saves ~ 200 episodes generated from a random policy
"""
import os
import random

import gym
import numpy as np

from controller import make_controller
from env import make_env
from utils import PARSER, get_path


def main():
    print("Setting niceness to 19")
    if "nice" in os.__dict__:
        os.nice(19)

    args = PARSER.parse_args()

    def make_env_with_args():
        return make_env(args=args, keep_image=True, wrap_rnn=False)

    dir_name = get_path(args, "record", create=True)

    controller = None
    if args.extract_use_controller:
        controller = make_controller(args=args)
    env = make_env_with_args()

    has_camera_data = isinstance(env.observation_space, gym.spaces.Dict) and "camera" in env.observation_space.spaces

    format_str = "[{success:s}] {done:s} after {frames:4d} frames, reward {reward:6.1f} " \
                 "(Total: {total_frames:7d} frames, {successful_trials:3d}/{total_trials:3d} successful trials)"

    total_frames = 0
    successful_trials = 0
    for trial in range(args.max_trials):
        try:
            seed = random.randint(0, 2 ** 31 - 1)
            filename = dir_name / (str(seed) + ".npz")

            np.random.seed(seed)
            env.seed(seed)

            recording_image  = []
            recording_camera = []
            recording_action = []
            recording_reward = []
            recording_done   = []

            # random policy
            if args.extract_use_controller:
                controller.init_random_model_params(stddev=np.random.rand() * 0.01)
            repeat_action = np.random.randint(1, 11)
            action = [0] * args.a_width

            total_reward = 0
            obs = env.reset()

            frame = 0
            ended_early = False
            for frame in range(args.max_frames):
                # Save current observation
                recording_image.append(obs["image"])
                if has_camera_data:
                    recording_camera.append(obs["camera"])

                # Get next action (random)
                if not args.extract_repeat_actions or frame % repeat_action == 0:
                    if args.extract_use_controller:
                        action = controller.get_action(obs["features"])
                    else:
                        action = np.random.rand(args.a_width) * 2.0 - 1.0
                    if args.extract_repeat_actions:
                        repeat_action = np.random.randint(1, 11)

                # Save action
                recording_action.append(action)

                # Perform action
                obs, reward, done, _info = env.step(action)
                total_reward += reward

                # Save reward and done flag
                recording_reward.append(reward)
                recording_done.append(done)

                # Stop when done
                if done:
                    ended_early = True
                    break

            total_frames += (frame + 1)
            enough_frames = len(recording_image) >= args.min_frames

            # Save episode to disk (if it has required minimum length)
            if enough_frames:
                successful_trials += 1

                recording_image  = np.array(recording_image,  dtype=np.uint8  )
                recording_camera = np.array(recording_camera, dtype=np.float16)
                recording_action = np.array(recording_action, dtype=np.float16)
                recording_reward = np.array(recording_reward, dtype=np.float16)
                recording_done   = np.array(recording_done,   dtype=np.bool   )

                data = {
                    "image" : recording_image,
                    "action": recording_action,
                    "reward": recording_reward,
                    "done"  : recording_done
                }
                if has_camera_data:
                    data["camera"] = recording_camera

                np.savez_compressed(str(filename), **data)

            print(format_str.format(
                success="O" if enough_frames else " ",
                done="Done" if ended_early else "Stop",
                frames=frame + 1,
                reward=total_reward,
                total_frames=total_frames,
                successful_trials=successful_trials,
                total_trials=trial + 1
            ))

        except gym.error.Error as e:
            print("Gym raised an error: " + str(e))
            env.close()
            env = make_env_with_args()

    env.close()


if __name__ == '__main__':
    main()
