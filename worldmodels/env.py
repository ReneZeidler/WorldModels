import gc
import math
from collections import OrderedDict
from typing import Union, Tuple, Optional

import gym
import gym.spaces
import gym.utils.seeding
import numpy as np
import tensorflow as tf
from PIL import Image
from gym.wrappers.pixel_observation import PixelObservationWrapper

# noinspection PyUnresolvedReferences
import gym_minigrid
# noinspection PyUnresolvedReferences
import vizdoomgym

from gqn.gqn import GenerativeQueryNetwork
from rnn.rnn import MDNRNN, rnn_next_state, rnn_init_state, FeatureMode, rnn_output, rnn_output_size, rnn_sim
from utils import get_path
from vae.vae import CVAE


class DreamEnv(gym.Env):
    def __init__(self, initial_z: np.ndarray, z_size: int, rnn: MDNRNN,
                 features_mode: FeatureMode = FeatureMode.MODE_ZH):
        # noinspection PyTypeChecker
        self.np_random = None  # type: np.random.RandomState
        self.seed()

        self.features_mode = features_mode

        self.initial_z = initial_z
        self.rnn = rnn

        # TODO: Support different action spaces
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=())

        num_features = rnn_output_size(self.rnn.num_units, z_size, features_mode)
        self.observation_space = gym.spaces.Dict(
            features=gym.spaces.Box(low=-50., high=50., shape=(num_features,))  # TODO: Shouldn't it be -inf and inf?
        )

        self.rnn_state = None
        self.z = None

    def _sample_init_z(self):
        # Sample random initial z distribution
        shape = self.initial_z.shape
        idx = self.np_random.randint(shape[0])
        if len(shape) == 3:
            init_mu, init_logvar = self.initial_z[idx]
            init_mu /= 10000.0
            init_logvar /= 10000.0
            # Sample z from distribution
            init_z = init_mu + np.exp(init_logvar / 2.0) * self.np_random.randn(*init_logvar.shape)
        else:
            init_z = self.initial_z[idx] / 10000.0
        # Add batch dimension to be consistent with the output of all other functions that return a z value
        init_z = np.expand_dims(init_z, axis=0)
        return init_z

    def reset(self):
        self.rnn_state = rnn_init_state(self.rnn)
        self.z = self._sample_init_z()
        obs = OrderedDict(features=rnn_output(self.rnn_state, self.z, self.features_mode))
        return obs

    def step(self, action):
        self.rnn_state, self.z, reward, done = rnn_sim(self.rnn, self.z, self.rnn_state, action)
        obs = OrderedDict(features=rnn_output(self.rnn_state, self.z, self.features_mode))
        info = {}  # Additional info, not used
        return obs, reward, done, info

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        tf.random.set_seed(seed)
        return [seed]

    def close(self):
        tf.keras.backend.clear_session()
        gc.collect()

    def render(self, mode='human'):
        # Not implemented
        # TODO: replace with decoding of z's?
        pass


class MDNRNNWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, encoder: Union[CVAE, GenerativeQueryNetwork], rnn: MDNRNN,
                 keep_image: bool = True, features_mode: FeatureMode = FeatureMode.MODE_ZH):
        super().__init__(env)

        assert isinstance(env.observation_space, gym.spaces.Dict) and "image" in env.observation_space.spaces, \
            "Observation space needs to be a dict with key \"image\""
        orig = env.observation_space["image"]
        assert isinstance(orig, gym.spaces.Box), "image must be of observation type Box"

        self.keep_image = keep_image  # Whether or not to keep image in the observation dict
        self.features_mode = features_mode  # Which values to include in the encoding observation

        self.encoder = encoder  # Encoder passed as parameter, should already be initialized with weights
        self.rnn = rnn  # RNN passed as parameter, should already be initialized with weights

        self.rnn_state = None  # [h, c]
        self.z = None

        num_features = rnn_output_size(self.rnn.num_units, self.encoder.z_size, features_mode)
        self.observation_space.spaces["features"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,))
        if not self.keep_image:
            del self.observation_space.spaces["image"]

    def encode_image_to_z(self, obs):
        # Encode pixel observation using autoencoder, returns the encoding z
        image = np.copy(obs["image"]).astype(np.float) / 255.0
        # Autoencoder expects batches, so transform into "batch" of 1
        image = image.reshape(1, 64, 64, 3)

        if isinstance(self.encoder, CVAE):
            z = self.encoder.encode(image=image)
        elif isinstance(self.encoder, GenerativeQueryNetwork):
            camera = np.copy(obs["camera"]).reshape(1, 7)
            z = self.encoder.encode(image=image, camera=camera)
            z = np.concatenate([z, camera], axis=1)
        else:
            raise RuntimeError(f"Unknown encoder of type {type(self.encoder)}")
        return z

    def modify_observation(self, obs: dict):
        features = rnn_output(self.rnn_state, self.z, self.features_mode)

        # Append encoded observation (encoding z and hidden state h)
        obs["features"] = features
        if not self.keep_image:
            del obs["image"]

        return obs

    def reset(self, **kwargs):
        # TODO: Is this zeroes? Can this just be None?
        self.rnn_state = rnn_init_state(self.rnn)  # [h, c]

        # reset() of wrapped environment might call step() and already set self.z in the process
        self.z = None
        obs = self.env.reset(**kwargs)
        # If step() was not called, calculate z from initial observation
        if self.z is None:
            self.z = self.encode_image_to_z(obs)

        obs = self.modify_observation(obs)

        return obs

    def step(self, action):
        # Advance RNN
        # NOTE: Uses z from previous frame, which makes sense as this is the frame where the action was input.
        #       The original code advanced the RNN with the z generated from the next frame (after this action was
        #       input), but only in the CarRacing example, not in the Doom example.
        if action is not None:  # no action is given on reset
            self.rnn_state = rnn_next_state(self.rnn, self.z, action, self.rnn_state)

        # Advance simulation
        obs, reward, done, info = self.env.step(action)

        # Encode pixel observation
        self.z = self.encode_image_to_z(obs)

        obs = self.modify_observation(obs)

        return obs, reward, done, info

    def close(self):
        self.env.close()
        tf.keras.backend.clear_session()
        gc.collect()

    def seed(self, seed=None):
        seeds = self.env.seed(seed)
        if seed or isinstance(seeds, list) and len(seeds) >= 1:
            tf.random.set_seed(seed or seeds[0])
        return seeds


class NoEarlyStopWrapper(gym.Wrapper):
    """
    Forces environment to never stop early by always returning done = False. Might lead to undefined behaviour in
    some environments.
    """

    def step(self, action):
        obs, reward, _done, info = self.env.step(action)
        return obs, reward, False, info


class PixelsToDictObservationWrapper(gym.ObservationWrapper):
    """
    Wraps environments which already return a pixel observation to return a dictionary with a "image" key to match
    the format of the PixelObservationWrapper.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict(image=env.observation_space)

    def observation(self, observation):
        return OrderedDict(image=observation)


class ClipPixelObservationWrapper(gym.ObservationWrapper):
    """
    Clips a pixel observation using a given (tuple of) slice(s).
    """

    def __init__(self, env: gym.Env, slice_arg: Union[slice, Tuple[slice, ...]]):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Dict) and "image" in env.observation_space.spaces, \
            "Observation space needs to be a dict with key \"image\" but is {}".format(env.observation_space)
        orig = env.observation_space["image"]
        assert isinstance(orig, gym.spaces.Box), "image must be of observation type Box but is {}".format(orig)

        if isinstance(slice_arg, slice):
            slice_arg = (slice_arg,)
        missing_dimensions = len(orig.shape) - len(slice_arg)
        assert missing_dimensions >= 0, "Given more slices than observation dimensions"
        if missing_dimensions > 0:
            slice_arg += (slice(None),) * missing_dimensions

        self.slices = slice_arg
        new_low  = orig.low [self.slices]
        new_high = orig.high[self.slices]
        env.observation_space.spaces["image"] = gym.spaces.Box(new_low, new_high)

    def observation(self, observation):
        observation["image"] = observation["image"][self.slices]
        return observation


class PadPixelObservationWrapper(gym.ObservationWrapper):
    """
    Pads image observation to a given size by adding a border that repeats the edge values as necessary
    (the original image will be centered).
    """

    def __init__(self, env: gym.Env, target_size: Union[int, Tuple[int, int]]):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Dict) and "image" in env.observation_space.spaces, \
            "Observation space needs to be a dict with key \"image\" but is {}".format(env.observation_space)
        orig = env.observation_space["image"]
        assert isinstance(orig, gym.spaces.Box), "image must be of observation type Box but is {}".format(orig)

        if isinstance(target_size, int):
            t_h = t_w = target_size
        else:
            t_h, t_w = target_size
        o_h, o_w = orig.shape[:2]

        # Calculate difference between target size and original size
        d_h = max(t_h - o_h, 0)
        d_w = max(t_w - o_w, 0)

        if d_h > 0 or d_w > 0:
            # Calculate necessary padding for all sides
            pad_t = d_h // 2
            pad_b = d_h - pad_t
            pad_l = d_w // 2
            pad_r = d_w - pad_b

            self.pad_width = ((pad_t, pad_b), (pad_l, pad_r)) + tuple((0, 0) for _ in orig.shape[2:])
            print("Padding image with {}".format(self.pad_width))

            new_low  = np.pad(orig.low,  self.pad_width, mode='edge')
            new_high = np.pad(orig.high, self.pad_width, mode='edge')
            env.observation_space.spaces["image"] = gym.spaces.Box(new_low, new_high)
        else:
            # Original size is already same as or greater than target size, do nothing
            print("PadPixelObservationWrapper was called with target size {}x{}, but image is already {}x{}"
                  .format(t_w, t_h, o_w, o_h))
            self.pad_width = None

    def observation(self, observation):
        if self.pad_width is not None:
            observation["image"] = np.pad(observation["image"], self.pad_width, mode='edge')
        return observation


class ResizePixelObservationWrapper(gym.ObservationWrapper):
    """
    Resizes an image observation to the given dimensions.
    """

    def __init__(self, env: gym.Env, size: Union[int, Tuple[int, int]], mode: str = "RGB"):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Dict) and "image" in env.observation_space.spaces, \
            "Observation space needs to be a dict with key \"image\" but is {}".format(env.observation_space)
        orig = env.observation_space["image"]
        assert isinstance(orig, gym.spaces.Box), "image must be of observation type Box but is {}".format(orig)

        if isinstance(size, int):
            size = (size, size)  # If given a single dimension, assume square

        self.size = size
        self.mode = mode

        if orig.shape[:2] == self.size:
            print("Observation already has desired shape {}".format(orig.shape))
            self.no_resize = True
        else:
            self.no_resize = False
            shape = self.size + orig.shape[2:]
            # We assume that low and high are the same for every pixel
            low = orig.low.item(0)
            high = orig.high.item(0)
            env.observation_space.spaces["image"] = gym.spaces.Box(low, high, shape=shape, dtype=orig.dtype)

    def observation(self, observation):
        if self.no_resize:
            return observation
        img = Image.fromarray(observation["image"], mode=self.mode)
        img = img.resize(self.size)
        observation["image"] = np.array(img)
        return observation


class CarRacingActionWrapper(gym.ActionWrapper):
    @staticmethod
    def clip(x, lo=0.0, hi=1.0):
        return np.minimum(np.maximum(x, lo), hi)

    def action(self, action: np.ndarray):
        """
        Transform 3 inputs between -1 and 1 each to the required format for CarRacing.
        action[0] (steering) is kept as-is.
        action[1] (gas) is scaled between 0 and 1.
        action[2] (brake) is clipped between 0 and 1.
        (Copied from original implementation)
        """
        assert action.shape == (3,)
        action[1] = (action[1] + 1.0) / 2.0
        action[2] = self.clip(action[2])
        return action

    def reverse_action(self, action):
        raise NotImplementedError


class CarRacingObservationWrapper(gym.ObservationWrapper):
    """
    Wraps a CarRacing environment, adding additional position data and transforms it into the expected format
    (a dictionary with "image" and "camera" entries).
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict(
            image=env.observation_space,
            camera=gym.spaces.Box(-np.Inf, np.Inf, (5,))  # x, y, z, pitch, yaw
        )

    def observation(self, observation):
        image_data = observation

        # Calculate in the same way as in the render method of the CarRacing env to match the rendered view
        car_hull = self.unwrapped.car.hull
        scroll_x = car_hull.position[0]
        scroll_y = car_hull.position[1]
        angle = -car_hull.angle
        vel = car_hull.linearVelocity
        if np.linalg.norm(vel) > 0.5:
            angle = math.atan2(vel[0], vel[1])
        camera_data = (scroll_x, scroll_y, 0, 0, angle)

        return OrderedDict(image=image_data, camera=camera_data)


class VizdoomTakeCoverActionWrapper(gym.ActionWrapper):
    def action(self, action: np.ndarray):
        """
        Transform a single input between -1 and 1 into discrete left or right inputs.
        The action space [-1, 1] is evenly partitioned into left, nothing, right.
        The output action is the expected discrete action for VizdoomTakeCover-v0.
        """
        assert action.shape == (1,)
        if action[0] < -.333333:
            new_action = 0  # MOVE_LEFT
        elif action[0] > .333333:
            new_action = 1  # MOVE_RIGHT
        else:
            new_action = 2  # NOOP
        return new_action

    def reverse_action(self, action):
        raise NotImplementedError


class VizdoomObservationWrapper(gym.ObservationWrapper):
    """
    Wraps a Vizdoom environment that includes image and position data to transform it into the expected format
    (a dictionary with "image" and "camera" entries).
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict(
            image=env.observation_space[0],
            camera=gym.spaces.Box(-np.Inf, np.Inf, (5,))  # x, y, z, pitch, yaw
        )

    def observation(self, observation):
        image_data, camera_data = observation
        # Vizdoom returns (x, y, z, angle), where angle is yaw (looking left and right)
        camera_data = (camera_data[0], camera_data[1], camera_data[2], 0, camera_data[3])
        return OrderedDict(image=image_data, camera=camera_data)


class MiniGridActionWrapper(gym.ActionWrapper):
    def action(self, action: np.ndarray):
        """
        Transform a single input between -1 and 1 into discrete actions for MiniGrid.
        The action space [-1, 1] is evenly partitioned into turn left, move forward, turn right.
        """
        assert action.shape == (1,)
        if action[0] < -.333333:
            new_action = 0  # turn left
        elif action[0] > .333333:
            new_action = 1  # turn right
        else:
            new_action = 2  # move forward
        return new_action

    def reverse_action(self, action):
        raise NotImplementedError


class MiniGridObservationWrapper(gym.ObservationWrapper):
    """
    Wraps a MiniGrid environment, adding additional position data and transforms it into the expected format
    (a dictionary with "image" and "camera" entries).
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # MiniGrid already has a Dict observation space, we just add an additional entry
        self.observation_space.spaces["camera"] = gym.spaces.Box(-np.Inf, np.Inf, (5,))  # x, y, z, pitch, yaw

    def observation(self, observation):
        # agent_pos is (x, y) in grid coordinates, agent_dir is an integer from 0 to 3, which we convert to an angle
        env = self.unwrapped
        camera_data = (env.agent_pos[0], env.agent_pos[1], 0, 0, env.agent_dir * math.pi / 2)

        observation["camera"] = camera_data
        return observation


def make_env(args, dream_env: bool = False, seed: Optional[int] = None,
             keep_image: bool = False, wrap_rnn: bool = True, load_model: bool = True):
    # Prepares an environment that matches the expected format:
    # - The environment returns a 64x64 image in observation["image"]
    #   and camera data (x, y, z, pitch, yaw) in observation["camera"]
    # - If wrapped in the RNN, observation["features"] returns the RNN output to be used for the controller
    # - A dream environment simulates the actual environment using the RNN. It never returns an image
    #   (because the actual environment doesn't get run) and only returns the features
    # - A wrapped environment always returns the features, and can return the original image when keep_image is True

    full_episode = args.full_episode

    # Initialize VAE and MDNRNN networks
    if dream_env or wrap_rnn:
        features_mode = FeatureMode.MODE_ZCH if args.state_space == 2 else FeatureMode.MODE_ZH

        if args.use_gqn:
            encoder = GenerativeQueryNetwork(args.gqn_x_dim, args.gqn_r_dim,
                                             args.gqn_h_dim, args.gqn_z_dim, args.gqn_l, name="gqn")
            encoder_path = get_path(args, "tf_gqn")
        else:
            encoder = CVAE(args)
            encoder_path = get_path(args, "tf_vae")
        rnn = MDNRNN(args)
        rnn_path = get_path(args, "tf_rnn")

        # TODO: Is this still needed? Do we ever NOT load the model?
        if load_model:
            encoder.load_weights(str(encoder_path))
            rnn.load_weights(str(rnn_path))

    if dream_env:
        assert keep_image is False, "Dream environment doesn't support image observations"

        import json
        initial_z_dir = get_path(args, "tf_initial_z")
        if args.use_gqn:
            initial_z_path = initial_z_dir / "initial_z_gqn.json"
            with open(str(initial_z_path), 'r') as f:
                initial_z = json.load(f)
        else:
            initial_z_path = initial_z_dir / "initial_z_vae.json"
            with open(str(initial_z_path), 'r') as f:
                [initial_mu, initial_logvar] = json.load(f)
            # This could probably be done more efficiently
            initial_z = np.array([list(elem) for elem in zip(initial_mu, initial_logvar)], dtype=np.float)

        # Create dream environment
        # noinspection PyUnboundLocalVariable
        env = DreamEnv(initial_z, args.z_size, rnn, features_mode)

    else:
        # Create real environment
        kwargs = {}
        if args.env_name.startswith("VizdoomTakeCover"):
            kwargs["position"] = True  # Include position data as observation for Vizdoom environment

        print("Making environment {}...".format(args.env_name))
        env = gym.make(args.env_name, **kwargs)
        print("Raw environment:", env)

        from gym.envs.box2d import CarRacing
        from vizdoomgym.envs import VizdoomTakeCover
        from gym_minigrid.minigrid import MiniGridEnv
        if isinstance(env.unwrapped, CarRacing):
            # Accept actions in the required format
            env = CarRacingActionWrapper(env)
            # Transform CarRacing observations into expected format and add camera data
            env = CarRacingObservationWrapper(env)
            # Cut off "status bar" at the bottom of CarRacing observation (copied from original paper)
            env = ClipPixelObservationWrapper(env, (slice(84),))
        elif isinstance(env.unwrapped, VizdoomTakeCover):
            # Accept actions in the required format
            env = VizdoomTakeCoverActionWrapper(env)
            # Transform Vizdoom observations into expected format
            env = VizdoomObservationWrapper(env)
            # Cut off "status bar" at the bottom of the screen (copied from original paper)
            env = ClipPixelObservationWrapper(env, (slice(400),))
        elif isinstance(env.unwrapped, MiniGridEnv):
            from gym_minigrid.wrappers import RGBImgPartialObsWrapper
            # Accept actions in the required format
            env = MiniGridActionWrapper(env)
            # Get RGB image observations from the agent's viewpoint
            # (7x7 grid of tiles, with tile size 9 this results in a 63x63 image)
            env = RGBImgPartialObsWrapper(env, tile_size=9)
            # Add camera data to the observation
            env = MiniGridObservationWrapper(env)
            # Pad image to 64x64 to match the requirements (in effect just adding one row at the right and bottom edge
            # with repeated values from the edge)
            env = PadPixelObservationWrapper(env, target_size=64)
        else:
            env = PixelObservationWrapper(env, pixel_keys=("image",))

        if env.observation_space["image"].shape[:2] != (64, 64):
            # Resize image to 64x64
            env = ResizePixelObservationWrapper(env, size=(64, 64))

        # Wrap in RNN to add features to observation
        if wrap_rnn:
            # noinspection PyUnboundLocalVariable
            env = MDNRNNWrapper(env, encoder, rnn, keep_image=keep_image, features_mode=features_mode)

    # TODO: Is this needed? It was only ever implemented for CarRacing and didn't work
    # Force done=False if full_episode is True
    if full_episode:
        env = NoEarlyStopWrapper(env)

    # Set seed if given
    if seed is not None:
        env.seed(seed)

    print("Wrapped environment:", env)
    return env
