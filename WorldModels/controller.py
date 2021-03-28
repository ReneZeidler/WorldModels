import json
import random

import gym
import numpy as np

from rnn.rnn import rnn_output_size


def make_controller(args):
    input_size = rnn_output_size(args.rnn_size, args.z_size, args.feature_mode)
    controller = Controller(input_size, args.a_width, args.controller_hidden_layer, args.max_frames)
    return controller


class Controller:
    """
    Simple one-layer model that maps features to actions. Can optionally use a hidden layer with 40 neurons.
    Each output action is a value between -1 and 1.
    """

    def __init__(self, input_size: int, output_size: int, use_hidden_layer: bool = False, max_frames: int = 1000):
        self.input_size = input_size
        self.output_size = output_size
        self.use_hidden_layer = use_hidden_layer

        # TODO: This really doesn't belong here, but it was previously accessed with controller.args.max_frames
        self.max_frames = max_frames

        if self.use_hidden_layer:  # one hidden layer
            self.hidden_size = 40
            self.weight_hidden = np.random.randn(self.input_size, self.hidden_size)
            self.bias_hidden = np.random.randn(self.hidden_size)
            self.weight_output = np.random.randn(self.hidden_size, self.output_size)
            self.bias_output = np.random.randn(self.output_size)
            self.param_count = ((self.input_size + 1) * self.hidden_size) + \
                               (self.hidden_size * self.output_size + self.output_size)
        else:
            self.weight = np.random.randn(self.input_size, self.output_size)
            self.bias = np.random.randn(self.output_size)
            self.param_count = self.input_size * self.output_size + self.output_size

    def get_action(self, features: np.ndarray):
        """
        Returns action for given input features.
        Output is an array of output_size values between -1 and 1.
        """
        if self.use_hidden_layer:  # one hidden layer
            hidden = np.tanh(np.dot(features, self.weight_hidden) + self.bias_hidden)
            action = np.tanh(np.dot(hidden, self.weight_output) + self.bias_output)
        else:
            action = np.tanh(np.dot(features, self.weight) + self.bias)

        return action

    def set_model_params(self, model_params):
        if self.use_hidden_layer:  # one hidden layer
            params = np.array(model_params)
            cut_off = (self.input_size + 1) * self.hidden_size
            params_1 = params[:cut_off]
            params_2 = params[cut_off:]
            self.bias_hidden = params_1[:self.hidden_size]
            self.weight_hidden = params_1[self.hidden_size:].reshape(self.input_size, self.hidden_size)
            self.bias_output = params_2[:self.output_size]
            self.weight_output = params_2[self.output_size:].reshape(self.hidden_size, self.output_size)
        else:
            self.bias = np.array(model_params[:self.output_size])
            self.weight = np.array(model_params[self.output_size:]).reshape(self.input_size, self.output_size)

    def load_model(self, filename):
        with open(filename) as f:
            data = json.load(f)
        print("loading file %s" % filename)
        model_params = np.array(data[0])  # assuming other stuff is in data
        self.set_model_params(model_params)

    def get_random_model_params(self, stddev=0.1):
        # return np.random.randn(self.param_count)*stddev
        return np.random.standard_cauchy(self.param_count) * stddev  # spice things up

    def init_random_model_params(self, stddev=0.1):
        params = self.get_random_model_params(stddev=stddev)
        self.set_model_params(params)


def simulate(controller: Controller, env: gym.Env,
             train_mode=False, render_mode=True,
             num_episode=5, seed=-1, max_len=-1):
    reward_list = []
    t_list = []

    max_episode_length = controller.max_frames

    if train_mode and 0 < max_len < max_episode_length:
        max_episode_length = max_len

    if seed >= 0:
        random.seed(seed)
        np.random.seed(seed)
        env.seed(seed)

    for episode in range(num_episode):
        # if train_mode:
        #     print("episode: {}/{}".format(episode, num_episode))
        obs = env.reset()

        total_reward = 0.0
        t = 0
        for t in range(max_episode_length):
            if render_mode:
                env.render("human")
            else:
                env.render("rgb_array")

            action = controller.get_action(obs["features"])
            obs, reward, done, info = env.step(action)

            total_reward += reward
            if done:
                break

        if render_mode:
            print("total reward", total_reward, "timesteps", t)
            env.close()

        reward_list.append(total_reward)
        t_list.append(t)
    return reward_list, t_list
