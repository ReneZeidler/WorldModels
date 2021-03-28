import os
from pathlib import Path

import configargparse

from rnn.rnn import FeatureMode

PARSER = configargparse.ArgParser(default_config_files=['configs/doom.config'])
PARSER.add('-c', '--config_path', required=False, is_config_file=True, help='config file path')
PARSER.add('-r', '--results_path', default=os.path.join("..", "results"))

PARSER.add('--exp_name', required=True, help='name of experiment')
PARSER.add('--env_name', required=True, help='name of environment')
PARSER.add('--max_frames', required=True, type=int, help='max number of frames in episode')
PARSER.add('--min_frames', required=True, type=int, help='min number of frames in episode')
PARSER.add('--max_trials', required=True, type=int, help='max number of trials')
PARSER.add('--full_episode', dest='full_episode', action='store_true', help='ignore dones')  # DEPRECATED
PARSER.add('--no_full_episode', dest='full_episode', action='store_false', help='ignore dones')  # DEPRECATED
PARSER.add('--render_mode', dest='render_mode', action='store_true')
PARSER.add('--no_render_mode', dest='render_mode', action='store_false')
PARSER.add('--a_width', required=True, type=int, help='width of action vector')
PARSER.add('--z_size', required=True, type=int, help='vae: repr. size; gqn: repr. size + 7')
PARSER.add('--exp_mode', required=True, help='defines controller architecture')  # DEPRECATED
PARSER.add('--state_space', required=True, type=int,
           help='1 to only include hidden state. 2 to include both h and c')  # DEPRECATED
PARSER.add('--feature_mode', required=True, type=FeatureMode,
           help='which values to include in the features for the controller, e.g. mode_zh')
PARSER.add('--controller_hidden_layer', dest='controller_hidden_layer', action='store_true',
           help='Adds an additional hidden layer to the controller '
                '(in the original paper only used in conjunction with --feature_mode=mode_z)')

PARSER.add('--extract_use_controller', dest='extract_use_controller', action='store_true',
           help='Use a controller with random weights instead of directly sampling action space')
PARSER.add('--extract_repeat_actions', dest='extract_repeat_actions', action='store_true',
           help='Repeat the same action a few times before choosing a new random action')

PARSER.add('--normalize_images', action='store_true',
           help='if enabled, transforms images to zero mean, unit variance before giving them to the vae')
PARSER.add('--vae_batch_size', required=True, type=int, help='batch size for vae train')
PARSER.add('--vae_learning_rate', required=True, type=float, help='vae learning rate')
PARSER.add('--vae_kl_tolerance', required=True, type=float, help='vae kl tolerance for clipping')
PARSER.add('--vae_num_epoch', required=True, type=int, help='vae num epoch for training')

PARSER.add('--use_gqn', dest='use_gqn', action='store_true', help='use gqn instead of vae as the encoder')
PARSER.add('--use_vae', dest='use_gqn', action='store_false', help='use vae as the encoder')
PARSER.add('--gqn_consecutive_frames', action='store_true', help='gqn number of context images')
PARSER.add('--gqn_context_size', default=3, type=int, help='gqn number of context images')
PARSER.add('--gqn_batch_size', default=36, type=int, help='gqn batch size')
PARSER.add('--gqn_x_dim', default=3, type=int, help='gqn number of image channels')
PARSER.add('--gqn_r_dim', default=256, type=int, help='gqn encoder features')
PARSER.add('--gqn_h_dim', default=128, type=int, help='gqn generator lstm output size')
PARSER.add('--gqn_z_dim', default=64, type=int, help='gqn latent vector size')
PARSER.add('--gqn_l', default=12, type=int, help='gqn generator model size')

PARSER.add('--rnn_predict_done', action='store_true', help='if the rnn should also predict the done bit')
PARSER.add('--rnn_predict_reward', action='store_true', help='if the rnn should also predict the reward')
PARSER.add('--rnn_num_steps', required=True, type=int, help='number of rnn training steps')
PARSER.add('--rnn_epoch_steps', required=True, type=int, help='number of rnn training steps per epoch')
PARSER.add('--rnn_max_seq_len', required=True, type=int, help='sequence length to train rnn on')
PARSER.add('--rnn_input_seq_width', required=True, type=int, help='size of rnn input')
PARSER.add('--rnn_out_size', required=True, type=int, help='size of mdn-rnn params + auxiliaries')
PARSER.add('--rnn_size', required=True, type=int, help='size of hidden and cell state')
PARSER.add('--rnn_batch_size', required=True, type=int, help='batch size rnn uses')
PARSER.add('--rnn_grad_clip', required=True, type=float, help='clip rnn gradients by value to this')
PARSER.add('--rnn_num_mixture', required=True, type=int, help='number of mixtures in MDNRNN')
PARSER.add('--rnn_learning_rate', required=True, type=float, help='initial learning rate used by the rnn')
PARSER.add('--rnn_decay_rate', required=True, type=float, help='decay rate for rnn learning rate (per epoch)')

PARSER.add('--controller_train_in_dream', action='store_true', help='Use dream environment for training the controller')
PARSER.add('--controller_optimizer', type=str, help='ses, pepg, openes, ga, cma.', default='cma')
PARSER.add('--controller_num_episode', type=int, default=16, help='num episodes per trial')
PARSER.add('--controller_num_test_episode', type=int, default=100,
           help='number of random episodes to evaluate agent on')
PARSER.add('--controller_eval_steps', type=int, default=25, help='evaluate every eval_steps step')
PARSER.add('--controller_num_worker', type=int, default=64)
PARSER.add('--controller_num_worker_trial', type=int, help='trials per worker', default=1)
PARSER.add('--controller_antithetic', type=int, default=1, help='set to 0 to disable antithetic sampling')
PARSER.add('--controller_cap_time', type=int, default=0, help='set to 0 to disable capping timesteps to 2x of average.')
PARSER.add('--controller_retrain', type=int, default=0,
           help='set to 0 to disable retraining every eval_steps if results suck.\n only works w/ ses, openes, pepg.')
PARSER.add('--controller_seed_start', type=int, default=0, help='initial seed')
PARSER.add('--controller_sigma_init', type=float, default=0.1, help='sigma_init')
PARSER.add('--controller_sigma_decay', type=float, default=0.999, help='sigma_decay')
PARSER.add('--controller_batch_mode', type=str, default='mean', help='optimize for either min or mean across episodes')


def get_path(args: configargparse.Namespace, subfolder: str, create: bool = False) -> Path:
    results_dir = Path(args.results_path)
    if not results_dir.exists():
        raise RuntimeError(f"Results path \"{results_dir}\" does not exist.")

    path = results_dir / args.exp_name / args.env_name / subfolder
    if not path.exists():
        if create:
            print(f"Creating directory \"{path}\"")
            path.mkdir(parents=True, exist_ok=True)
        else:
            print(f"WARNING: Directory \"{path}\" does not exist.")
    else:
        print(f"Using directory \"{path}\"")

    return path
