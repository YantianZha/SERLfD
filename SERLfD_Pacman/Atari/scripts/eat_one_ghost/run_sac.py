# run_deep_sac.py
# ------------------
# script to run soft actor-critic algorithm

import argparse
import os

import gym
import torch

from eat_ghost_env import eatGhostPacmanGymEnv
from learning_agents.rl.sac.sac import SACAgent
from addict import Dict
from utils.experiment_record_utils import ExperimentLogger
from utils.gym_atari_env_wrapper import PytorchImage, GymRenderWrapper

TEMP_RESULT_SAVING_DIR = 'tmp/'
RESULT_SAVING_DIR = '../../experiment_results/'

experiment_log_dir = TEMP_RESULT_SAVING_DIR

"""
experiment_name = 'sac-lunar-lander'
wandb_project_name = 'lunar-lander-continuous'
"""
experiment_name = 'sac-lunar-lander'
wandb_project_name = 'lunar-lander-continuous'


def parse_args():
    # configurations
    parser = argparse.ArgumentParser(description="Pytorch RL rl_algorithms")
    parser.add_argument("--virtual-display", dest="virtual_display", action="store_true", default=False,
                        help="open virtual display")
    parser.add_argument("--seed", type=int, default=500,
                        help="random seed for reproducibility")
    parser.add_argument("--cfg-path", type=str, default=None,
                        help="config path")
    parser.add_argument("--test", dest="test", action="store_true",
                        help="test mode (no training)")
    parser.add_argument("--load-from", type=str, default=None,
                        help="load the saved model and optimizer at the beginning")
    parser.add_argument("--use_wandb", dest="use_wandb", action="store_true", default=False,
                        help="whether store the results in wandb")
    parser.add_argument('--render-freq', type=int, default=10,
                        help='render frequency (default: 10)')
    parser.add_argument("--render", dest="render", action="store_true", default=False,
                        help="turn on rendering")
    parser.add_argument("--render-after", type=int, default=0,
                        help="start rendering after the input number of episode")
    parser.add_argument("--log", dest="log", action="store_true",
                        help="turn on logging")
    parser.add_argument("--save-period", type=int, default=800,
                        help="save model period")
    parser.add_argument("--iteration-num", type=int, default=5000,
                        help="total iteration num")
    parser.add_argument("--avg_score_window", dest="avg_score_window", type=int, default=100,
                        help="avg score window size")
    parser.add_argument("--eval_score_window", dest="eval_score_window", type=int, default=20,
                        help="avg evaluation score window size")
    parser.add_argument("--eval_period", type=int, default=8,
                        help="evaluation period (unit: episode)")
    parser.add_argument("--deterministic", dest="deterministic", action="store_true", default=False,
                        help="deterministic action")
    parser.add_argument("--demo-path", type=str, default=None,
                        help="(not used in SAC) demonstration path for learning from demo")

    return parser.parse_args()


def run_game():
    args = parse_args()

    expr_logger = ExperimentLogger(experiment_log_dir, experiment_name, save_trajectories=False)
    expr_logger.redirect_output_to_logfile_as_well()
    expr_logger.copy_file(os.path.abspath(__file__))

    expr_logger.set_is_use_wandb(args.use_wandb)
    expr_logger.set_wandb_project_name(wandb_project_name)
    expr_logger.set_wandb()

    env = gym.make('LunarLanderContinuous-v2')

    # define the agent
    log_cfg = Dict()
    hyper_params = Dict(dict(gamma=0.99,
                             tau=5e-3,
                             buffer_size=int(1e4),  # open-ai baselines: int(1e4)
                             batch_size=128,
                             init_random_actions=int(1e3),
                             update_starts_from=int(1e3),
                             max_traj_length=1000,
                             multiple_update=1,  # multiple learning updates
                             train_freq=4,  # in open-ai baselines, train_freq = 4
                             target_update_freq=1,
                             reward_scale=1.0,
                             reward_clip=[-1.0, 1.0],   # set to None to disable reward clipping
                             w_entropy=1e-3,
                             w_mean_reg=0.0,
                             w_std_reg=0.0,
                             w_pre_activation_reg=0.0,
                             auto_entropy_tuning=True,
                             deterministic_action=args.deterministic,
                             # State Processing
                             frame_stack=1,
                             # N-Step Buffer
                             n_step=5,  # if n_step <= 1, use common replay buffer otherwise n_step replay buffer
                             w_n_step=1.0,  # used in n-step update
                             # Prioritized Replay Buffer
                             use_prioritized=True,
                             per_alpha=0.6,  # open-ai baselines default: 0.6, alpha -> 1, full prioritization
                             per_beta=0.4,  # beta can start small (for stability concern and anneals towards 1
                             per_eps=1e-6
                             ))
    network_cfg = Dict(dict(use_cnn=False,
                            # MLP setting
                            fc_hidden_sizes_actor=[64, 64],
                            fc_hidden_sizes_qf=[64, 64],
                            fc_hidden_activation=torch.relu,
                            # CNN setting
                            fc_input_size=3136,
                            nonlinearity=torch.relu,
                            channels=[32, 64, 64],
                            kernel_sizes=[8, 4, 3],
                            strides=[4, 2, 1],
                            paddings=[0, 0, 0],
                            ))
    optim_cfg = Dict(dict(lr_actor=3e-4,
                          lr_qf=3e-4,
                          lr_entropy=3e-4,
                          eps_entropy=1e-4,
                          eps_qf=1e-4,
                          eps_actor=1e-4,
                          weight_decay=0.0))

    sac_agent = SACAgent(env, args, log_cfg, hyper_params, network_cfg, optim_cfg, logger=expr_logger)
    if args.test:
        sac_agent.test()
    else:
        sac_agent.train()

    expr_logger.save_trajectories(is_save_utility=True)


if __name__ == '__main__':
    run_game()
