# run_trpo.py
# ------------------
# script to run the eat-ghost pacman game (integrated with Open AI gym)

import os
from PIL import Image
import argparse

from eat_ghost_env import eatGhostPacmanGymEnv
import gym
from utils.gym_atari_env_wrapper import PytorchImage
from learning_agents.rl.trpo.trpo import TRPOAgent
from addict import Dict
from utils.experiment_record_utils import ExperimentLogger


TEMP_RESULT_SAVING_DIR = 'tmp/'
RESULT_SAVING_DIR = '../../experiment_results/'

experiment_log_dir = TEMP_RESULT_SAVING_DIR
experiment_name = 'trpo-lunar-lander'
wandb_project_name = 'lunar-lander-continuous'  # 'pacman-eat-ghost'


def parse_args():
    # configurations
    parser = argparse.ArgumentParser(description="Pytorch RL rl_algorithms")
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
    parser.add_argument('--render-freq', type=int, default=5,
                        help='render frequency (default: 10)')
    parser.add_argument("--off-render", dest="render", action="store_false", default=True,
                        help="turn off rendering")
    parser.add_argument("--render-after", type=int, default=0,
                        help="start rendering after the input number of episode")
    parser.add_argument("--log", dest="log", action="store_true",
                        help="turn on logging")
    parser.add_argument("--save-period", type=int, default=200,
                        help="save model period")
    parser.add_argument("--iteration-num", type=int, default=20000,
                        help="total iteration num")
    parser.add_argument("--interim-test-num", type=int, default=0,
                        help="number of test during training")
    parser.add_argument("--demo-path", type=str, default=None,
                        help="demonstration path for learning from demo")

    return parser.parse_args()


def run_game():
    args = parse_args()

    expr_logger = ExperimentLogger(experiment_log_dir, experiment_name)
    expr_logger.redirect_output_to_logfile_as_well()
    expr_logger.set_is_use_wandb(args.use_wandb)
    config_file = '../../experiment_configs/expr_pacman_eat_1_ghost.cnf'
    expr_logger.copy_file(config_file)
    # env = PytorchImage(eatGhostPacmanGymEnv.EatGhostPacmanGymEnv(config_file), is_to_grey=True, is_normalize=True)
    env = gym.make('LunarLanderContinuous-v2')
    expr_logger.set_wandb_project_name(wandb_project_name)

    # define the controller_policy module
    policy_args = args
    policy_log_cfg = Dict()
    policy_hyper_params = Dict(dict(gamma=0.99,
                                    lamda=0.98,
                                    tau=0.95,   # gae (default: 0.95)
                                    damping=1e-2,   # damping (default: 1e-2)
                                    max_kl=1e-2,
                                    is_fusion=False,    # whether to use past experiences
                                    sample_size_iter=2048,  # total sample size (state-action pairs) to collect before TRPO update (default: 2048)
                                    buffer_size=int(1e3),
                                    traj_batch_size=10,     # number of trajectories used to update controller_policy each iteration
                                    traj_fixed_length=180,
                                    initial_random_action=int(1000),   # must be zero if used as controller_policy in irl
                                    num_iteration_update=2,  # multiple learning updates per iteration
                                    ))
    policy_network_cfg = Dict(dict(use_cnn=False,
                                   hidden_sizes_actor=[100, 100],
                                   hidden_sizes_vf=[100, 100]))
    policy_optim_cfg = Dict(dict(lr_actor=3e-4,
                                 lr_vf=3e-4,
                                 weight_decay=1e-5   # l2 regularization regression (default: 1e-3)
                                 ))

    trpo_agent = TRPOAgent(env, policy_args, policy_log_cfg, policy_hyper_params,
                           policy_network_cfg, policy_optim_cfg, encoder=None, logger=expr_logger)

    trpo_agent.train()

    expr_logger.save_trajectories(is_save_utility=True)


if __name__ == '__main__':
    run_game()
