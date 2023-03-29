# run_pacman_gym.py
# ------------------
# script to run the eat-ghost pacman game (integrated with Open AI gym)

import os
from PIL import Image
import argparse

import gym
from eat_ghost_env import eatGhostPacmanGymEnv
from learning_agents.rl.ddpg.ddpg import DDPGAgent
from addict import Dict
from utils.experiment_record_utils import ExperimentLogger


TEMP_RESULT_SAVING_DIR = 'tmp/'
RESULT_SAVING_DIR = '../../experiment_results/'


def parse_args():
    # configurations
    parser = argparse.ArgumentParser(description="Pytorch RL rl_algorithms")
    parser.add_argument(
        "--seed", type=int, default=777, help="random seed for reproducibility"
    )
    parser.add_argument(
        "--cfg-path",
        type=str,
        default=None,
        help="config path",
    )
    parser.add_argument(
        "--test", dest="test", action="store_true", help="test mode (no training)"
    )
    parser.add_argument(
        "--load-from",
        type=str,
        default=None,
        help="load the saved model and optimizer at the beginning",
    )
    parser.add_argument(
        "--off-render", dest="render", action="store_false", help="turn off rendering"
    )
    parser.add_argument(
        "--render-after",
        type=int,
        default=0,
        help="start rendering after the input number of episode",
    )
    parser.add_argument(
        "--log", dest="log", action="store_true", help="turn on logging"
    )
    parser.add_argument(
        "--save-period", type=int, default=200, help="save model period"
    )
    parser.add_argument(
        "--episode-num", type=int, default=20000, help="total episode num"
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=-1, help="max episode step"
    )
    parser.add_argument(
        "--interim-test-num",
        type=int,
        default=0,
        help="number of test during training",
    )
    parser.add_argument(
        "--demo-path",
        type=str,
        default=None,
        help="demonstration path for learning from demo",
    )

    return parser.parse_args()


def run_game():
    expr_logger = ExperimentLogger(TEMP_RESULT_SAVING_DIR, 'ddpg-test')
    expr_logger.redirect_output_to_logfile_as_well()
    # config_file = 'experiment_configs/expr_pacman_eat_1_ghost.cnf'
    # expr_logger.copy_file(config_file)
    # env = eatGhostPacmanGymEnv.EatGhostPacmanGymEnv(config_file)
    env = gym.make('LunarLander-v2')

    # define the agent
    args = parse_args()
    log_cfg = Dict()
    hyper_params = Dict(dict(gamma=1.0,
                             tau=5e-3,
                             buffer_size=int(1e4),
                             batch_size=64,
                             initial_random_action=int(1000),
                             multiple_update=1,  # multiple learning updates
                             gradient_clip_ac=0.5,
                             gradient_clip_cr=1.0))
    network_cfg = Dict(dict(hidden_sizes_actor=[32, 32]
                            , hidden_sizes_critic=[32, 32]))
    optim_cfg = Dict(dict(lr_actor=3e-4, lr_critic=3e-4, weight_decay=1e-6))
    noise_cfg = Dict(dict(ou_noise_theta=0.0, ou_noise_sigma=0.0))

    ddpg_agent = DDPGAgent(env, args, log_cfg, hyper_params, network_cfg, optim_cfg, noise_cfg, logger=expr_logger)
    ddpg_agent.train()

    expr_logger.save_trajectories(is_save_utility=True)


if __name__ == '__main__':
    run_game()
