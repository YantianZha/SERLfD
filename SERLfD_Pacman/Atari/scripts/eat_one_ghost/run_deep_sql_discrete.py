# run_deep_sql_discrete.py
# ------------------
# script to run the eat-ghost pacman game (integrated with Open AI gym)

import os

import torch
from PIL import Image
import argparse

from eat_ghost_env import eatGhostPacmanGymEnv
import gym

from learning_agents.rl.dqn.dqn import DQNAgent
from learning_agents.rl.soft_q_learning.soft_dqn_discrete import SQL_Discrete_Agent
from utils.gym_atari_env_wrapper import PytorchImage
from learning_agents.rl.trpo.trpo import TRPOAgent
from addict import Dict
from utils.experiment_record_utils import ExperimentLogger


TEMP_RESULT_SAVING_DIR = 'tmp/'
RESULT_SAVING_DIR = '../../experiment_results/'

experiment_log_dir = TEMP_RESULT_SAVING_DIR

"""
experiment_name = 'dqn-freeway'
wandb_project_name = 'freeway'
"""
experiment_name = 'sql-pacman'
wandb_project_name = 'pacman-eat-1-ghost'
"""
experiment_name = 'dqn-breakout'
wandb_project_name = 'breakout'
"""

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
    parser.add_argument("--iteration-num", type=int, default=20000,
                        help="total iteration num")
    parser.add_argument("--avg_score_window", dest="avg_score_window", type=int, default=100,
                        help="avg score window size")
    parser.add_argument("--eval_score_window", dest="eval_score_window", type=int, default=20,
                        help="avg evaluation score window size")
    parser.add_argument("--eval_period", type=int, default=5,
                        help="evaluation period (unit: episode)")
    parser.add_argument("--deterministic", dest="deterministic", action="store_true", default=False,
                        help="deterministic action")
    parser.add_argument("--demo-path", type=str, default=None,
                        help="demonstration path for learning from demo")

    return parser.parse_args()


# noinspection DuplicatedCode
def run_game():
    args = parse_args()

    expr_logger = ExperimentLogger(experiment_log_dir, experiment_name, save_trajectories=False)
    expr_logger.redirect_output_to_logfile_as_well()

    expr_logger.set_is_use_wandb(args.use_wandb)
    expr_logger.set_wandb_project_name(wandb_project_name)
    expr_logger.set_wandb()

    config_file = '../../experiment_configs/expr_pacman_eat_1_ghost.cnf'
    expr_logger.copy_file(config_file)
    expr_logger.copy_file(os.path.abspath(__file__))
    if args.virtual_display:
        expr_logger.open_virtual_display()

    env = PytorchImage(eatGhostPacmanGymEnv.EatGhostPacmanGymEnv(config_file), is_to_gray=True, is_normalize=False,
                       rescale=None, resize=[84, 84], binary_threshold=1)

    # define the controller_policy module
    policy_args = args
    policy_log_cfg = Dict()
    policy_hyper_params = Dict(dict(gamma=0.99,
                                    tau=1e-3,
                                    buffer_size=int(5e4),  # open-ai baselines: int(1e4)
                                    batch_size=64,  # open-ai baselines: 32
                                    init_random_actions=int(1e3),
                                    update_starts_from=int(1e3),  # open-ai baselines: int(1e4)
                                    multiple_update=1,  # multiple learning updates
                                    train_freq=4,  # in open-ai baselines, train_freq = 4
                                    max_traj_length=5000,  # maximum length of a single rollout
                                    reward_clip=[-1.0, 1.0],
                                    gradient_clip=None,  # dueling: 10.0
                                    max_alpha=0.1,  # entropy coefficient
                                    min_alpha=0.01,
                                    alpha_decay=0.995,  # entropy coefficient annealing
                                    # N-Step Buffer
                                    n_step=5,   # if n_step <= 1, use common replay buffer otherwise n_step replay buffer
                                    w_n_step=1.0,   # used in n-step update
                                    # Double Q-Learning
                                    use_double_q_update=False,
                                    # Prioritized Replay Buffer
                                    use_prioritized=True,
                                    per_alpha=0.6,  # open-ai baselines default: 0.6, alpha -> 1, full prioritization
                                    per_beta=0.4,   # beta can start small (for stability concern and anneals towards 1
                                    per_eps=1e-6,
                                    # NoisyNet
                                    use_noisy_net=False,
                                    std_init=0.5,
                                    # Epsilon Greedy
                                    max_epsilon=1.0,
                                    min_epsilon=0.01,  # open-ai baselines: 0.01
                                    epsilon_decay=0.995,  # default: 0.9995
                                    # State Processing
                                    frame_stack=3
                                    ))
    policy_network_cfg = Dict(dict(fc_input_size=3136,
                                   nonlinearity=torch.relu,
                                   channels=[32, 64, 64],
                                   kernel_sizes=[8, 4, 3],
                                   strides=[4, 2, 1],
                                   paddings=[0, 0, 0],
                                   fc_hidden_sizes=[512, 512],
                                   fc_hidden_activation=torch.relu))
    policy_optim_cfg = Dict(dict(lr_dqn=1e-4,
                                 adam_eps=1e-6,     # default value in pytorch 1e-6
                                 weight_decay=1e-6,
                                 w_q_reg=0,     # use q_value regularization
                                 ))
    expr_logger.save_config_wandb(config={"policy_hyper_params": policy_hyper_params,
                                          "policy_network_cfg": policy_network_cfg,
                                          "policy_optim_cfg": policy_optim_cfg})

    dqn_agent = SQL_Discrete_Agent(env, policy_args, policy_log_cfg, policy_hyper_params,
                                   policy_network_cfg, policy_optim_cfg, encoder=None, logger=expr_logger)

    if args.test:
        dqn_agent.test()
    else:
        dqn_agent.train()

    expr_logger.save_trajectories(is_save_utility=False)


if __name__ == '__main__':
    run_game()
