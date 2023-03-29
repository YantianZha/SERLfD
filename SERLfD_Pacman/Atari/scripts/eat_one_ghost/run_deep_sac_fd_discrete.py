# run_deep_sac_fd_discrete.py
# ------------------
# script to run soft actor-critic from demonstration algorithm

import argparse
import os

import torch
import gym

from eat_ghost_env import eatGhostPacmanGymEnv
from addict import Dict

from learning_agents.rl.sac.sac_fd_discrete import SACfD_Discrete_Agent
from utils.experiment_record_utils import ExperimentLogger
from utils.gym_atari_env_wrapper import PytorchImage

TEMP_RESULT_SAVING_DIR = 'tmp/'
RESULT_SAVING_DIR = '../../experiment_results/'

experiment_log_dir = TEMP_RESULT_SAVING_DIR

experiment_name = 'sacfd-pacman'
wandb_project_name = 'pacman-eat-1-ghost'

demo_dir = '../../experiment_results/pacman_eat_1_ghost_demo_5/'
demo_expr_id = 'eat_1_ghost_5_demo_aggressive_2'
demo_fname = demo_dir + demo_expr_id + '/traj_' + demo_expr_id + '.pickle'


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
    parser.add_argument("--iteration-num", type=int, default=10000,
                        help="total iteration num")
    parser.add_argument("--avg_score_window", dest="avg_score_window", type=int, default=100,
                        help="avg score window size")
    parser.add_argument("--eval_score_window", dest="eval_score_window", type=int, default=20,
                        help="avg evaluation score window size")
    parser.add_argument("--eval_period", type=int, default=5,
                        help="evaluation period (unit: episode)")
    parser.add_argument("--deterministic", dest="deterministic", action="store_true", default=False,
                        help="deterministic action")
    parser.add_argument("--demo-path", type=str, default=demo_fname,
                        help="demonstration path for learning from demo")

    return parser.parse_args()


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

    # define the agent
    log_cfg = Dict()
    hyper_params = Dict(dict(gamma=0.99,
                             tau=1e-3,
                             buffer_size=int(5e4),  # open-ai baselines: int(1e4)
                             batch_size=64,
                             init_random_actions=int(2e4),
                             update_starts_from=int(1e3),
                             max_traj_length=1000,
                             multiple_update=1,  # multiple learning updates
                             train_freq=4,  # in open-ai baselines, train_freq = 4
                             target_update_freq=1,
                             target_entropy_ratio=0.98,
                             gradient_clip=None,
                             reward_scale=1.0,
                             reward_clip=[-1.0, 1.0],   # set to None to disable reward clipping
                             w_entropy=1e-3,
                             auto_entropy_tuning=True,
                             deterministic_action=args.deterministic,
                             # State Processing
                             frame_stack=3,
                             # N-Step Buffer
                             n_step=5,  # if n_step <= 1, use common replay buffer otherwise n_step replay buffer
                             w_n_step=1.0,  # used in n-step update
                             # Prioritized Replay Buffer
                             use_prioritized=True,
                             per_alpha=0.4,  # paper default: 0.4, alpha -> 1, full prioritization
                             per_beta=0.6,  # paper default: 0.6, beta can start small for stability concern and anneals towards 1
                             per_eps=1e-4,  # default: 1e-3
                             per_eps_demo=1.0,  # default: 1.0
                             # LfD setting
                             policy_bc_margin=0.8,
                             max_policy_bc_weight=0,
                             min_policy_bc_weight=0,
                             policy_bc_weight_decay=0.99,
                             margin=0.8,
                             pretrain_step=int(1e3),  # the paper use 1,000,000 mini-batch updates
                             max_lambda2=1.0,  # Supervised loss weight
                             min_lambda2=1.0,  # Supervised loss weight
                             lambda2_decay=0.99,  # 1.0 means no lambda 2 annealing
                             discrete_to_one_hot=False,
                             ))
    network_cfg = Dict(dict(use_cnn=True,
                            # MLP setting
                            fc_hidden_sizes_actor=[256, 256],
                            fc_hidden_sizes_qf=[256, 256],
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
                          eps_entropy=1e-8,
                          eps_qf=1e-8,
                          eps_actor=1e-8,
                          weight_decay=0))
    expr_logger.save_config_wandb(config={"policy_hyper_params": hyper_params,
                                          "policy_network_cfg": network_cfg,
                                          "policy_optim_cfg": optim_cfg})

    sac_agent = SACfD_Discrete_Agent(env, args, log_cfg, hyper_params,
                                     network_cfg, optim_cfg, demo_preprocessor=env.observation,
                                     encoder=None, logger=expr_logger)

    if args.test:
        sac_agent.test()
    else:
        sac_agent.train()

    expr_logger.save_trajectories(is_save_utility=True)


if __name__ == '__main__':
    run_game()
