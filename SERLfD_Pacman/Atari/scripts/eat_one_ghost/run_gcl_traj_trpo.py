# run_gcl_traj_trpo.py
# ------------------
# script to run the eat-ghost pacman game (integrated with Open AI gym)

import os
import argparse

import gym
from eat_ghost_env import eatGhostPacmanGymEnv
from learning_agents.irl.gan_gcl.gan_gcl_trajectory import GAN_GCL_Trajectory
from learning_agents.controller_policy.sac_policy import SACPolicy
from addict import Dict

from learning_agents.controller_policy.trpo_policy import TRPOPolicy
from utils.experiment_record_utils import ExperimentLogger


TEMP_RESULT_SAVING_DIR = 'tmp/'
RESULT_SAVING_DIR = '../../experiment_results/'

experiment_log_dir = TEMP_RESULT_SAVING_DIR
experiment_name = 'gcl-trpo-lunar-continuous'
wandb_project_name = 'lunar-lander-continuous'

demo_dir = '../../experiment_results/'
demo_expr_id = 'lunar_lander_clean_demo'
demo_fname = demo_dir + demo_expr_id + '/traj_' + demo_expr_id + '.pickle'


def parse_args():
    # configurations
    parser = argparse.ArgumentParser(description="Pytorch RL rl_algorithms")
    parser.add_argument("--seed", type=int, default=777,
                        help="random seed for reproducibility")
    parser.add_argument("--cfg-path", type=str, default=None,
                        help="config path", )
    parser.add_argument("--test", dest="test", action="store_true",
                        help="test mode (no training)")
    parser.add_argument("--load-from", type=str, default=None,
                        help="load the saved model and optimizer at the beginning")
    parser.add_argument("--off-render", dest="render", action="store_false", default=True,
                        help="turn off rendering")
    parser.add_argument("--use-wandb", dest="use_wandb", action="store_false", default=False,
                        help="whether store the results in wandb")
    parser.add_argument("--print-episode-log", dest="print_episode_log", action="store_true",
                        help="whether to print episode score")
    parser.add_argument('--render-freq', type=int, default=20,
                        help='render frequency (default: 20)')
    parser.add_argument("--log", dest="log", action="store_true",
                        help="turn on logging")
    parser.add_argument("--save-period", type=int, default=20,
                        help="save model period")
    parser.add_argument("--iteration-num", type=int, default=5000,
                        help="total iteration num")
    parser.add_argument("--interim-test-num", type=int, default=0,
                        help="number of test during training")
    parser.add_argument("--demo-path", type=str, default=os.path.join(os.getcwd(), demo_fname),
                        help="demonstration path for learning from demo")

    return parser.parse_args()


def run_game():
    args = parse_args()

    expr_logger = ExperimentLogger(experiment_log_dir, experiment_name)
    expr_logger.redirect_output_to_logfile_as_well()
    expr_logger.set_is_use_wandb(args.use_wandb)
    # config_file = 'experiment_configs/expr_pacman_eat_1_ghost.cnf'
    # expr_logger.copy_file(config_file)
    # env = eatGhostPacmanGymEnv.EatGhostPacmanGymEnv(config_file)
    env = gym.make('LunarLanderContinuous-v2')
    expr_logger.set_wandb_project_name(wandb_project_name)

    # define the controller_policy module
    policy_args = args
    policy_log_cfg = Dict()
    policy_hyper_params = Dict(dict(gamma=0.99,
                                    lamda=0.98,
                                    tau=0.95,  # gae (default: 0.95)
                                    damping=1e-2,  # damping (default: 1e-2)
                                    max_kl=1e-2,
                                    is_fusion=False,  # whether to use past experiences
                                    # total sample size (state-action pairs) to collect before TRPO update (default: 2048)
                                    buffer_size=int(1e3),
                                    traj_batch_size=128,  # number of trajectories used to update controller_policy each iteration
                                    initial_random_action=int(1000),  # must be zero if used as controller_policy in irl
                                    num_iteration_update=2,  # multiple learning updates per iteration
                                    ))
    policy_network_cfg = Dict(dict(use_cnn=False,
                                   hidden_sizes_actor=[100, 100],
                                   hidden_sizes_vf=[100, 100]))
    policy_optim_cfg = Dict(dict(lr_actor=3e-4,
                                 lr_vf=3e-4,
                                 weight_decay=1e-5  # l2 regularization regression (default: 1e-3)
                                 ))

    trpo_policy = TRPOPolicy(env, policy_args, policy_log_cfg, policy_hyper_params,
                             policy_network_cfg, policy_optim_cfg, encoder=None, logger=expr_logger)

    # define the irl module
    args = parse_args()
    log_cfg = Dict()
    hyper_params = Dict(dict(buffer_size=int(1e3),  # size of (sampled/expert) trajectories buffer
                             n_samples_epoch=2048,      # minimum number of sampled state-action pairs per epoch
                             is_fusion=True,  # whether also use past samples to update irl (otherwise only use new sampled traj)
                             traj_batch_size=1024,  # number of trajectories used to update discriminator per iteration (-1 means using all trajectories)
                             traj_fixed_length=16,  # the fixed length of trajectories
                             batch_size=64,     # mini-batch size of state-action pairs to update the discriminator
                             num_iteration_update=2,  # number of iteration per update
                             max_traj_length=1000,  # maximum length of a single rollout
                             ))
    network_cfg = Dict(dict(hidden_sizes_irl=[100, 100]))
    optim_cfg = Dict(dict(irl_lr=3e-4, weight_decay=1e-4))

    gan_gcl = GAN_GCL_Trajectory(env, trpo_policy, args, log_cfg, hyper_params, network_cfg, optim_cfg, encoder=None, logger=expr_logger)
    gan_gcl.train()

    expr_logger.save_trajectories(is_save_utility=True)


if __name__ == '__main__':
    run_game()
