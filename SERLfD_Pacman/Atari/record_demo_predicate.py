# run_segl_single_dqfd.py
# ------------------
# script to run the eat-ghost pacman game (integrated with Open AI gym)

import os
import argparse
from collections import deque

import cv2
import numpy as np
import torch
from skimage.transform import resize

from eat_ghost_env import eatGhostPacmanGymEnv
from learning_agents.common import common_utils
from learning_agents.controller_policy.segl_dqfd_policy import SEGL_DQfD_Policy
from learning_agents.controller_policy.segl_sql_fd_discrete_policy import SEGL_SQLfD_Discrete_Policy
from learning_agents.controller_policy.segl_sql_fd_discrete_predicate_policy import SEGL_SQLfD_Discrete_Predicate_Policy
from learning_agents.irl.sesl.segl_single_update_step import SEGL_Single_Update_Step
from addict import Dict

from utils.experiment_record_utils import ExperimentLogger
from utils.gym_atari_env_wrapper import PytorchImage
from utils.img2video import im2predicates
from utils.trajectory_utils import TRAJECTORY_INDEX

TEMP_RESULT_SAVING_DIR = 'tmp/'
RESULT_SAVING_DIR = 'experiment_results/'

experiment_log_dir = TEMP_RESULT_SAVING_DIR
experiment_name = 'segl-sql-fd-pacman'
wandb_project_name = 'pacman-eat-1-ghost'

demo_fname = ''
demo_util_fname = ''

explainer_model_path = '/Users/lguan/Desktop/model/explainer_169605_0.tar'
policy_model_path = '/Users/lguan/Desktop/model/dqfd_169605_0.tar'


def parse_args():
    # configurations
    parser = argparse.ArgumentParser(description="arguments of segl with single step update")
    parser.add_argument("--virtual-display", dest="virtual_display", action="store_true", default=False,
                        help="open virtual display")
    parser.add_argument("--seed", type=int, default=500,
                        help="random seed for reproducibility")
    parser.add_argument("--cfg-path", type=str, default=None,
                        help="config path")
    parser.add_argument("--test", dest="test", default=True,
                        help="test mode (no training)")
    parser.add_argument("--explainer-load-from", type=str, default=explainer_model_path,
                        help="load the saved model and optimizer at the beginning")
    parser.add_argument("--policy-load-from", type=str, default=policy_model_path,
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
    parser.add_argument("--save-period", type=int, default=1000,
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
    parser.add_argument("--demo-util-path", type=str, default=demo_util_fname,
                        help="utility value file path for learning from demo")
    parser.add_argument("--no_shaping", dest="no_shaping", action="store_true", default=False,
                        help="always return zero shaping reward (no shaping reward)")
    parser.add_argument("--print_shaping", dest="print_shaping", action="store_true", default=False,
                        help="print info about shaping reward")
    parser.add_argument("--manual_shaping", dest="manual_shaping", action="store_true", default=False,
                        help="use manually-designed shaping for comparison")

    return parser.parse_args()


def run_game():
    args = parse_args()

    expr_logger = ExperimentLogger(experiment_log_dir, experiment_name, save_trajectories=False)
    expr_logger.redirect_output_to_logfile_as_well()

    expr_logger.set_is_use_wandb(args.use_wandb)
    expr_logger.set_wandb_project_name(wandb_project_name)
    expr_logger.set_wandb()

    config_file = 'experiment_configs/expr_pacman_eat_1_ghost.cnf'
    expr_logger.copy_file(config_file)
    expr_logger.copy_file(os.path.abspath(__file__))
    if args.virtual_display:
        expr_logger.open_virtual_display()

    env = PytorchImage(eatGhostPacmanGymEnv.EatGhostPacmanGymEnv(config_file), is_to_gray=True, is_normalize=False,
                       rescale=None, resize=[84, 84])

    # define the controller_policy module
    policy_args = args
    policy_log_cfg = Dict()
    policy_hyper_params = Dict(dict(gamma=0.99,
                                    tau=1e-3,
                                    buffer_size=int(5e4),  # open-ai baselines: int(1e4)
                                    batch_size=64,  # open-ai baselines: 32
                                    reward_clip=None,
                                    gradient_clip=None,  # dueling: 10.0
                                    max_alpha=0.1,  # entropy coefficient
                                    min_alpha=0.01,
                                    alpha_decay=0.995,  # entropy coefficient annealing
                                    # N-Step Buffer
                                    n_step=3,
                                    # if n_step <= 1, use common replay buffer otherwise n_step replay buffer
                                    # Double Q-Learning
                                    use_double_q_update=False,
                                    # Prioritized Replay Buffer
                                    use_prioritized=True,
                                    per_alpha=0.4,  # paper default: 0.4, alpha -> 1, full prioritization
                                    per_beta=0.6,
                                    # paper default: 0.6, beta can start small for stability concern and anneals towards 1
                                    per_eps=1e-4,  # default: 1e-3
                                    per_eps_demo=1.0,  # default: 1.0
                                    # NoisyNet
                                    use_noisy_net=False,
                                    std_init=0.5,
                                    # Epsilon Greedy
                                    max_epsilon=1.0,
                                    min_epsilon=0.01,  # open-ai baselines: 0.01
                                    epsilon_decay=0.995,  # default: 0.9995
                                    # State Processing
                                    frame_stack=3,
                                    # LfD Setting
                                    margin=0.8,
                                    pretrain_step=int(1e3),  # the paper use 1,000,000 mini-batch updates
                                    lambda1=1.0,  # N-step return weight
                                    max_lambda2=1.0,  # Supervised loss weight
                                    min_lambda2=0.0001,  # Supervised loss weight
                                    lambda2_decay=0.99,  # Supervised loss weight
                                    discrete_to_one_hot=False,
                                    ))
    policy_network_cfg = Dict(dict(fc_input_size=3143,
                                   nonlinearity=torch.relu,
                                   channels=[32, 64, 64],
                                   kernel_sizes=[8, 4, 3],
                                   strides=[4, 2, 1],
                                   paddings=[0, 0, 0],
                                   fc_hidden_sizes=[512, 512],
                                   fc_hidden_activation=torch.relu))
    policy_optim_cfg = Dict(dict(lr_dqn=1e-4,
                                 adam_eps=1e-6,  # default value in pytorch 1e-6
                                 weight_decay=1e-6,
                                 w_q_reg=0,  # use q_value regularization
                                 ))
    expr_logger.save_config_wandb(config={"policy_hyper_params": policy_hyper_params,
                                          "policy_network_cfg": policy_network_cfg,
                                          "policy_optim_cfg": policy_optim_cfg})

    sql_fd_policy = SEGL_SQLfD_Discrete_Predicate_Policy(env, policy_args, policy_log_cfg, policy_hyper_params,
                                                         policy_network_cfg,
                                                         policy_optim_cfg, demo_preprocessor=env.observation,
                                                         encoder=None, logger=expr_logger)

    # define the irl module
    args = parse_args()
    log_cfg = Dict()
    hyper_params = Dict(dict(buffer_size=int(1e4),  # size of (sampled/expert) trajectories buffer
                             n_samples_epoch=128,  # minimum number of sampled state-action pairs per epoch
                             mini_batch_size=64,  # number of states per mini-batch
                             batch_size=512,  # number of states per batch
                             update_starts_from=int(1e3),
                             multiple_update=5,  # number of iteration per update
                             max_traj_length=2000,  # maximum length of a single rollout
                             gradient_clip=10.0,  # dueling: 10.0
                             discrete_to_one_hot=False,  # whether to convert discrete action to one hot representation
                             bias_in_predicate=True,  # whether to add bias term in predicate vector
                             predicate_one_hot=True,
                             # (now only support one hot)  whether use one-hot representation of predicates
                             augmented_feature=True,  # whether add utility value to feature representation
                             shaping_reward_clip=[-10, 10],
                             shaping_reward_weight=0.05,
                             negative_reward_only=True,
                             eval_verion=2,
                             # State Processing
                             frame_stack=3,
                             # Policy Related Setting
                             policy_update_starts_from=int(1e3),  # open-ai baselines: int(1e4)
                             policy_multiple_update=1,  # multiple learning updates
                             policy_train_freq=4,  # in open-ai baselines, train_freq = 4
                             policy_random_actions=int(1e3),
                             ))
    network_cfg = Dict(dict(use_cnn=True,
                            # MLP setting
                            hidden_sizes_irl=[100, 100],
                            # CNN setting
                            fc_input_size=3136,
                            nonlinearity=torch.relu,
                            channels=[32, 64, 64],
                            kernel_sizes=[8, 4, 3],
                            strides=[4, 2, 1],
                            paddings=[0, 0, 0],
                            fc_hidden_sizes=[512, 512],
                            fc_hidden_activation=torch.relu))
    optim_cfg = Dict(dict(lr_explainer=3e-4, weight_decay=1e-6))
    expr_logger.save_config_wandb(config={"irl_hyper_params": hyper_params,
                                          "irl_network_cfg": network_cfg,
                                          "irl_optim_cfg": optim_cfg})

    segl_agent = SEGL_Single_Update_Step(env, sql_fd_policy, args, log_cfg, hyper_params, network_cfg,
                                         optim_cfg, demo_preprocessor=env.observation, encoder=None, logger=expr_logger)

    n_episode = 0
    while n_episode < 1:
        state = env.reset()
        state = np.squeeze(state, axis=0)
        states_queue = deque(maxlen=hyper_params.frame_stack)
        states_queue.extend([state for _ in range(hyper_params.frame_stack)])

        done = False
        i_episode_step = 0
        episode_score = 0

        while not done and i_episode_step < 1000:
            if True:
                env.render()

            # get current predicate
            predicate_values = env.get_current_predicate()
            predicate_vector = segl_agent.to_predicate_vectors([predicate_values])
            stacked_states = np.copy(np.stack(list(states_queue), axis=0))

            util_vector = segl_agent.get_utility_values_vectors(np.expand_dims(stacked_states, axis=0), [predicate_values]) if hyper_params.augmented_feature else None
            action = segl_agent.policy.get_action(stacked_states, is_test=True, is_random=False,
                                                  info={TRAJECTORY_INDEX.PREDICATE_VECTOR.value: predicate_vector, TRAJECTORY_INDEX.UTILITY_VECTOR.value: util_vector})

            # get agent explanation
            print(predicate_values)
            agent_explanation = segl_agent.get_utility_values_dicts(stacked_states, [predicate_values])
            print(agent_explanation)

            rgb_obs = cv2.cvtColor(env.get_rgb_obs(), cv2.COLOR_RGB2BGR)
            rgb_obs = cv2.resize(rgb_obs, (650, 300), interpolation=cv2.INTER_AREA)
            background = cv2.imread('background.png')
            background[0:rgb_obs.shape[0], 0:rgb_obs.shape[1]] = rgb_obs
            rgb_obs = background
            rgb_obs_fname = '/Users/lguan/Desktop/ICRA/demo/tmp/' + 'color_img_' + str(n_episode) \
                            + '_' + str(i_episode_step) + '.jpg'
            cv2.imwrite(rgb_obs_fname, rgb_obs)
            demo_fname = '/Users/lguan/Desktop/ICRA/demo/demo/' + 'frame_' + str(n_episode) \
                            + '_' + str(i_episode_step) + '.jpg'
            im2predicates(rgb_obs_fname, demo_fname, predicate_values, agent_explanation[0])


            next_state, true_reward, done, info = env.step(action)
            next_predicate_values = env.get_current_predicate()
            episode_score += true_reward

            # add next state into states queue
            next_state = np.squeeze(next_state, axis=0)
            states_queue.append(next_state)

            i_episode_step += 1

        n_episode += 1


if __name__ == '__main__':
    run_game()
