# -*- coding: utf-8 -*-
"""Run module for TD3 on LunarLanderContinuous-v2.

- Author: whikwon
- Contact: whikwon@gmail.com
"""

import torch
import torch.optim as optim
import os
from algorithms.common.networks.cnn import Conv2d_Flatten_MLP_v2
from algorithms.common.noise import GaussianNoise
from algorithms.td3.agent_cnn import Agent
from algorithms.common.load_config_utils import loadYAML

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Robot Env Configurations
conf_str, robot_conf = loadYAML(os.getcwd() + "/../config/fetch_serl_push_env.yaml")

# hyper parameters
hyper_params = {
    "GAMMA": 0.99,
    "TAU": 5e-3,
    "BUFFER_SIZE": int(5e5),
    "BATCH_SIZE": 16,
    "LR_ACTOR": 1e-3,
    "LR_CRITIC": 1e-3,
    "WEIGHT_DECAY": 0.0,
    # "WEIGHT_DECAY": 1e-5,  # l2 regularization weight
    "EXPLORATION_NOISE": 0.1,
    "TARGET_POLICY_NOISE": 0.2,
    "TARGET_POLICY_NOISE_CLIP": 0.5,
    "POLICY_UPDATE_FREQ": 2,
    "INITIAL_RANDOM_ACTIONS": 5e3,
    "IF_PRETRAIN_DEMO": False,
    "MULTIPLE_LEARN": 1,
    "SIMPLE_STATES_SIZE": robot_conf["env"]["num_simple_states"],
    "NUM_PREDICATES": len(robot_conf["env"]["predicates_list"]),
    "NETWORK": {"ACTOR_CHANNELS": [32, 32, 64, 64, 128], "CRITIC_CHANNELS": [32, 32, 64, 64, 128],
                "ACTOR_KERNEL_SZ": [8, 8, 4, 4, 3], "CRITIC_KERNEL_SZ": [8, 8, 4, 4, 3],
                "ACTOR_STRIDES": [4, 4, 2, 2, 1], "CRITIC_STRIDES": [4, 4, 2, 2, 1],
                "ACTOR_PADDINGS": [4, 4, 2, 2, 0], "CRITIC_PADDINGS": [4, 4, 2, 2, 0],
                "ACTOR_FC_HIDDEN_SIZES": [400, 200], "CRITIC_FC_HIDDEN_SIZES": [400, 200]},
    # "NETWORK": {"ACTOR_CHANNELS": [32, 64, 64, 128], "CRITIC_CHANNELS": [32, 64, 64, 128],
    #             "ACTOR_KERNEL_SZ": [8, 4, 4, 3], "CRITIC_KERNEL_SZ": [8, 4, 4, 3],
    #             "ACTOR_STRIDES": [4, 2, 2, 1], "CRITIC_STRIDES": [4, 2, 2, 1],
    #             "ACTOR_PADDINGS": [4, 2, 2, 0], "CRITIC_PADDINGS": [4, 2, 2, 0],
    #             "ACTOR_FC_HIDDEN_SIZES": [400, 200], "CRITIC_FC_HIDDEN_SIZES": [400, 200]},
}


def get(env, args):
    """Run training or test.

    Args:
        env (gym.Env): openAI Gym environment with continuous action space
        args (argparse.Namespace): arguments including training settings

    """
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_dim_actor = action_dim - robot_conf['fetch']['exe_group_num'] if robot_conf['fetch']['exe_single_group'] else action_dim

    simple_states_dim = hyper_params["SIMPLE_STATES_SIZE"]
    channels_actor = hyper_params["NETWORK"]["ACTOR_CHANNELS"]
    channels_critic = hyper_params["NETWORK"]["CRITIC_CHANNELS"]
    kernel_sz_actor = hyper_params["NETWORK"]["ACTOR_KERNEL_SZ"]
    kernel_sz_critic = hyper_params["NETWORK"]["CRITIC_KERNEL_SZ"]
    paddings_actor = hyper_params["NETWORK"]["ACTOR_PADDINGS"]
    paddings_critic = hyper_params["NETWORK"]["CRITIC_PADDINGS"]
    strides_actor = hyper_params["NETWORK"]["ACTOR_STRIDES"]
    strides_critic = hyper_params["NETWORK"]["CRITIC_STRIDES"]
    fc_hidden_sizes_actor = hyper_params["NETWORK"]["ACTOR_FC_HIDDEN_SIZES"]
    fc_hidden_sizes_critic = hyper_params["NETWORK"]["CRITIC_FC_HIDDEN_SIZES"]

    # create actor
    actor = Conv2d_Flatten_MLP_v2(
        observation_space=(env.observation_space, simple_states_dim),
        channels=channels_actor,
        kernel_sizes=kernel_sz_actor,
        paddings=paddings_actor,
        strides=strides_actor,
        fc_hidden_sizes=fc_hidden_sizes_actor,
        fc_output_size=action_dim_actor,  # [action_dim-2, 2],
        fc_output_activation=torch.tanh,
        use_maxpool=True,
    ).to(device)

    actor_target = Conv2d_Flatten_MLP_v2(
        observation_space=(env.observation_space, simple_states_dim),
        channels=channels_actor,
        kernel_sizes=kernel_sz_actor,
        paddings=paddings_actor,
        strides=strides_actor,
        fc_hidden_sizes=fc_hidden_sizes_actor,
        fc_output_size=action_dim_actor,  # [action_dim-2, 2],
        fc_output_activation=torch.tanh,
        use_maxpool=True,
    ).to(device)
    actor_target.load_state_dict(actor.state_dict())

    # create critic1
    critic1 = Conv2d_Flatten_MLP_v2(
        observation_space=(env.observation_space, simple_states_dim + action_dim),
        channels=channels_critic,
        kernel_sizes=kernel_sz_critic,
        paddings=paddings_critic,
        strides=strides_critic,
        fc_hidden_sizes=fc_hidden_sizes_critic,
        fc_output_size=1,
        use_maxpool=True,
    ).to(device)

    critic1_target = Conv2d_Flatten_MLP_v2(
        observation_space=(env.observation_space, simple_states_dim + action_dim),
        channels=channels_critic,
        kernel_sizes=kernel_sz_critic,
        paddings=paddings_critic,
        strides=strides_critic,
        fc_hidden_sizes=fc_hidden_sizes_critic,
        fc_output_size=1,
        use_maxpool=True,
    ).to(device)
    critic1_target.load_state_dict(critic1.state_dict())

    # create critic2
    critic2 = Conv2d_Flatten_MLP_v2(
        observation_space=(env.observation_space, simple_states_dim + action_dim),
        channels=channels_critic,
        kernel_sizes=kernel_sz_critic,
        paddings=paddings_critic,
        strides=strides_critic,
        fc_hidden_sizes=fc_hidden_sizes_critic,
        fc_output_size=1,
        use_maxpool=True,
    ).to(device)

    critic2_target = Conv2d_Flatten_MLP_v2(
        observation_space=(env.observation_space, simple_states_dim + action_dim),
        channels=channels_critic,
        kernel_sizes=kernel_sz_critic,
        paddings=paddings_critic,
        strides=strides_critic,
        fc_hidden_sizes=fc_hidden_sizes_critic,
        fc_output_size=1,
        use_maxpool=True,
    ).to(device)
    critic2_target.load_state_dict(critic2.state_dict())

    # concat critic parameters to use one optim
    critic_parameters = list(critic1.parameters()) + list(critic2.parameters())

    # create optimizer
    actor_optim = optim.Adam(
        actor.parameters(),
        lr=hyper_params["LR_ACTOR"],
        weight_decay=hyper_params["WEIGHT_DECAY"],
    )

    critic_optim = optim.Adam(
        critic_parameters,
        lr=hyper_params["LR_CRITIC"],
        weight_decay=hyper_params["WEIGHT_DECAY"],
    )

    # noise
    exploration_noise = GaussianNoise(
        action_dim,
        min_sigma=hyper_params["EXPLORATION_NOISE"],
        max_sigma=hyper_params["EXPLORATION_NOISE"],
    )

    target_policy_noise = GaussianNoise(
        action_dim,
        min_sigma=hyper_params["TARGET_POLICY_NOISE"],
        max_sigma=hyper_params["TARGET_POLICY_NOISE"],
    )

    # make tuples to create an agent
    models = (actor, actor_target, critic1, critic1_target, critic2, critic2_target)
    optims = (actor_optim, critic_optim)
    noises = (exploration_noise, target_policy_noise)

    # create an agent
    return Agent(env, args, hyper_params, models, optims, noises)
