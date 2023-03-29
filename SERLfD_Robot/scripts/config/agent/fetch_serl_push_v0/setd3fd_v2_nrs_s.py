# -*- coding: utf-8 -*-
import os
import torch
import torch.optim as optim

from algorithms.common.networks.mlp import MLP
from algorithms.common.noise import GaussianNoise
from algorithms.fd.se_td3_mlp_agent_v2 import Agent

from algorithms.common.load_config_utils import loadYAML

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Robot Env Configurations
# conf_str, robot_conf = loadYAML(os.getcwd() + "/../config/fetch_serl_push_env.yaml")

# hyper parameters
hyper_params = {
    "N_STEP": 5, # n_step q update for LfD
    "IF_PRETRAIN_DEMO": True, # if do pre-training from demo
    "DEMO_STARTS": 0,
    "GAMMA": 0.99,
    "TAU": 1e-3,
    "BUFFER_SIZE": int(2e5),
    "BATCH_SIZE": 64,
    "MINI_BATCH_SIZE": 32,
    "LR_ACTOR": 3e-4,
    "LR_CRITIC": 3e-4,
    "LR_EXPLAINER": 1e-3,
    "EXPLORATION_NOISE": 0.1,
    "TARGET_POLICY_NOISE": 0.1,
    "EXPLORATION_NOISE_MIN": 0.005,
    "TARGET_POLICY_NOISE_MIN": 0.005,
    "TARGET_POLICY_NOISE_CLIP": 0.5,
    "NOISE_DECAY_PERIOD": 50 * 100,
    "POLICY_UPDATE_FREQ": 2, # How much delay you want to update the actor than Q net
    "INITIAL_RANDOM_ACTIONS": int(5e3),
    "PRETRAIN_STEP": 200, #100,
    "MULTIPLE_LEARN": 2,  # multiple learning updates
    "LAMBDA1": 1.0,  # N-step return weight
    "LAMBDA2": 1e-5,  # l2 regularization weight
    "LAMBDA3": 1.0,  # actor loss contribution of prior weight
    "PER_ALPHA": 0.3,
    "PER_BETA": 1.0,
    "PER_EPS": 1e-6,
    "PER_EPS_DEMO": 1.0,
    # "SIMPLE_STATES_SIZE": robot_conf["env"]["num_simple_states"],
    # "NUM_PREDICATES": len(robot_conf["env"]["predicates_list"]),
    "STATES_WITH_PREDICATES": True,
    "STATES_WITH_UTILITIES": True,
    "ONE_HOT_PREDICATES": True,
    "BIAS_IN_PREDICATE": True,
    "NO_SHAPING": True,
    "MANUAL_SHAPING": False,
    "PRINT_SHAPING": True,
    "SHAPING_REWARD_WEIGHT": 0.05,
    "NEGATIVE_REWARD_ONLY": True,
    "SHAPING_REWARD_CLIP": [-10, 10],
    "SE_GRAD_CLIP": 10,
    "MAX_ENERGY": 50,
    "LOG_REG": 1e-8,
    "NETWORK": {"ACTOR_HIDDEN_SIZES": [400, 300], "CRITIC_HIDDEN_SIZES": [400, 300]},
}


def get(env, args):
    """
    Run training or test.

    Args:
        env (gym.Env): openAI Gym environment with continuous action space
        args (argparse.Namespace): arguments including training settings

    """
    # Load robot env configs
    conf_str, robot_conf = loadYAML(args.robot_env_config)
    hyper_params["SIMPLE_STATES_SIZE"] = robot_conf["env"]["num_simple_states"]
    hyper_params["NUM_PREDICATES"] = len(robot_conf["env"]["predicates_list"])

    n_predicate_key = hyper_params["NUM_PREDICATES"]
    if hyper_params["ONE_HOT_PREDICATES"]:
        exp_out_sz = n_predicate_key * 2 + 1 if hyper_params["BIAS_IN_PREDICATE"] else n_predicate_key * 2
    else:
        exp_out_sz = n_predicate_key + 1 if hyper_params["BIAS_IN_PREDICATE"] else n_predicate_key

    explainer_state_dim = hyper_params["SIMPLE_STATES_SIZE"] - hyper_params["NUM_PREDICATES"]  #env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    # action_dim_actor = action_dim - robot_conf['fetch']['exe_group_num'] if robot_conf['fetch']['exe_single_group'] else action_dim

    simple_states_dim = hyper_params["SIMPLE_STATES_SIZE"]
    if hyper_params["STATES_WITH_UTILITIES"]:
        simple_states_dim_policy = simple_states_dim + hyper_params["NUM_PREDICATES"]


    hidden_sizes_actor = hyper_params["NETWORK"]["ACTOR_HIDDEN_SIZES"]
    hidden_sizes_critic = hyper_params["NETWORK"]["CRITIC_HIDDEN_SIZES"]

    # create actor
    actor = MLP(
        input_size=simple_states_dim_policy,
        output_size=action_dim,
        hidden_sizes=hidden_sizes_actor,
        output_activation=torch.tanh,
    ).to(device)

    actor_target = MLP(
        input_size=simple_states_dim_policy,
        output_size=action_dim,
        hidden_sizes=hidden_sizes_actor,
        output_activation=torch.tanh,
    ).to(device)
    actor_target.load_state_dict(actor.state_dict())

    # create critic1
    critic1 = MLP(
        input_size=simple_states_dim_policy + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes_critic,
    ).to(device)

    critic1_target = MLP(
        input_size=simple_states_dim_policy + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes_critic,
    ).to(device)
    critic1_target.load_state_dict(critic1.state_dict())

    # create critic2
    critic2 = MLP(
        input_size=simple_states_dim_policy + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes_critic,
    ).to(device)

    critic2_target = MLP(
        input_size=simple_states_dim_policy + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes_critic,
    ).to(device)
    critic2_target.load_state_dict(critic2.state_dict())


    explainer = MLP(
        input_size=explainer_state_dim,
        output_size=exp_out_sz,
        hidden_sizes=hidden_sizes_critic,
    ).to(device)

    # concat critic parameters to use one optim
    critic_parameters = list(critic1.parameters()) + list(critic2.parameters())

    # create optimizer
    actor_optim = optim.Adam(
        actor.parameters(),
        lr=hyper_params["LR_ACTOR"],
        weight_decay=hyper_params["LAMBDA2"],
    )

    critic_optim = optim.Adam(
        critic_parameters,
        lr=hyper_params["LR_CRITIC"],
        weight_decay=hyper_params["LAMBDA2"],
    )

    explainer_optim = optim.Adam(
            explainer.parameters(),
            lr=hyper_params["LR_EXPLAINER"],
            weight_decay=hyper_params["LAMBDA2"]
        )

    # noise
    exploration_noise = GaussianNoise(
        action_dim,
        min_sigma=hyper_params["EXPLORATION_NOISE_MIN"],
        max_sigma=hyper_params["EXPLORATION_NOISE"],
        decay_period=hyper_params["NOISE_DECAY_PERIOD"]
    )

    target_policy_noise = GaussianNoise(
        action_dim,
        min_sigma=hyper_params["TARGET_POLICY_NOISE_MIN"],
        max_sigma=hyper_params["TARGET_POLICY_NOISE"],
        decay_period=hyper_params["NOISE_DECAY_PERIOD"]
    )

    # make tuples to create an agent
    models = (actor, actor_target, critic1, critic1_target, critic2, critic2_target, explainer)
    optims = (actor_optim, critic_optim, explainer_optim)
    noises = (exploration_noise, target_policy_noise)

    # create an agent
    return Agent(env, args, hyper_params, models, optims, noises, robot_conf=robot_conf)
