# -*- coding: utf-8 -*-
"""Run module for SACfD on LunarLanderContinuous-v2.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""
import os
import numpy as np
import torch
import torch.optim as optim

from algorithms.common.networks.mlp import MLP, FlattenMLP, TanhGaussianDistParams
from algorithms.fd.sac_mlp_agent import Agent
from algorithms.common.load_config_utils import loadYAML

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Robot Env Configurations
# conf_str, robot_conf = loadYAML(os.getcwd() + "/../config/fetch_serl_push_env.yaml")

# hyper parameters
hyper_params = {
    "USE_HER": False,
    "N_STEP": 5,
    "IF_PRETRAIN_DEMO": True,  # if do pre-training from demo
    "DEMO_STARTS": 0,
    "GAMMA": 0.99,
    "TAU": 1e-3,
    "BUFFER_SIZE": int(2e5),
    "BATCH_SIZE": 64,
    "AUTO_ENTROPY_TUNING": True,
    "LR_ACTOR": 3e-4,
    "LR_VF": 3e-4,
    "LR_QF1": 3e-4,
    "LR_QF2": 3e-4,
    "LR_ENTROPY": 3e-4,
    "W_ENTROPY": 1e-3,
    "W_MEAN_REG": 1e-3,
    "W_STD_REG": 1e-3,
    "W_PRE_ACTIVATION_REG": 0.0,
    "DELAYED_UPDATE": 2,
    "PRETRAIN_STEP": 100,
    "MULTIPLE_LEARN": 2,  # multiple learning updates
    "LAMBDA1": 1.0,  # N-step return weight
    "LAMBDA2": 1e-5,  # l2 regularization weight
    "LAMBDA3": 1.0,  # actor loss contribution of prior weight
    "PER_ALPHA": 0.6,
    "PER_BETA": 0.4,
    "PER_EPS": 1e-6,
    "PER_EPS_DEMO": 1.0,
    "INITIAL_RANDOM_ACTION": int(5e3),
    # "SIMPLE_STATES_SIZE": robot_conf["env"]["num_simple_states"],
    # "NUM_PREDICATES": len(robot_conf["env"]["predicates_list"]),
    "NETWORK": {
        "ACTOR_HIDDEN_SIZES": [256, 256],
        "VF_HIDDEN_SIZES": [256, 256],
        "QF_HIDDEN_SIZES": [256, 256],
    },
}


def get(env, args):
    """Run training or test.

    Args:
        env (gym.Env): openAI Gym environment with continuous action space
        args (argparse.Namespace): arguments including training settings

    """

    # Load robot env configs
    conf_str, robot_conf = loadYAML(args.robot_env_config)
    hyper_params["SIMPLE_STATES_SIZE"] = robot_conf["env"]["num_simple_states"]
    hyper_params["NUM_PREDICATES"] = len(robot_conf["env"]["predicates_list"])


    state_dim = hyper_params["SIMPLE_STATES_SIZE"] #env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    hidden_sizes_actor = hyper_params["NETWORK"]["ACTOR_HIDDEN_SIZES"]
    hidden_sizes_vf = hyper_params["NETWORK"]["VF_HIDDEN_SIZES"]
    hidden_sizes_qf = hyper_params["NETWORK"]["QF_HIDDEN_SIZES"]

    # target entropy
    target_entropy = -np.prod((action_dim,)).item()  # heuristic

    # create actor
    actor = TanhGaussianDistParams(
        input_size=state_dim, output_size=action_dim, hidden_sizes=hidden_sizes_actor
    ).to(device)

    # create v_critic
    vf = MLP(input_size=state_dim, output_size=1, hidden_sizes=hidden_sizes_vf).to(
        device
    )
    vf_target = MLP(
        input_size=state_dim, output_size=1, hidden_sizes=hidden_sizes_vf
    ).to(device)
    vf_target.load_state_dict(vf.state_dict())

    # create q_critic
    qf_1 = FlattenMLP(
        input_size=state_dim + action_dim, output_size=1, hidden_sizes=hidden_sizes_qf
    ).to(device)
    qf_2 = FlattenMLP(
        input_size=state_dim + action_dim, output_size=1, hidden_sizes=hidden_sizes_qf
    ).to(device)

    # create optimizers
    actor_optim = optim.Adam(
        actor.parameters(),
        lr=hyper_params["LR_ACTOR"],
        weight_decay=hyper_params["LAMBDA2"],
    )
    vf_optim = optim.Adam(
        vf.parameters(), lr=hyper_params["LR_VF"], weight_decay=hyper_params["LAMBDA2"]
    )
    qf_1_optim = optim.Adam(
        qf_1.parameters(),
        lr=hyper_params["LR_QF1"],
        weight_decay=hyper_params["LAMBDA2"],
    )
    qf_2_optim = optim.Adam(
        qf_2.parameters(),
        lr=hyper_params["LR_QF2"],
        weight_decay=hyper_params["LAMBDA2"],
    )

    # make tuples to create an agent
    models = (actor, vf, vf_target, qf_1, qf_2)
    optims = (actor_optim, vf_optim, qf_1_optim, qf_2_optim)

    # create an agent
    return Agent(env, args, hyper_params, models, optims, target_entropy, None)
