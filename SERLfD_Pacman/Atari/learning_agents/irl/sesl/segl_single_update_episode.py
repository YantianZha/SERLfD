import argparse
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import gym

from learning_agents.agent import Agent
from learning_agents.architectures.cnn import Conv2d_MLP_Model
from learning_agents.common.trajectory_buffer import TrajectoryBuffer
import learning_agents.common.common_utils as common_utils
from learning_agents.architectures.mlp import MLP
from utils.trajectory_utils import split_fixed_length_indexed_traj, extract_experiences_from_indexed_trajs, \
    demo_discrete_actions_to_one_hot, read_expert_demo, get_indexed_trajs, read_expert_demo_utility_info
from learning_agents.utils.utils import ConfigDict, IndexedTraj
from utils.trajectory_utils import TRAJECTORY_INDEX

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOG_REG = 1e-8


# noinspection DuplicatedCode
class SEGL_Single_Update_Episode(Agent):
    """
    Self-Explanation Guided Learning on single state with episode update

    Attributes:
        ####################### attributes of the IRL (explainer) ##########################
        env (gym.Env): openAI Gym environment
        irl_model (torch.nn): the irl/explainer model
        irl_model_optim (torch.optim): the irl/explainer network's optimizer
        args (argparse.Namespace): arguments including hyperparameters and training settings
        hyper_params (ConfigDict): hyper-parameters
        network_cfg (ConfigDict): config of network for training agent
        encoder(nn): if encoder is not None, the irl/explainer will use the encoder to preprocess the state
        optim_cfg (ConfigDict): config of optimizer
        state_dim (int): state size of env
        action_dim (int): action size of env
        good_traj_buf (TrajectoryBuffer): buffer of good trajectories (including expert demo), trajs are stored in indexed format
        bad_traj_buf (TrajectoryBuffer): buffer of bad trajectories, trajs are stored in indexed format
        curr_state (np.ndarray): temporary storage of the current state
        total_step (int): total step numbers
        i_iteration (int): current iteration number
        ####################### attributes of the RL (controller_policy) ##############################
        policy_network (Agent): the pre-defined controller_policy network
    """

    pass
