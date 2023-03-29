from abc import ABC, abstractmethod
from enum import Enum
from learning_agents.agent import Agent

import gym
import numpy as np
import torch

from learning_agents.common.noise import OUNoise


class TRAJECTORY_INFO_INDEX(Enum):
    TRAJ_LOG_PROB = 'traj_log_prob'
    ACTION_LOG_PROBS = 'traj_action_log_probs'


class BasePolicy(Agent, ABC):

    def __init__(self, env, args, log_cfg, hyper_params, network_cfg, optim_cfg, logger=None):
        Agent.__init__(self, env, args, log_cfg)
        self.env = env
        self.args = args
        self.log_cfg = log_cfg
        self.total_step = 0
        self.episode_step = 0
        self.i_episode = 0

        self.hyper_params = hyper_params
        self.network_cfg = network_cfg
        self.optim_cfg = optim_cfg
        self.logger = logger

        # get state space info
        self.state_dim = self.env.observation_space.shape[0]
        # check if it's single channel or multi channel
        self.state_channel = 1 if len(self.env.observation_space.shape) == 2 else self.env.observation_space.shape[0]

        # get action space info
        self.is_discrete = False
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.action_dim = self.env.action_space.n
            self.is_discrete = True
        else:
            self.action_dim = self.env.action_space.shape[0]

        self.policy = None

    @abstractmethod
    def get_action(self, state, is_test=False, is_random=False, info=None):
        """
        return the selected action given the state
        :param: state: np.ndarray
        :param: is_test: is doing test
        :param: is_random: whether returns random action
        :param: info: a dict containing auxiliary information
        """
        pass

    @abstractmethod
    def evaluate_states_actions(self, states, actions, info=None):
        """
        get the log probability of actions
        :param: states: np.ndarray
        :param: actions: np.ndarray
        :param: info: a dict containing auxiliary information
        :return: a numpy array of log probability of each action
        """
        pass

    @abstractmethod
    def evaluate_state_action(self, state, action, info=None):
        """
        Compute the probability of the state, action pair
        :param: info: a dict containing auxiliary information
        :return: (torch.Tensor) the log probability of the state, action pair
        """
        pass

    @abstractmethod
    def update_policy(self, indexed_trajs, irl_model):
        """
        Update the controller_policy network. Some approaches only use trajectories from current iteration
            to do the update while some others might also sample from past experiences. Thus, trajectories from
            current iteration need to be passed as arguments if necessary
        """
        pass

    @abstractmethod
    def get_wandb_watch_list(self):
        pass

    @abstractmethod
    def write_policy_log(self, log_value):
        pass

    @abstractmethod
    def get_policy_wandb_log(self, log_value):
        pass

    @abstractmethod
    def write_policy_wandb_log(self, log_value, step=None):
        pass

    @abstractmethod
    def pretrain_policy(self, irl_model):
        pass

    def add_transition_to_memory(self, transition, utility_info=None):
        pass

    ###################################################################
    ######### Unused Abstract Inherited Function From Agent ###########
    ###################################################################

    def pretrain(self):
        pass

    def train(self):
        """ controller_policy module doesn't have train cycles """
        pass

    def step(self, action):
        """ controller_policy module doesn't have train cycles, so no need for step function """
        pass

    def select_action(self, state):
        """ return the selected action given the state """
        pass



