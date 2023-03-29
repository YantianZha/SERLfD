from abc import ABC, abstractmethod
import argparse
import os
import shutil
import subprocess
from typing import Tuple, Union

import gym
import numpy as np
import torch


class Agent(ABC):
    """
    Abstract Agent used for all agents.
    Attributes:
        env (gym.Env): openAI Gym environment
        args (argparse.Namespace): arguments including hyperparameters and training settings
        log_cfg (ConfigDict): configuration for saving log and checkpoint
        env_name (str) : gym env name for logging
        sha (str): sha code of current git commit
        state_dim (int): dimension of states
        action_dim (int): dimension of actions
        is_discrete (bool): shows whether the action is discrete
    """

    def __init__(self, env, args, log_cfg):
        """Initialize."""
        self.args = args
        self.env = env
        self.log_cfg = log_cfg

        self.env_name = env.spec.id if env.spec is not None else env.name

        if isinstance(env.action_space, gym.spaces.Discrete):
            self.is_discrete = True
        else:
            self.is_discrete = False

    @abstractmethod
    def select_action(self, state):
        """
        state: np.ndarray
        :return: Union[torch.Tensor, np.ndarray]
        """
        pass

    @abstractmethod
    def step(self, action):
        """
        action: Union[torch.Tensor, np.ndarray]
        :return: Tuple[np.ndarray, np.float64, bool, dict]
        """
        pass

    @abstractmethod
    def update_model(self):
        """
        :return:Tuple[torch.Tensor, ...]
        """
        pass

    @abstractmethod
    def load_params(self, path):
        pass

    @abstractmethod
    def save_params(self, params, n_episode):
        """
        n_episode: int
        params: dict
        """
        pass

    @abstractmethod
    def write_log(self, log_value):
        pass

    @abstractmethod
    def train(self):
        pass
