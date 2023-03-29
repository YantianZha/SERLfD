from abc import ABC, abstractmethod
from enum import Enum
from learning_agents.agent import Agent

import gym
import numpy as np
import torch

from learning_agents.common.noise import OUNoise
from learning_agents.controller_policy.base_policy import BasePolicy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SEGL_BasePolicy(BasePolicy, ABC):

    def __init__(self, env, args, log_cfg, hyper_params, network_cfg, optim_cfg, logger=None):
        BasePolicy.__init__(self, env, args, log_cfg, hyper_params, network_cfg, optim_cfg, logger=logger)

    def _to_float_tensor(self, np_data):
        tensor = torch.FloatTensor(np_data).to(device)
        if torch.cuda.is_available():
            tensor = tensor.cuda(non_blocking=True)
        return tensor

    def augment_experience(self, explainer, experiences, predicates, next_predicates, is_to_tensor=True):
        states, _, rewards, next_states, _ = experiences[:5]
        assert len(states) == len(predicates), 'the number of states and predicates should be the same'

        augmented_states = explainer.augment_states(states, predicates)
        augmented_next_states = explainer.augment_states(next_states, next_predicates)
        shaping_rewards = explainer.get_shaping_reward(states, predicates, next_states, next_predicates)
        augmented_rewards = shaping_rewards + rewards

        if is_to_tensor:
            augmented_states = self._to_float_tensor(augmented_states)
            augmented_next_states = self._to_float_tensor(augmented_next_states)
            augmented_rewards = self._to_float_tensor(augmented_rewards)
        return augmented_states, augmented_rewards, augmented_next_states

    def write_policy_wandb_log(self, log_value, step=None):
        if self.logger is not None:
            if step is not None:
                self.logger.log_wandb(log_value, step=step)
            else:
                self.logger.log_wandb(log_value)
