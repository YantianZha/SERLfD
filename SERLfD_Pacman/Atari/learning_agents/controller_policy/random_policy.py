import argparse

import gym
import numpy as np
import torch
from torch.distributions import Independent, Normal

from learning_agents.common import common_utils
from learning_agents.rl.trpo.trpo_utils import set_flat_params_to, get_flat_grad_from, get_flat_params_from, \
    normal_log_density
from learning_agents.rl.trpo.trpo import TRPOAgent
from learning_agents.controller_policy.base_policy import BasePolicy
from torch.autograd import Variable
import scipy.optimize
from utils.trajectory_utils import extract_experiences_from_indexed_trajs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# noinspection DuplicatedCode
class RandomPolicy(BasePolicy):
    """
    Random controller_policy that returns random actions
    """

    def __init__(self, env, args, log_cfg, hyper_params, network_cfg, optim_cfg, encoder=None, logger=None):
        """
        Initialize.
        Args:
            env (gym.Env): openAI Gym environment
            args (argparse.Namespace): arguments including hyperparameters and training settings
        """
        BasePolicy.__init__(self, env, args, log_cfg, hyper_params, network_cfg, optim_cfg, logger=logger)
        self.encoder = encoder
        self.used_as_policy = True

        # get state space info
        self.state_dim = self.env.observation_space.shape[0]
        # check if it's single channel or multi channel
        self.state_channel = 1 if len(self.env.observation_space.shape) <= 2 else self.env.observation_space.shape[0]

        # get action space info
        self.is_discrete = False
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.action_dim = self.env.action_space.n
            self.is_discrete = True
        else:
            self.action_dim = self.env.action_space.shape[0]

    # pylint: disable=no-self-use
    def _preprocess_state(self, state):
        """
        Preprocess state so that actor selects an action.
        state: np.ndarray
        """
        state = torch.FloatTensor(state).to(device)
        if self.encoder is not None:
            state = self.encoder(state)
        return state

    def evaluate_states_actions(self, states, actions, info=None):
        """
        get the log probability of actions
        :param: states: np.ndarray
        :param: actions: np.ndarray
        :return: a numpy array of log probability of each action
        """
        return self.evaluate_state_action(states, actions).detach().cpu().numpy()

    def select_action(self, state):
        if self.is_discrete and self.hyper_params.discrete_to_one_hot:
            return np.array(
                common_utils.discrete_action_to_one_hot(self.env.action_space.sample(), self.action_dim))
        else:
            return np.array(self.env.action_space.sample())

    def get_action(self, state, is_test=False, is_random=False, info=None):
        return self.select_action(state)

    def evaluate_state_action(self, state, action, info=None):
        """
        Compute the probability of the state, action pair
        :param action: vector of the action (if discrete, it should be the one-hot representation)
        :return: the log probability (torch.Tensor) of all the state-action pairs
        """
        n_state = state.shape[0]
        log_probs = torch.from_numpy(np.zeros(shape=(n_state, 1), dtype=np.float)).to(device)
        log_probs[:] = np.log(0.05)/n_state
        return log_probs

    def update_policy(self, indexed_sampled_trajs, irl_model):
        """ Train the model after each episode. """
        pass

    def write_policy_log(self, log_value):
        pass

    def get_policy_wandb_log(self, log_value):
        wandb_log = {}
        return wandb_log

    def load_params(self, path):
        """Load model and optimizer parameters."""
        pass

    def save_params(self, n_episode):
        pass

    def get_wandb_watch_list(self):
        return []

    def update_model(self):
        pass

    def write_log(self, log_value):
        pass



