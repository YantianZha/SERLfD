from abc import ABC, abstractmethod
import argparse

import gym


class Encoder(ABC):
    """
    Abstract Encoder.
    Attributes:
        env (gym.Env): openAI Gym environment
        state_dim (int): dimension of states
        action_dim (int): dimension of actions
    """

    def __init__(self, env, args):
        """Initialize."""
        self.args = args
        self.env = env

        self.original_state_dim = self.env.observation_space.shape[0]
        self.original_state_channel = 1 if len(self.env.observation_space.shape) == 2 else self.env.observation_space.shape[0]

        self.output_dim = self.args.output_dim
        self.output_channel = self.args.output_channel

        if args.load_path is not None:
            self.load_params(args.load_path)

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
    def train(self):
        pass