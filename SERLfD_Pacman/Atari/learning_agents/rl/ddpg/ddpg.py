import argparse
import time
from typing import Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym

from learning_agents.agent import Agent
from learning_agents.common.replay_buffer import ReplayBuffer
import learning_agents.common.common_utils as common_utils
from learning_agents.architectures.mlp import MLP, FlattenMLP
from learning_agents.rl.ddpg.ddpg_models import DefaultActorCNN, DefaultCriticCNN
from learning_agents.common.noise import OUNoise
from learning_agents.utils.utils import ConfigDict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPGAgent(Agent):
    """
    ActorCritic interacting with environment.
    Attributes:
        env (gym.Env): openAI Gym environment
        args (argparse.Namespace): arguments including hyperparameters and training settings
        hyper_params (ConfigDict): hyper-parameters
        network_cfg (ConfigDict): config of network for training agent
        optim_cfg (ConfigDict): config of optimizer
        state_dim (int): state size of env
        state_channel (int): number of channels of the state
        action_dim (int): action size of env
        memory (ReplayBuffer): replay memory
        noise (OUNoise): random noise for exploration
        actor (nn.Module): actor model to select actions
        actor_target (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        critic_target (nn.Module): target critic model to predict state values
        actor_optim (Optimizer): optimizer for training actor
        critic_optim (Optimizer): optimizer for training critic
        curr_state (np.ndarray): temporary storage of the current state
        total_step (int): total step numbers
        episode_step (int): step number of the current episode
        i_episode (int): current episode number
    """

    def __init__(self, env, args, log_cfg, hyper_params, network_cfg, optim_cfg, noise_cfg, logger=None):
        """
        Initialize.
        env: gym.Env,
        args: argparse.Namespace,
        log_cfg: ConfigDict,
        hyper_params: ConfigDict,
        network_cfg: ConfigDict,
        optim_cfg: ConfigDict,
        noise_cfg: ConfigDict,
        """
        Agent.__init__(self, env, args, log_cfg)

        self.curr_state = np.zeros((1,))
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

        # set noise
        self.noise = OUNoise(
            self.action_dim,
            theta=noise_cfg.ou_noise_theta,
            sigma=noise_cfg.ou_noise_sigma,
        )

        self._initialize()
        self._init_network()

    # pylint: disable=attribute-defined-outside-init
    def _initialize(self):
        """Initialize non-common things."""
        if not self.args.test:
            # replay memory
            self.memory = ReplayBuffer(
                self.hyper_params.buffer_size, self.hyper_params.batch_size
            )

    def _set_to_default_MLP(self):
        # create actor
        self.actor = MLP(
            input_size=self.state_dim,
            output_size=self.action_dim,
            hidden_sizes=self.network_cfg.hidden_sizes_actor,
            output_activation=torch.tanh,
        ).to(device)

        self.actor_target = MLP(
            input_size=self.state_dim,
            output_size=self.action_dim,
            hidden_sizes=self.network_cfg.hidden_sizes_actor,
            output_activation=torch.tanh,
        ).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # create critic
        self.critic = FlattenMLP(
            input_size=self.state_dim + self.action_dim,
            output_size=1,
            hidden_sizes=self.network_cfg.hidden_sizes_critic,
        ).to(device)

        self.critic_target = FlattenMLP(
            input_size=self.state_dim + self.action_dim,
            output_size=1,
            hidden_sizes=self.network_cfg.hidden_sizes_critic,
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

    def _set_to_default_CNN(self):
        self.actor = DefaultActorCNN(
            input_channel=self.state_channel,
            output_size=self.action_dim
        ).to(device)

        self.actor_target = DefaultActorCNN(
            input_channel=self.state_channel,
            output_size=self.action_dim
        ).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # create critic
        self.critic = DefaultCriticCNN(
            input_channel=self.state_channel,
            output_size=1
        ).to(device)

        self.critic_target = DefaultCriticCNN(
            input_channel=self.state_channel,
            output_size=1
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

    # pylint: disable=attribute-defined-outside-init
    def _init_network(self):
        """Initialize networks and optimizers."""
        self._set_to_default_MLP()

        # create optimizer
        self.actor_optim = optim.Adam(
            self.actor.parameters(),
            lr=self.optim_cfg.lr_actor,
            weight_decay=self.optim_cfg.weight_decay,
        )

        self.critic_optim = optim.Adam(
            self.critic.parameters(),
            lr=self.optim_cfg.lr_critic,
            weight_decay=self.optim_cfg.weight_decay,
        )

        # load the optimizer and model parameters
        if self.args.load_from is not None:
            self.load_params(self.args.load_from)

    def select_action(self, state):
        """Select an action from the input space."""
        self.curr_state = state
        state = self._preprocess_state(state)
        # if initial random action should be conducted
        if self.total_step < self.hyper_params.initial_random_action and not self.args.test:
            if self.is_discrete:
                return np.array(common_utils.discrete_action_to_one_hot(self.env.action_space.sample(), self.action_dim))
            else:
                # the sample() should return a 1-dimension vector
                return np.array(self.env.action_space.sample())

        # the actor returns a nested list [[one-hot action]]
        selected_action = self.actor(state).detach().cpu().numpy()

        if not self.args.test:
            noise = self.noise.sample()
            selected_action = np.clip(selected_action + noise, -1.0, 1.0)

        # the selected_action is in the format of [[one hot vectors]]
        return selected_action

    # pylint: disable=no-self-use
    def _preprocess_state(self, state):
        """Preprocess state so that actor selects an action."""
        state = torch.FloatTensor(state).to(device)
        return state

    def step(self, action):
        """
        Take an action and return the response of the env.
        Tuple[np.ndarray, np.float64, bool, dict]
        action: np.ndarray (for discrete action, it's a one-hot representation)
        """
        if self.is_discrete:
            action_id = common_utils.one_hot_to_discrete_action(action)
            next_state, reward, done, info = self.env.step(action_id)
        else:
            next_state, reward, done, info = self.env.step(action)

        if not self.args.test:
            # if the last state is not a terminal state, store done as false
            done_bool = (
                False if self.episode_step == self.args.max_episode_steps else done
            )
            transition = (self.curr_state, action, reward, next_state, done_bool)
            self._add_transition_to_memory(transition)

        return next_state, reward, done, info

    def _add_transition_to_memory(self, transition):
        """
        Add 1 step and n step transitions to memory.
        transition: Tuple[np.ndarray, ...]
        """
        self.memory.add(transition)

    def update_model(self):
        """Train the model after each episode."""
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences

        masks = 1 - dones
        next_actions = self.actor_target(next_states)
        next_values = self.critic_target((next_states, next_actions))
        curr_returns = rewards + self.hyper_params.gamma * next_values * masks
        curr_returns = curr_returns.to(device)

        # train critic
        gradient_clip_ac = self.hyper_params.gradient_clip_ac
        gradient_clip_cr = self.hyper_params.gradient_clip_cr

        values = self.critic((states, actions))
        critic_loss = F.mse_loss(values, curr_returns)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), gradient_clip_cr)
        self.critic_optim.step()

        # train actor
        actions = self.actor(states)
        actor_loss = -self.critic((states, actions)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), gradient_clip_ac)
        self.actor_optim.step()

        # update target networks
        common_utils.soft_update(self.actor, self.actor_target, self.hyper_params.tau)
        common_utils.soft_update(self.critic, self.critic_target, self.hyper_params.tau)

        return actor_loss.item(), critic_loss.item()

    def load_params(self, path):
        """Load model and optimizer parameters."""
        Agent.load_params(self, path)

        params = torch.load(path)
        self.actor.load_state_dict(params["actor_state_dict"])
        self.actor_target.load_state_dict(params["actor_target_state_dict"])
        self.critic.load_state_dict(params["critic_state_dict"])
        self.critic_target.load_state_dict(params["critic_target_state_dict"])
        self.actor_optim.load_state_dict(params["actor_optim_state_dict"])
        self.critic_optim.load_state_dict(params["critic_optim_state_dict"])
        print("[INFO] loaded the model and optimizer from", path)

    # noinspection PyMethodOverriding
    def save_params(self, n_episode):
        """Save model and optimizer parameters."""
        params = {
            "actor_state_dict": self.actor.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_optim_state_dict": self.actor_optim.state_dict(),
            "critic_optim_state_dict": self.critic_optim.state_dict(),
            "epoch": n_episode
        }
        if self.logger:
            self.logger.save_models(params, is_snapshot=True)
            self.logger.save_trajectories_snapshot(is_save_utility=True)
        Agent.save_params(self, params, n_episode)

    def write_log(self, log_value):
        """
        Write log about loss and score
        """
        i, loss, score, avg_time_cost = log_value
        total_loss = loss.sum()

        print(
            "[INFO] episode %d, episode step: %d, total step: %d, total score: %d\n"
            "total loss: %f actor_loss: %.3f critic_loss: %.3f (spent %.6f sec/step)\n"
            % (
                i,
                self.episode_step,
                self.total_step,
                score,
                total_loss,
                loss[0],
                loss[1],
                avg_time_cost,
            )
        )

        if self.logger is not None:
            self.logger.log_wandb({
                'score': score,
                'total loss': total_loss,
                'actor loss': loss[0],
                'critic loss': loss[1],
                'time per each step': avg_time_cost,
            })

    # pylint: disable=no-self-use, unnecessary-pass
    def pretrain(self):
        """ Pre-training steps."""
        pass

    def train(self):
        """Train the agent."""
        if self.logger is not None:
            self.logger.watch_wandb([self.actor, self.critic])

        # pre-training if needed
        self.pretrain()

        for episode_idx in range(1, self.args.episode_num + 1):
            self.i_episode = episode_idx
            state = self.env.reset()
            assert isinstance(state, np.ndarray), 'state is not a np.array'
            done = False
            score = 0
            self.episode_step = 0
            losses = list()

            t_begin = time.time()

            while not done:
                if self.args.render and self.i_episode >= self.args.render_after:
                    self.env.render()

                # get current predicate
                predicate_values = {}
                if 'get_current_predicate' in dir(self.env):
                    predicate_values = self.env.get_current_predicate()

                action = self.select_action(state)
                next_state, reward, done, info = self.step(action)

                # logging
                if self.logger:
                    action_taken = action
                    if self.is_discrete:
                        action_taken = common_utils.one_hot_to_discrete_action(action)
                    # log transition
                    self.logger.add_transition(state, action_taken, reward, next_state, done,
                                               is_save_utility=False, predicate_values=predicate_values,
                                               utility_map=None, utility_values=None)
                    # save the new image observation
                    # new_obs_img = info['rgb_img']
                    # save the new observation image
                    # self.logger.save_rgb_image('obs', new_obs_img, step=self.episode_step, episode=self.i_episode)

                self.total_step += 1
                self.episode_step += 1

                # update model and log the loss
                if len(self.memory) >= self.hyper_params.batch_size:
                    for _ in range(self.hyper_params.multiple_update):
                        loss = self.update_model()
                        # for logging
                        losses.append(loss)

                state = next_state
                score += reward

            t_end = time.time()
            avg_time_cost = (t_end - t_begin) / self.episode_step

            # logging episode information
            if losses:
                avg_loss = np.vstack(losses).mean(axis=0)
                log_value = (self.i_episode, avg_loss, score, avg_time_cost)
                self.write_log(log_value)
                losses.clear()

            if self.i_episode % self.args.save_period == 0:
                self.save_params(self.i_episode)

        # termination
        self.env.close()
        self.save_params(self.i_episode)
