import argparse
import pickle
import time
from collections import deque
from typing import Tuple

import gym
import numpy as np
import torch
import torch.nn.functional as F
import random
import torch.optim as optim

from learning_agents.architectures.cnn import Conv2d_MLP_TanhGaussian, Conv2d_Flatten_MLP
from learning_agents.common.priortized_replay_buffer import PrioritizedReplayBuffer
from learning_agents.common.replay_buffer import ReplayBuffer
from learning_agents.common import common_utils
from learning_agents.architectures.mlp import MLP, FlattenMLP, TanhGaussianDistParams
from learning_agents.utils.utils import ConfigDict
from learning_agents.agent import Agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SACAgent(Agent):
    """ SAC agent interacting with environment.
    Attrtibutes:
        env (gym.Env): openAI Gym environment
        args (argparse.Namespace): arguments including hyperparameters and training settings
        hyper_params (ConfigDict): hyper-parameters
        network_cfg (ConfigDict): config of network for training agent
        optim_cfg (ConfigDict): config of optimizer
        state_dim (int): state size of env
        action_dim (int): action size of env
        memory (ReplayBuffer): replay memory
        actor (nn.Module): actor model to select actions
        actor_target (nn.Module): target actor model to select actions
        actor_optim (Optimizer): optimizer for training actor
        critic_1 (nn.Module): critic model to predict state values
        critic_2 (nn.Module): critic model to predict state values
        critic_target1 (nn.Module): target critic model to predict state values
        critic_target2 (nn.Module): target critic model to predict state values
        critic_optim1 (Optimizer): optimizer for training critic_1
        critic_optim2 (Optimizer): optimizer for training critic_2
        curr_state (np.ndarray): temporary storage of the current state
        total_step (int): total step numbers
        episode_step (int): step number of the current episode
        update_step (int): step number of updates
        i_episode (int): current episode number
        target_entropy (int): desired entropy used for the inequality constraint
        log_alpha (torch.Tensor): weight for entropy
        alpha_optim (Optimizer): optimizer for alpha
    """

    def __init__(self, env, args, log_cfg, hyper_params, network_cfg, optim_cfg, encoder=None, logger=None):
        """
        Initialize.
        Args:
            env (gym.Env): openAI Gym environment
            args (argparse.Namespace): arguments including hyper-parameters and training settings
        """
        Agent.__init__(self, env, args, log_cfg)

        self.total_step = 0
        self.episode_step = 0
        self.update_step = 0
        self.i_episode = 0
        self.logger = logger
        self.encoder = encoder

        self.hyper_params = hyper_params
        self.network_cfg = network_cfg
        self.optim_cfg = optim_cfg

        # get state space info
        self.state_dim = self.env.observation_space.shape[0]
        print('[INFO] state shape: ', self.env.observation_space.shape)
        # check if it's single channel or multi channel
        self.state_channel = self.hyper_params.frame_stack
        print('[INFO] stack state: ', self.state_channel)

        # get action space info
        self.is_discrete = False    # this implementation only supports continuous action space
        self.action_dim = self.env.action_space.shape[0]
        print('[INFO] action dim: ', self.action_dim)

        # target entropy
        target_entropy = -np.prod((self.action_dim, )).item()
        # automatic entropy tuning
        if hyper_params.auto_entropy_tuning:
            self.target_entropy = target_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=optim_cfg.lr_entropy, eps=optim_cfg.eps_entropy)
        else:
            alpha = self.hyper_params.w_entropy
            self.log_alpha = torch.Tensor([np.log(alpha)]).to(device)

        self.per_beta = hyper_params.per_beta
        self.use_n_step = hyper_params.n_step > 1
        self.use_prioritized = hyper_params.use_prioritized

        self._initialize()
        self._init_network()

        self.is_test = self.args.test

    # pylint: disable=attribute-defined-outside-init
    def _initialize(self):
        """ Initialize non-common things."""
        if not self.args.test:
            # replay memory for a single step
            if self.use_prioritized:
                self.memory = PrioritizedReplayBuffer(
                    self.hyper_params.buffer_size,
                    self.hyper_params.batch_size,
                    alpha=self.hyper_params.per_alpha,
                )
            # use ordinary replay buffer
            else:
                self.memory = ReplayBuffer(
                    self.hyper_params.buffer_size,
                    batch_size=self.hyper_params.batch_size,
                    gamma=self.hyper_params.gamma,
                )

            # replay memory for multi-steps
            if self.use_n_step:
                self.memory_n = ReplayBuffer(
                    self.hyper_params.buffer_size,
                    batch_size=self.hyper_params.batch_size,
                    n_step=self.hyper_params.n_step,
                    gamma=self.hyper_params.gamma,
                )

    def _set_to_default_MLP(self):
        # create actor
        self.actor = TanhGaussianDistParams(
            input_size=self.state_dim,
            output_size=self.action_dim,
            hidden_sizes=self.network_cfg.fc_hidden_sizes_actor,
            hidden_activation=self.network_cfg.fc_hidden_activation
        ).to(device)

        # create double q_critic
        # qf 1
        self.qf1 = FlattenMLP(
            input_size=self.state_dim + self.action_dim,
            output_size=1,
            hidden_sizes=self.network_cfg.fc_hidden_sizes_qf,
            hidden_activation=self.network_cfg.fc_hidden_activation
        ).to(device)
        # target qf 1
        self.target_qf1 = FlattenMLP(
            input_size=self.state_dim + self.action_dim,
            output_size=1,
            hidden_sizes=self.network_cfg.fc_hidden_sizes_qf,
            hidden_activation=self.network_cfg.fc_hidden_activation
        ).to(device)
        self.target_qf1.load_state_dict(self.qf1.state_dict())
        # qf 2
        self.qf2 = FlattenMLP(
            input_size=self.state_dim + self.action_dim,
            output_size=1,
            hidden_sizes=self.network_cfg.fc_hidden_sizes_qf,
            hidden_activation=self.network_cfg.fc_hidden_activation
        ).to(device)
        # target qf 2
        self.target_qf2 = FlattenMLP(
            input_size=self.state_dim + self.action_dim,
            output_size=1,
            hidden_sizes=self.network_cfg.fc_hidden_sizes_qf,
            hidden_activation=self.network_cfg.fc_hidden_activation
        ).to(device)
        self.target_qf2.load_state_dict(self.qf2.state_dict())

    def _set_to_default_CNN(self):
        # create actor
        self.actor = Conv2d_MLP_TanhGaussian(input_channels=self.state_channel,
                                             fc_input_size=self.network_cfg.fc_input_size,
                                             fc_output_size=self.action_dim,
                                             nonlinearity=self.network_cfg.nonlinearity,
                                             channels=self.network_cfg.channels,
                                             kernel_sizes=self.network_cfg.kernel_sizes,
                                             strides=self.network_cfg.strides,
                                             paddings=self.network_cfg.paddings,
                                             fc_hidden_sizes=self.network_cfg.fc_hidden_sizes_actor,
                                             fc_hidden_activation=self.network_cfg.fc_hidden_activation).to(device)

        # create double q_critic
        # qf 1
        self.qf1 = Conv2d_Flatten_MLP(input_channels=self.state_channel,
                                      fc_input_size=self.network_cfg.fc_input_size + self.action_dim,
                                      fc_output_size=1,
                                      nonlinearity=self.network_cfg.nonlinearity,
                                      channels=self.network_cfg.channels,
                                      kernel_sizes=self.network_cfg.kernel_sizes,
                                      strides=self.network_cfg.strides,
                                      paddings=self.network_cfg.paddings,
                                      fc_hidden_sizes=self.network_cfg.fc_hidden_sizes_actor,
                                      fc_hidden_activation=self.network_cfg.fc_hidden_activation).to(device)
        # target qf 1
        self.target_qf1 = Conv2d_Flatten_MLP(input_channels=self.state_channel,
                                             fc_input_size=self.network_cfg.fc_input_size + self.action_dim,
                                             fc_output_size=1,
                                             nonlinearity=self.network_cfg.nonlinearity,
                                             channels=self.network_cfg.channels,
                                             kernel_sizes=self.network_cfg.kernel_sizes,
                                             strides=self.network_cfg.strides,
                                             paddings=self.network_cfg.paddings,
                                             fc_hidden_sizes=self.network_cfg.fc_hidden_sizes_actor,
                                             fc_hidden_activation=self.network_cfg.fc_hidden_activation).to(device)
        self.target_qf1.load_state_dict(self.qf1.state_dict())
        # qf 2
        self.qf2 = Conv2d_Flatten_MLP(input_channels=self.state_channel,
                                      fc_input_size=self.network_cfg.fc_input_size + self.action_dim,
                                      fc_output_size=1,
                                      nonlinearity=self.network_cfg.nonlinearity,
                                      channels=self.network_cfg.channels,
                                      kernel_sizes=self.network_cfg.kernel_sizes,
                                      strides=self.network_cfg.strides,
                                      paddings=self.network_cfg.paddings,
                                      fc_hidden_sizes=self.network_cfg.fc_hidden_sizes_actor,
                                      fc_hidden_activation=self.network_cfg.fc_hidden_activation).to(device)
        # target qf 2
        self.target_qf2 = Conv2d_Flatten_MLP(input_channels=self.state_channel,
                                             fc_input_size=self.network_cfg.fc_input_size + self.action_dim,
                                             fc_output_size=1,
                                             nonlinearity=self.network_cfg.nonlinearity,
                                             channels=self.network_cfg.channels,
                                             kernel_sizes=self.network_cfg.kernel_sizes,
                                             strides=self.network_cfg.strides,
                                             paddings=self.network_cfg.paddings,
                                             fc_hidden_sizes=self.network_cfg.fc_hidden_sizes_actor,
                                             fc_hidden_activation=self.network_cfg.fc_hidden_activation).to(device)
        self.target_qf2.load_state_dict(self.qf2.state_dict())

    # pylint: disable=attribute-defined-outside-init
    def _init_network(self):
        """ Initialize networks and optimizers. """
        if self.network_cfg.use_cnn:
            self._set_to_default_CNN()
        else:
            self._set_to_default_MLP()

        # create optimizers
        self.actor_optim = optim.Adam(
            self.actor.parameters(),
            lr=self.optim_cfg.lr_actor,
            eps=self.optim_cfg.eps_actor,
            weight_decay=self.optim_cfg.weight_decay,
        )

        self.qf1_optim = optim.Adam(
            self.qf1.parameters(),
            lr=self.optim_cfg.lr_qf,
            eps=self.optim_cfg.eps_qf,
            weight_decay=self.optim_cfg.weight_decay,
        )
        self.qf2_optim = optim.Adam(
            self.qf2.parameters(),
            lr=self.optim_cfg.lr_qf,
            eps=self.optim_cfg.eps_qf,
            weight_decay=self.optim_cfg.weight_decay,
        )

        # init network from file
        self._init_from_file()

    def _init_from_file(self):
        # load the optimizer and model parameters
        if self.args.load_from is not None:
            self.load_params(self.args.load_from)

    # pylint: disable=no-self-use
    def _preprocess_state(self, state):
        """ Preprocess state so that actor selects an action. """
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(device)
        # if state is a single state, we unsqueeze it
        if len(state.size()) == 3 or len(state.size()) == 1:
            state = state.unsqueeze(0)
        if self.encoder is not None:
            state = self.encoder(state)
        return state

    def select_action(self, state):
        """
        Select an action from the input space.
        state: np.ndarray
        """
        # if initial random action should be conducted
        if self.total_step < self.hyper_params.init_random_actions and not self.is_test:
            selected_action = np.array(self.env.action_space.sample())
        elif self.is_test or self.hyper_params.deterministic_action:
            state = self._preprocess_state(state)
            self.actor.eval()
            with torch.no_grad():
                # actor returns squashed action, log_prob, z, mean, std
                _, _, _, mean, _ = self.actor(state)
                selected_action = torch.tanh(mean)
            self.actor.train()
            selected_action = selected_action.squeeze(0).detach().cpu().numpy()
        else:
            state = self._preprocess_state(state)
            self.actor.eval()
            with torch.no_grad():
                # actor returns squashed action, log_prob, z, mean, std
                selected_action, _, _, _, _ = self.actor(state)
            self.actor.train()
            selected_action = selected_action.squeeze(0).detach().cpu().numpy()
        return selected_action

    def step(self, action):
        """
        Take an action and return the response of the env.
        action: np.ndarray
        """
        next_state, reward, done, info = self.env.step(action)

        # if the last state is not a terminal state, store done as false
        done = (
            True if self.episode_step == self.hyper_params.max_traj_length else done
        )
        return next_state, reward, done, info

    def _add_transition_to_memory(self, transition):
        """ Add 1 step and n step transitions to memory. """
        # add n-step transition
        if self.use_n_step:
            transition = self.memory_n.add(transition)

        # add a single step transition
        # if transition is not an empty tuple
        if transition:
            self.memory.add(transition)

    def _get_qf_loss(self, obs, actions, rewards, next_obs, dones, alpha, gamma):
        masks = 1 - dones
        with torch.no_grad():
            next_new_actions, next_log_probs, _, _, _ = self.actor(next_obs)
            soft_v_targets = torch.min(
                self.target_qf1(next_obs, next_new_actions),
                self.target_qf2(next_obs, next_new_actions),
            ) - alpha * next_log_probs

            soft_q_targets = self.hyper_params.reward_scale * rewards + gamma * masks * soft_v_targets

        q1_prediction = self.qf1(obs, actions)
        q2_prediction = self.qf2(obs, actions)

        q1_loss_element_wise = F.mse_loss(q1_prediction, soft_q_targets.detach(), reduction="none")
        q2_loss_element_wise = F.mse_loss(q2_prediction, soft_q_targets.detach(), reduction="none")

        return q1_loss_element_wise, q2_loss_element_wise, q1_prediction, q2_prediction

    # noinspection DuplicatedCode
    def update_model(self):
        """ Train the model after each episode. """
        self.update_step += 1

        # 1 step loss
        if self.use_prioritized:
            experiences_one_step = self.memory.sample(self.per_beta)
            weights, indices = experiences_one_step[-3:-1]
            indices = np.array(indices)
            # re-normalize the weights such that they sum up to the value of batch_size
            weights = weights / torch.sum(weights) * float(self.hyper_params.batch_size)
        else:
            indices = np.random.choice(len(self.memory), size=self.hyper_params.batch_size, replace=False)
            weights = torch.from_numpy(np.ones(shape=(indices.shape[0], 1), dtype=np.float64)).type(
                torch.FloatTensor).to(device)
            experiences_one_step = self.memory.sample(indices=indices)

        obs, actions, rewards, next_obs, dones = experiences_one_step[:5]
        if self.hyper_params.reward_clip is not None:
            rewards = torch.clamp(rewards, min=self.hyper_params.reward_clip[0],
                                  max=self.hyper_params.reward_clip[1])
        curr_actions, log_probs, predicted_tanh_values, means, stds = self.actor(obs)

        ###################
        ### Train Alpha ###
        ###################
        if self.hyper_params.auto_entropy_tuning:
            # alpha loss (equation 17 in the paper)
            alpha_loss = (
                    (-self.log_alpha * ((log_probs + self.target_entropy).detach())) * weights
            ).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.zeros(1)
            alpha = self.hyper_params.w_entropy

        ###################
        ### Policy Loss ###
        ###################
        new_actions_q_values = torch.min(
            self.qf1(obs, curr_actions),
            self.qf2(obs, curr_actions)
        )
        # actor loss (equation 7 and 9 in the paper)
        actor_loss = (alpha * log_probs - new_actions_q_values).mean()

        # regularization
        if not self.is_discrete:  # iff the action is continuous
            mean_reg = self.hyper_params.w_mean_reg * means.pow(2).mean()
            std_reg = self.hyper_params.w_std_reg * stds.pow(2).mean()
            pre_activation_reg = self.hyper_params.w_pre_activation_reg * (
                predicted_tanh_values.pow(2).sum(dim=-1).mean()
            )
            actor_reg = mean_reg + std_reg + pre_activation_reg
            # actor loss + regularization
            actor_loss += actor_reg

        ###############################
        ### Q functions 1-Step Loss ###
        ###############################
        loss_info = self._get_qf_loss(obs=obs, actions=actions, rewards=rewards, next_obs=next_obs, dones=dones,
                                      alpha=alpha, gamma=self.hyper_params.gamma)
        q1_loss_element_wise, q2_loss_element_wise, q1_values, q2_values = loss_info
        q1_loss = torch.mean(q1_loss_element_wise * weights)
        q2_loss = torch.mean(q2_loss_element_wise * weights)

        ###############################
        ### Q functions n-Step Loss ###
        ###############################
        if self.use_n_step:
            experiences_n = self.memory_n.sample(indices)
            obs_n, actions_n, rewards_n, next_obs_n, dones_n = experiences_n
            if self.hyper_params.reward_clip is not None:
                rewards_n = torch.clamp(rewards_n, min=self.hyper_params.reward_clip[0],
                                        max=self.hyper_params.reward_clip[1])
            gamma_n = self.hyper_params.gamma ** self.hyper_params.n_step

            loss_info_n = self._get_qf_loss(obs=obs_n, actions=actions_n, rewards=rewards_n, next_obs=next_obs_n,
                                            dones=dones_n, alpha=alpha, gamma=gamma_n)
            q1_loss_element_wise_n, q2_loss_element_wise_n, q1_values_n, q2_values_n = loss_info_n
            q1_values = 0.5 * (q1_values + q1_values_n)
            q2_values = 0.5 * (q2_values + q2_values_n)

            # mix of 1-step and n-step returns
            q1_loss_element_wise += q1_loss_element_wise_n * self.hyper_params.w_n_step
            q2_loss_element_wise += q2_loss_element_wise_n * self.hyper_params.w_n_step
            q1_loss = torch.mean(q1_loss_element_wise * weights)
            q2_loss = torch.mean(q2_loss_element_wise * weights)

        #######################
        ### Update Networks ###
        #######################
        # train Q functions
        self.qf1_optim.zero_grad()
        q1_loss.backward()
        self.qf1_optim.step()

        self.qf2_optim.zero_grad()
        q2_loss.backward()
        self.qf2_optim.step()

        # train actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self.update_step % self.hyper_params.target_update_freq == 0:
            # soft update target network
            common_utils.soft_update(self.qf1, self.target_qf1, self.hyper_params.tau)
            common_utils.soft_update(self.qf2, self.target_qf2, self.hyper_params.tau)

        #########################
        ### Update Priorities ###
        #########################
        # update priorities in PER
        if self.use_prioritized:
            qf_loss_element_wise = 0.5 * (q1_loss_element_wise + q2_loss_element_wise)
            loss_for_prior = qf_loss_element_wise.detach().cpu().numpy().squeeze()
            new_priorities = loss_for_prior + self.hyper_params.per_eps
            # noinspection PyUnresolvedReferences
            self.memory.update_priorities(indices, new_priorities)

            # increase beta
            fraction = min(float(self.i_episode) / self.args.iteration_num, 1.0)
            self.per_beta = self.per_beta + fraction * (1.0 - self.per_beta)

        return actor_loss.item(), q1_loss.item(), q2_loss.item(), alpha_loss.item(), q1_values.mean().item(), q2_values.mean().item()

    def load_params(self, path):
        """Load model and optimizer parameters."""
        Agent.load_params(self, path)

        params = torch.load(path, map_location=device)
        self.actor.load_state_dict(params["actor"])
        self.qf1.load_state_dict(params["qf1"])
        self.qf2.load_state_dict(params["qf2"])
        self.target_qf1.load_state_dict(params["target_qf1"])
        self.target_qf2.load_state_dict(params["target_qf2"])
        self.actor_optim.load_state_dict(params["actor_optim"])
        self.qf1_optim.load_state_dict(params["qf1_optim"])
        self.qf2_optim.load_state_dict(params["qf2_optim"])

        if self.hyper_params.auto_entropy_tuning:
            self.alpha_optim.load_state_dict(params["alpha_optim"])

        print("[INFO] loaded the model and optimizer from", path)

    def save_params(self, n_step):  # type: ignore
        """Save model and optimizer parameters."""
        params = {
            "actor": self.actor.state_dict(),
            "qf1": self.qf1.state_dict(),
            "qf2": self.qf2.state_dict(),
            "target_qf1": self.target_qf1.state_dict(),
            "target_qf2": self.target_qf2.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "qf1_optim": self.qf1_optim.state_dict(),
            "qf2_optim": self.qf2_optim.state_dict()
        }

        if self.hyper_params.auto_entropy_tuning:
            params["alpha_optim"] = self.alpha_optim.state_dict()

        Agent.save_params(self, params, n_step)
        if self.logger is not None:
            self.logger.save_models(params, postfix=str(n_step), is_snapshot=True)

    def write_log(self, log_value):
        """
        Write log about loss and score
        log_value should be in the form of i_episode, avg_loss, score, avg_time_cost, avg_scores_window
        """
        i_episode, avg_loss, score, avg_time_cost, avg_scores_window, alpha_value = log_value

        print(
            "[INFO] episode %d, episode_step %d, total step %d, total score: %d\n"
            "actor_loss: %.3f, qf1_loss: %.3f, qf2_loss: %.3f\n"
            "alpha_loss: %.3f, alpha: %.3f, avg window score: %.3f\n"
            "avg qf 1: %.3f, avg qf 2: %.3f (spent %.6f sec/step)\n"
            % (
                i_episode,
                self.episode_step,
                self.total_step,
                score,
                avg_loss[0],  # actor loss
                avg_loss[1],  # qf1 loss
                avg_loss[2],  # qf2 loss
                avg_loss[3],  # alpha loss
                alpha_value,
                avg_scores_window,
                avg_loss[4],
                avg_loss[5],
                avg_time_cost,
            )
        )

        if self.logger is not None:
            self.logger.log_wandb({
                'score': score,
                "episode": self.i_episode,
                "episode step": self.episode_step,
                "total step": self.total_step,
                "alpha": alpha_value,
                'actor loss': avg_loss[0],
                'qf1 loss': avg_loss[1],
                'qf2 loss': avg_loss[2],
                'alpha loss': avg_loss[3],
                'time per each step': avg_time_cost,
                'avg score window': avg_scores_window,
                'avg q1 values': avg_loss[4],
                'avg q2 values': avg_loss[5],
                'avg q values': 0.5 * (avg_loss[4] + avg_loss[5])
            }, step=self.total_step)

    # pylint: disable=no-self-use, unnecessary-pass
    def pretrain(self):
        """ Pretraining steps."""
        pass

    def train(self):
        """ Train the agent."""
        if self.logger is not None:
            self.logger.watch_wandb([self.actor, self.qf1, self.target_qf1, self.qf2, self.target_qf2])

        # pre-training if needed
        self.pretrain()

        use_cnn = self.network_cfg.use_cnn
        avg_scores_window = deque(maxlen=self.args.avg_score_window)
        eval_scores_window = deque(maxlen=self.args.eval_score_window)

        for i_episode in range(1, self.args.iteration_num + 1):
            self.i_episode = i_episode
            self.testing = (self.i_episode % self.args.eval_period == 0)

            state = np.squeeze(self.env.reset(), axis=0) if use_cnn else self.env.reset()

            self.episode_step = 0
            done = False
            score = 0
            losses = list()

            # stack states
            states_queue = deque(maxlen=self.hyper_params.frame_stack)
            states_queue.extend([state for _ in range(self.hyper_params.frame_stack)])

            t_begin = time.time()

            while not done:
                if self.args.render \
                        and self.i_episode >= self.args.render_after \
                        and self.i_episode % self.args.render_freq == 0:
                    self.env.render()

                stacked_states = np.copy(np.stack(list(states_queue), axis=0)) if use_cnn else state
                action = self.select_action(stacked_states)
                next_state, reward, done, _ = self.step(action)
                # add next state into states queue
                next_state = np.squeeze(next_state, axis=0) if use_cnn else next_state

                states_queue.append(next_state)
                stacked_next_states = np.copy(np.stack(list(states_queue), axis=0)) if use_cnn else next_state
                # save the new transition
                transition = (stacked_states, action, reward, stacked_next_states, done)
                self._add_transition_to_memory(transition)

                self.total_step += 1
                self.episode_step += 1

                state = next_state
                score += reward

                # training
                if len(self.memory) >= self.hyper_params.update_starts_from:
                    if self.total_step % self.hyper_params.train_freq == 0:
                        for _ in range(self.hyper_params.multiple_update):
                            loss = self.update_model()
                            losses.append(loss)  # for logging

            self.do_post_episode_update()
            t_end = time.time()
            avg_time_cost = (t_end - t_begin) / self.episode_step
            avg_scores_window.append(score)

            if self.testing:
                eval_scores_window.append(score)
                # noinspection PyStringFormat
                print('[EVAL INFO] episode: %d, total step %d, '
                      'evaluation score: %.3f, window avg: %.3f\n'
                      % (self.i_episode,
                         self.total_step,
                         score,
                         np.mean(eval_scores_window)))

                if self.logger is not None:
                    self.logger.log_wandb({
                        'eval scores': score,
                        "eval window avg": np.mean(eval_scores_window),
                    }, step=self.total_step)

                self.testing = False

            # logging
            if losses:
                alpha_value = self.log_alpha.exp().item() if self.hyper_params.auto_entropy_tuning else self.hyper_params.w_entropy
                avg_loss = np.vstack(losses).mean(axis=0)
                log_value = (
                    self.i_episode,
                    avg_loss,
                    score,
                    avg_time_cost,
                    np.mean(avg_scores_window),
                    alpha_value
                )
                self.write_log(log_value)

            if self.i_episode % self.args.save_period == 0:
                self.save_params(self.total_step)

        # termination
        self.env.close()
        self.save_params(self.i_episode)

    def test(self, n_episode=1e3, verbose=True):
        use_cnn = self.network_cfg.use_cnn
        scores = []
        self.testing = True

        for i_episode in range(int(n_episode)):
            state = np.squeeze(self.env.reset(), axis=0) if use_cnn else self.env.reset()
            done = False
            score = 0
            self.episode_step = 0

            # stack states
            states_queue = deque(maxlen=self.hyper_params.frame_stack)
            states_queue.extend([state for _ in range(self.hyper_params.frame_stack)])

            while not done:
                self.env.render()

                stacked_states = np.copy(np.stack(list(states_queue), axis=0)) if use_cnn else state
                action = self.select_action(stacked_states)
                next_state, reward, done, _ = self.step(action)
                # add next state into states queue
                next_state = np.squeeze(next_state, axis=0) if use_cnn else next_state
                states_queue.append(next_state)

                state = next_state
                score += reward
                self.episode_step += 1
                self.total_step += 1
            if verbose:
                print('[EVAL INFO] evaluation episode: %d, episode step: %d, score: %f'
                      % (i_episode, self.episode_step, score))
            scores.append(score)

        # termination
        self.env.close()
        return scores

    def do_post_episode_update(self, *argv):
        pass
