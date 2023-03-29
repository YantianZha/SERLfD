import argparse

import gym
import numpy as np
import torch
from torch.distributions import Independent, Normal
import torch.nn.functional as F
import torch.optim as optim

from learning_agents.common import common_utils
from utils import trajectory_utils
from learning_agents.rl.sac.sac import SACAgent
from learning_agents.controller_policy.base_policy import BasePolicy
from learning_agents.controller_policy.base_policy import TRAJECTORY_INFO_INDEX
from utils.trajectory_utils import TRAJECTORY_INDEX, extract_experiences_from_indexed_trajs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# noinspection DuplicatedCode
class SACPolicy(BasePolicy, SACAgent):
    """SAC agent interacting with environment.
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
            args (argparse.Namespace): arguments including hyperparameters and training settings
        """
        BasePolicy.__init__(self, env, args, log_cfg, hyper_params, network_cfg, optim_cfg, logger=logger)
        SACAgent.__init__(self, env, args, log_cfg, hyper_params, network_cfg, optim_cfg, encoder=encoder, logger=logger)

    def evaluate_states_actions(self, states, actions, info=None):
        """
        get the log probability of actions
        :param: states: np.ndarray
        :param: actions: np.ndarray
        :return: a numpy array of log probability of each action
        """
        if not isinstance(actions, torch.Tensor):
            if not isinstance(actions, np.ndarray):
                actions = torch.Tensor(actions).type(torch.FloatTensor).to(device)
            else:
                actions = torch.from_numpy(actions).type(torch.FloatTensor).to(device)
        states = self._preprocess_state(states)
        _, _, _, means, stds = self.actor(states)
        dist = Independent(Normal(means, stds), 1)
        return dist.log_prob(actions).detach().cpu().numpy()

    def evaluate_state_action(self, state, action, info=None):
        """
        Compute the probability of the state, action pair
        :param action: vector of the action (if discrete, it should be the one-hot representation)
        :return: the log probability (torch.Tensor) of the state, action pair
        """
        if not isinstance(action, torch.Tensor):
            if not isinstance(action, np.ndarray):
                action = torch.Tensor(action).type(torch.FloatTensor).to(device)
            else:
                action = torch.from_numpy(action).type(torch.FloatTensor).to(device)
        state = self._preprocess_state(state)
        _, _, _, mean, std = self.actor(state)

        dist = Independent(Normal(mean, std), 1)
        return dist.log_prob(action)

    def get_reward(self, state, action):
        return self.evaluate_state_action(state, action).detach().cpu().numpy()

    def update_policy(self, indexed_sampled_trajs, irl_model):
        """ Train the model after each episode. """
        actor_losses = []
        q_1_losses = []
        q_2_losses = []
        vf_losses = []
        alpha_losses = []

        # sample batch_size
        n_sampled_trajs = len(irl_model.sampled_traj_buf)
        # if use past trajectories, calculate how many past trajectories it uses
        if self.hyper_params.is_fusion:
            past_batch_size = min(n_sampled_trajs,
                                  max(0, self.hyper_params.traj_batch_size - len(indexed_sampled_trajs)))
        else:
            past_batch_size = 0

        for it in range(self.hyper_params.num_iteration_update):
            # sampled sample trajectories
            # if traj_batch_size is negative, use all sampled trajectories
            if self.hyper_params.traj_batch_size < 0:
                sample_indices = np.random.randint(low=0, high=n_sampled_trajs, size=n_sampled_trajs)
                # use irl model's sampled_traj_buf, so we don't need two copies of trajectories
                states, actions, rewards, next_states, dones = irl_model.sampled_traj_buf.get_experiences_from_trajs(
                    indices=sample_indices)
            # if it doesn't use all past-sampled trajectories
            else:
                past_trajs = []
                if past_batch_size > 0:
                    past_trajs = irl_model.sampled_traj_buf.sample(batch_size=past_batch_size)
                sampled_trajs = indexed_sampled_trajs + past_trajs
                states, actions, rewards, next_states, dones = extract_experiences_from_indexed_trajs(sampled_trajs)
            # the mask: shape=(n_states, )
            mask = np.array(1 - dones, dtype=np.float)
            # let irl module re-evaluate the state-action pairs (rewards): shape=(n_states, )
            rewards = irl_model.get_reward(states, actions)

            n_states = states.shape[0]
            arrange = np.arange(n_states)
            batch_size = self.hyper_params.batch_size
            for i in range(n_states // batch_size):
                start_idx = batch_size * i
                end_idx = batch_size * (i + 1)
                batch_index = arrange[start_idx:end_idx]
                states_batch = self._preprocess_state(states[batch_index])
                actions_batch = torch.FloatTensor(torch.from_numpy(actions[batch_index])).to(device)
                rewards_batch = torch.FloatTensor(torch.from_numpy(rewards[batch_index])).unsqueeze(1).to(device)
                mask_batch = torch.from_numpy(mask[batch_index]).type(torch.FloatTensor).unsqueeze(1).to(device)
                next_states_batch = self._preprocess_state(next_states[batch_index])

                new_actions, log_probs, predicted_tanh_values, means, stds = self.actor(states_batch)
                # train alpha
                if self.hyper_params.auto_entropy_tuning:
                    alpha_loss = (
                            -self.log_alpha * (log_probs + self.target_entropy).detach()
                    ).mean()

                    self.alpha_optim.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optim.step()

                    alpha = self.log_alpha.exp()
                else:
                    alpha_loss = torch.zeros(1)
                    alpha = self.hyper_params.w_entropy

                # Q function loss
                q_1_prediction = self.qf1((states_batch, actions_batch))
                q_2_prediction = self.qf2((states_batch, actions_batch))
                v_target = self.vf_target(next_states_batch)
                q_target = rewards_batch + self.hyper_params.gamma * v_target * mask_batch
                q_1_loss = F.mse_loss(q_1_prediction, q_target.detach())
                q_2_loss = F.mse_loss(q_2_prediction, q_target.detach())

                # V function loss
                v_prediction = self.vf(states_batch)
                q_prediction = torch.min(self.qf1((states_batch, new_actions)), self.qf2((states_batch, new_actions)))
                v_target = q_prediction - alpha * log_probs
                vf_loss = F.mse_loss(v_prediction, v_target.detach())

                # train Q functions
                self.qf1_optim.zero_grad()
                q_1_loss.backward()
                self.qf1_optim.step()

                self.qf2_optim.zero_grad()
                q_2_loss.backward()
                self.qf2_optim.step()

                # train V function
                self.vf_optim.zero_grad()
                vf_loss.backward()
                self.vf_optim.step()

                # actor loss
                advantage = q_prediction - v_prediction.detach()
                actor_loss = (alpha * log_probs - advantage).mean()

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

                # train actor
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                # update target networks
                common_utils.soft_update(self.vf, self.vf_target, self.hyper_params.tau)

                actor_losses.append(actor_loss.item())
                q_1_losses.append(q_1_loss.item())
                q_2_losses.append(q_2_loss.item())
                vf_losses.append(vf_loss.item())
                alpha_losses.append(alpha_loss.item())

        return np.mean(actor_losses), np.mean(q_1_losses), np.mean(q_2_losses), np.mean(vf_losses), np.mean(alpha_losses)

    def write_policy_log(self, log_value):
        actor_loss, q_1_loss, q_2_loss, vf_loss, alpha_loss = log_value
        print(
            "controller_policy actor loss: %.3f\n"
            "controller_policy q1 loss loss: %.3f, controller_policy q2 loss: %.3f\n"
            "controller_policy vf loss: %.3f, controller_policy alpha loss: %.3f\n"
            % (
                actor_loss,
                q_1_loss,
                q_2_loss,
                vf_loss,
                alpha_loss
            )
        )

    def get_policy_wandb_log(self, log_value):
        actor_loss, q_1_loss, q_2_loss, vf_loss, alpha_loss = log_value
        wandb_log = {
            'actor_loss': actor_loss,
            'q_1_loss': q_1_loss,
            'q_2_loss': q_2_loss,
            'vf_loss': vf_loss,
            'alpha_loss': alpha_loss
        }
        return wandb_log

    def load_params(self, path):
        """Load model and optimizer parameters."""
        SACAgent.load_params(self, path)

    def save_params(self, n_episode):
        SACAgent.save_params(self, n_episode)

    def get_wandb_watch_list(self):
        return [self.qf1, self.qf2, self.actor, self.vf]



