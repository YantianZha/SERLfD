import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import gym

from learning_agents.agent import Agent
from learning_agents.common.trajectory_buffer import TrajectoryBuffer
from learning_agents.architectures.mlp import MLP
from learning_agents.irl.gan_gcl.gan_gcl_single import GAN_GCL_Single
from utils.trajectory_utils import extract_experiences_from_indexed_trajs
from learning_agents.utils.utils import ConfigDict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOG_REG = 1e-8


class AIRL_Single(GAN_GCL_Single):
    """
    Adversarial Inverse Reinforcement Learning
    This module is inherited from GAIL

    Attributes:
        ####################### attributes of the IRL (explainer) ##########################
        env (gym.Env): openAI Gym environment
        irl_model (torch.nn): the irl/explainer model
        reward_fn_optim (torch.optim): the irl/explainer network's optimizer
        args (argparse.Namespace): arguments including hyperparameters and training settings
        hyper_params (ConfigDict): hyper-parameters
        network_cfg (ConfigDict): config of network for training agent
        encoder(nn): if encoder is not None, the irl/explainer will use the encoder to preprocess the state
        optim_cfg (ConfigDict): config of optimizer
        state_dim (int): state size of env
        action_dim (int): action size of env
        demo_buf (TrajectoryBuffer): buffer of good trajectories (including expert demo), trajs are stored in indexed format
        sampled_traj_buf (TrajectoryBuffer): buffer of bad trajectories, trajs are stored in indexed format
        curr_state (np.ndarray): temporary storage of the current state
        total_step (int): total step numbers
        i_iteration (int): current iteration number
        ####################### attributes of the RL (controller_policy) ##############################
        policy_network (Agent): the pre-defined controller_policy network
    """

    def __init__(self, env, policy, args, log_cfg, hyper_params, network_cfg, optim_cfg, encoder=None, logger=None):
        """
        Initialize.
        env: gym.Env,
        args: argparse.Namespace,
        log_cfg: ConfigDict,
        hyper_params: ConfigDict,
        network_cfg: ConfigDict,
        optim_cfg: ConfigDict,
        controller_policy (Agent): the controller_policy module
        encoder(nn): if encoder is not None, the irl/explainer will use the encoder to preprocess the state
        logger (ExperimentLogger): the experiment logger
        """

        GAN_GCL_Single.__init__(self, env, policy, args, log_cfg, hyper_params, network_cfg, optim_cfg, encoder=encoder, logger=logger)

    def _set_to_default_MLP(self):
        # the irl model
        self.reward_fn = MLP(
            input_size=self.state_dim + self.action_dim,
            output_size=1,
            hidden_sizes=self.network_cfg.hidden_sizes_reward,
            hidden_activation=torch.relu
        ).to(device)

        self.value_fn = MLP(
            input_size=self.state_dim,
            output_size=1,
            hidden_sizes=self.network_cfg.hidden_size_value,
            hidden_activation=torch.relu
        ).to(device)

    def _set_to_default_CNN(self):
        pass

    def _init_network(self):
        """ Initialize the IRL (explainer) network """
        self._set_to_default_MLP()

        # define optimizer
        self.reward_fn_optim = optim.Adam(
            self.reward_fn.parameters(),
            lr=self.optim_cfg.lr_reward_fn,
            weight_decay=self.optim_cfg.weight_decay
        )

        self.value_fn_optim = optim.Adam(
            self.reward_fn.parameters(),
            lr=self.optim_cfg.lr_value_fn,
            weight_decay=self.optim_cfg.weight_decay
        )

        # load the optimizer and model parameters
        if self.args.load_from is not None:
            self.load_params(self.args.load_from)

    def load_params(self, path):
        """Load model and optimizer parameters."""
        Agent.load_params(self, path)

        params = torch.load(path, map_location=device)
        self.reward_fn.load_state_dict(params["reward_fn_state_dict"])
        self.reward_fn_optim.load_state_dict(params["reward_fn_optim_state_dict"])
        self.value_fn.load_state_dict(params["value_fn_state_dict"])
        self.value_fn_optim.load_state_dict(params["value_fn_optim_state_dict"])
        print("[INFO] loaded the model and optimizer from", path)

    # noinspection PyMethodOverriding
    def save_params(self, n_episode):
        """Save model and optimizer parameters."""
        params = {
            "reward_fn_state_dict": self.reward_fn.state_dict(),
            "reward_fn_optim_state_dict": self.reward_fn_optim.state_dict(),
            "value_fn_state_dict": self.value_fn.state_dict(),
            "value_fn_optim_state_dict": self.value_fn_optim.state_dict(),
            "epoch": n_episode
        }
        if self.logger is not None:
            self.logger.save_models(params, postfix=str(n_episode), is_snapshot=True)
        Agent.save_params(self, params, n_episode)

    def get_reward(self, state, action=None):
        """
        Evaluate a single state-action pair. Will return the scalar value.

        :param states: numpy.ndarray
        :param actions: numpy.ndarray (in discrete case, use one-hot representation)
        :return: reward function, i.e. the log of energy function (if want cost, use the negation of the reward)
        """
        reward = torch.sum(self.eval(state, action), dim=1, keepdim=False).detach().cpu().numpy()
        return reward

    def eval(self, states, actions=None):
        """
        Evaluate the state action pairs (compute the energy value)
        Note this will return a Variable rather than the data in the numpy format.

        :param states: numpy.ndarray
        :param actions: numpy.ndarray (in discrete case, use one-hot representation)
        :return: reward function, i.e. the log of energy function (if want cost, use the negation of the reward)
        """
        states = self._preprocess_state(states)
        if not isinstance(actions, torch.Tensor):
            if isinstance(actions, np.ndarray):
                actions = Variable(torch.from_numpy(actions)).type(torch.FloatTensor).to(device)
            else:
                actions = Variable(torch.Tensor(actions)).type(torch.FloatTensor).to(device)

        return self.reward_fn(torch.cat((states, actions), 1))

    # noinspection PyArgumentList
    def update_irl(self, indexed_sampled_trajs):
        """
        update the discriminator
        :param: indexed_sampled_traj: the sampled trajectories from current iteration
        """
        # calculate the expert batch size
        n_expert_trajs = len(self.demo_buf)
        if self.hyper_params.traj_batch_size > 0:
            expert_batch_size = min(n_expert_trajs, self.hyper_params.traj_batch_size)
        else:
            expert_batch_size = n_expert_trajs

        # sample batch_size
        n_sampled_trajs = len(self.sampled_traj_buf)
        if self.hyper_params.is_fusion:
            past_batch_size = min(n_sampled_trajs,
                                  max(0, self.hyper_params.traj_batch_size - len(indexed_sampled_trajs)))
        else:
            past_batch_size = 0

        losses_iteration = []
        demo_predict_acc = []
        sampled_predict_acc = []
        n_sample_trained = 0
        for it in range(self.hyper_params.num_iteration_update):
            # sampled expert trajectories
            demo_indices = np.random.randint(low=0, high=n_expert_trajs, size=expert_batch_size)
            demo_states, demo_actions, _, demo_next_states, _ = self.demo_buf.get_experiences_from_trajs(indices=demo_indices)

            # sampled sample trajectories
            # if traj_batch_size is negative, use all sampled trajectories
            if self.hyper_params.traj_batch_size < 0:
                sample_indices = np.random.randint(low=0, high=n_sampled_trajs, size=n_sampled_trajs)
                sample_states, sample_actions, _, sample_next_states, _ = self.sampled_traj_buf.get_experiences_from_trajs(
                    indices=sample_indices)
            else:
                past_trajs = []
                if past_batch_size > 0:
                    past_trajs = self.sampled_traj_buf.sample(batch_size=past_batch_size)
                sampled_trajs = indexed_sampled_trajs + past_trajs
                sample_states, sample_actions, _, sample_next_states, _ = extract_experiences_from_indexed_trajs(sampled_trajs)

            # compute the log-probability of each action
            demo_log_probs = self.policy.evaluate_states_actions(demo_states, demo_actions)
            sampled_log_probs = self.policy.evaluate_states_actions(sample_states, sample_actions)

            n_demo_states = demo_states.shape[0]
            n_sampled_states = sample_states.shape[0]
            arrange = np.arange(min(n_demo_states, n_sampled_states))
            np.random.shuffle(arrange)
            n_sample_trained = len(arrange)
            batch_size = int(self.hyper_params.batch_size/2)
            for i in range(arrange.shape[0] // batch_size):
                start_idx = batch_size * i
                end_idx = batch_size * (i + 1)
                batch_index = arrange[start_idx:end_idx]

                # demo states
                batch_demo_states = demo_states[batch_index]
                batch_demo_actions = demo_actions[batch_index]
                batch_demo_next_states = demo_next_states[batch_index]
                batch_demo_log_probs = demo_log_probs[batch_index]
                batch_demo_label = np.ones(shape=(batch_demo_states.shape[0],), dtype=int)
                n_demo = batch_demo_label.shape[0]
                # sample states
                batch_sample_states = sample_states[batch_index]
                batch_sample_actions = sample_actions[batch_index]
                batch_sample_next_states = sample_next_states[batch_index]
                batch_sample_log_probs = sampled_log_probs[batch_index]
                batch_sample_label = np.zeros(shape=(batch_sample_states.shape[0],), dtype=int)
                n_sample = batch_sample_label.shape[0]

                batch_actions = torch.FloatTensor(torch.from_numpy(np.concatenate((batch_demo_actions, batch_sample_actions), axis=0))).to(device)
                batch_states = self._preprocess_state(np.concatenate((batch_demo_states, batch_sample_states), axis=0)).to(device)
                batch_next_states = self._preprocess_state(np.concatenate((batch_demo_next_states, batch_sample_next_states), axis=0)).to(device)
                batch_log_probs = torch.FloatTensor(torch.from_numpy(np.concatenate((batch_demo_log_probs, batch_sample_log_probs), axis=0))).to(device)
                batch_labels = torch.FloatTensor(torch.from_numpy(np.concatenate((batch_demo_label, batch_sample_label), axis=0)).float()).to(device).unsqueeze(1)

                irl_rewards = self.reward_fn(torch.cat((batch_states, batch_actions), 1))

                value_states = self.value_fn(batch_states)
                value_states_next = self.value_fn(batch_next_states)

                # log(p_tau) = -energy = log(1/Z*exp(reward)) = log(exp(reward+const)) = reward + const = advantage
                log_p_tau = irl_rewards + self.hyper_params.gamma * value_states_next - value_states
                log_q_tau = batch_log_probs.unsqueeze(1)
                # find log(p_tau + q_tau) in the format of Tensor(shape=[n_traj, 1])
                log_p_q = (log_q_tau.exp() + log_p_tau.exp() + LOG_REG).log()
                # find the D(tau)=p/(p+q)=exp(log(p/(p+q)))=exp(log_p_tau-log_p_q)
                loss = batch_labels * (log_p_tau - log_p_q) + (1 - batch_labels) * (log_q_tau - log_p_q)

                # maximize the log likelihood -> minimize mean loss
                mean_loss = -torch.mean(loss)
                self.reward_fn_optim.zero_grad()
                mean_loss.backward(retain_graph=True)
                self.reward_fn_optim.step()

                self.value_fn_optim.zero_grad()
                mean_loss.backward()
                self.value_fn_optim.step()

                # for logging: get the discriminator output
                clone_log_p_tau = log_p_tau.clone().detach().cpu().numpy()
                clone_log_p_q = log_p_q.clone().detach().cpu().numpy()
                demo_prediction = np.exp(clone_log_p_tau[0:n_demo] - clone_log_p_q[0:n_demo])
                sample_prediction = np.exp(clone_log_p_tau[n_demo:n_demo + n_sample] - clone_log_p_q[n_demo:n_demo + n_sample])
                demo_predict_acc.append(np.mean((demo_prediction > 0.5).astype(float)))
                sampled_predict_acc.append(np.mean((sample_prediction < 0.5).astype(float)))

                losses_iteration.append(mean_loss.item())

        avg_loss = sum(losses_iteration) / float(len(losses_iteration))
        demo_avg_acc = sum(demo_predict_acc) / float(len(demo_predict_acc))
        sampled_avg_acc = sum(sampled_predict_acc) / float(len(sampled_predict_acc))
        return avg_loss, demo_avg_acc, sampled_avg_acc, n_sample_trained

    def get_wandb_watch_list(self):
        return [self.reward_fn, self.value_fn]




