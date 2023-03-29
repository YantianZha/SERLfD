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
import learning_agents.common.common_utils as common_utils
from learning_agents.architectures.mlp import MLP
from learning_agents.irl.gan_gcl.gan_gcl_models import DefaultDiscriminatorCNN
from learning_agents.irl.gan_gcl.gan_gcl_single import GAN_GCL_Single
from learning_agents.controller_policy.base_policy import TRAJECTORY_INFO_INDEX
from utils.trajectory_utils import split_fixed_length_indexed_traj, extract_experiences_from_indexed_trajs
from learning_agents.utils.utils import ConfigDict, IndexedTraj
from utils.trajectory_utils import TRAJECTORY_INDEX

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOG_REG = 1e-8


# noinspection DuplicatedCode
class GAN_GCL_Trajectory(GAN_GCL_Single):
    """
    Guided Cost Learning in GAN formula (take single state-action pair as input)
    This module is inherited from GAIL

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

    def compute_trajs_info(self, indexed_trajs):
        """
        Compute log_q_tau and log_p_tau of each traj in indexed_trajs
        :return: Tensors storing log_q_tau and log_p_tau of each traj
        """
        log_q_taus = []
        log_p_taus = []
        for traj_idx, traj in enumerate(indexed_trajs):
            # extract states and actions
            states = np.array(traj[TRAJECTORY_INDEX.STATE.value])
            actions = np.array(traj[TRAJECTORY_INDEX.ACTION.value])

            torch_actions = torch.from_numpy(actions).type(torch.FloatTensor).to(device)
            torch_states = self._preprocess_state(states)
            states_actions = torch.cat((torch_states, torch_actions), 1)
            # energy is -log(p_tau) = -log(1/Z*exp(-cost(tau))) or -log(1/Z*exp(reward))= -reward in a
            energy = self.irl_model(states_actions)
            log_p_tau = torch.sum(-energy, dim=0, keepdim=True)
            log_p_taus.append(log_p_tau)

            # compute log_q_tau
            policy_log_probs = self.policy.evaluate_states_actions(states, actions)
            log_q_taus.append(np.sum(policy_log_probs).reshape(1))

        batch_log_p = torch.cat(log_p_taus)
        batch_log_q = torch.Tensor(log_q_taus).type(torch.FloatTensor).to(device)
        batch_log_p_q = (batch_log_q.exp() + batch_log_p.exp() + LOG_REG).log()

        return batch_log_p, batch_log_q, batch_log_p_q

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
            # sample expert trajectories
            demo_indices = np.random.randint(low=0, high=n_expert_trajs, size=expert_batch_size)
            fixed_len_demo_trajs = self.demo_buf.sample(indices=demo_indices)
            n_fixed_demo_trajs = len(fixed_len_demo_trajs)

            # sample sampled trajectories
            # if traj_batch_size is negative, use all sampled trajectories
            if self.hyper_params.traj_batch_size < 0:
                sample_indices = np.random.randint(low=0, high=n_sampled_trajs, size=n_sampled_trajs)
                fixed_len_sampled_trajs = self.sampled_traj_buf.sample(indices=sample_indices)
            # if not use all sampled trajs, this current sampled ones and sample subsets of past trajs
            else:
                past_trajs = []
                if past_batch_size > 0:
                    past_trajs = self.sampled_traj_buf.sample(batch_size=past_batch_size)
                fixed_len_sampled_trajs = indexed_sampled_trajs + past_trajs
            n_fixed_sampled_trajs = len(fixed_len_sampled_trajs)

            arrange = np.arange(min(n_fixed_sampled_trajs, n_fixed_demo_trajs))
            np.random.shuffle(arrange)
            n_sample_trained = len(arrange)
            batch_size = int(self.hyper_params.batch_size / 2)
            for i in range(arrange.shape[0] // batch_size):
                start_idx = batch_size * i
                end_idx = batch_size * (i + 1)
                batch_index = arrange[start_idx:end_idx]

                batch_demo_trajs = [fixed_len_demo_trajs[i] for i in batch_index]
                batch_demo_labels = np.ones(shape=(len(batch_demo_trajs),), dtype=int)
                n_batch_demo = len(batch_demo_trajs)

                batch_sample_trajs = [fixed_len_sampled_trajs[i] for i in batch_index]
                batch_sample_labels = np.zeros(shape=(len(batch_sample_trajs),), dtype=int)
                n_batch_sample = len(batch_sample_trajs)

                batch_labels = torch.FloatTensor(
                    torch.from_numpy(np.concatenate((batch_demo_labels, batch_sample_labels), axis=0)).float()).to(
                    device).unsqueeze(1)
                # compute log_q, log_p and log_p_q
                indexed_trajs = batch_demo_trajs + batch_sample_trajs
                batch_log_p, batch_log_q, batch_log_p_q = self.compute_trajs_info(indexed_trajs)

                loss = batch_labels * (batch_log_p - batch_log_p_q) + (1 - batch_labels) * (batch_log_q - batch_log_p_q)

                # maximize the log likelihood -> minimize mean loss
                mean_loss = -torch.mean(loss)
                self.irl_model_optim.zero_grad()
                mean_loss.backward()
                self.irl_model_optim.step()

                # for logging: get the discriminator output
                clone_log_p_tau = batch_log_p.clone().detach().cpu().numpy()
                clone_log_p_q = batch_log_p_q.clone().detach().cpu().numpy()
                demo_prediction = np.exp(clone_log_p_tau[0:n_batch_demo] - clone_log_p_q[0:n_batch_demo])
                sample_prediction = np.exp(
                    clone_log_p_tau[n_batch_demo:n_batch_demo + n_batch_sample]
                    - clone_log_p_q[n_batch_demo:n_batch_demo + n_batch_sample])
                demo_predict_acc.append(np.mean((demo_prediction > 0.5).astype(float)))
                sampled_predict_acc.append(np.mean((sample_prediction < 0.5).astype(float)))
                losses_iteration.append(mean_loss.item())

        avg_loss = sum(losses_iteration) / float(len(losses_iteration))
        demo_avg_acc = sum(demo_predict_acc) / float(len(demo_predict_acc))
        sampled_avg_acc = sum(sampled_predict_acc) / float(len(sampled_predict_acc))
        return avg_loss, demo_avg_acc, sampled_avg_acc, n_sample_trained









