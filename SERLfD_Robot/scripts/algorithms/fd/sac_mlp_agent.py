# -*- coding: utf-8 -*-
"""SAC agent from demonstration for episodic tasks in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: https://arxiv.org/pdf/1801.01290.pdf
         https://arxiv.org/pdf/1812.05905.pdf
         https://arxiv.org/pdf/1511.05952.pdf
         https://arxiv.org/pdf/1707.08817.pdf
"""

import pickle
import glob
import numpy as np
import torch
import os
import cv2
from collections import deque
import copy
import algorithms.common.helper_functions as common_utils
from algorithms.common.buffer.priortized_replay_buffer import PrioritizedReplayBufferfD
from algorithms.common.buffer.replay_buffer import NStepTransitionBuffer
from algorithms.sac.agent_mlp import Agent as SACAgent
from algorithms.common.load_config_utils import loadYAML
from algorithms.fd.se_utils import img2simpleStates, simpleStates2img
from copy import deepcopy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)

class Agent(SACAgent):
    """SAC agent interacting with environment.

    Attrtibutes:
        memory (PrioritizedReplayBufferfD): replay memory
        beta (float): beta parameter for prioritized replay buffer

    """

    # pylint: disable=attribute-defined-outside-init
    def _initialize(self):
        """Initialize non-common things."""
        # conf_str, self.conf_data = loadYAML(os.getcwd() + "/../config/fetch_serl_push_env.yaml")
        conf_str, self.conf_data = loadYAML(self.args.robot_env_config)

        self.use_n_step = self.hyper_params["N_STEP"] > 1
        self.reached_goal_reward = self.conf_data['env']['reached_goal_reward']
        self.exe_single_group = self.conf_data['fetch']['exe_single_group']
        self.exe_group_num = self.conf_data['fetch']['exe_group_num']
        self.use_shaping = self.conf_data['env']['use_shaping']
        self.use_bi_reward = not self.use_shaping
        self.avg_scores_window = deque(maxlen=100)

        if not self.args.test:
            # load demo replay memory
            # TODO: should make new demo to set protocol 2
            #       e.g. pickle.dump(your_object, your_file, protocol=2)
            demo_files = glob.glob(self.args.demo_path + '/good*/traj*.pickle')
            demos = []
            self.demos = []
            # with open(self.args.demo_path, "rb") as f:
            #     demos = pickle.load(f)
            for file in demo_files:
                with open(file, "rb") as f:
                    d = pickle.load(f, encoding="latin1")
                    demos.extend(d[0][self.hyper_params["DEMO_STARTS"]:])
                    self.demos.append(d[0])

            # Note the current environment is wrapped by "NormalizedActions", so here we need to normalize the demo actions to be within [-1, 1]
            # demos = common_utils.preprocess_demos(demos, resz=self.env.observation_space.shape[-1:-3:-1], action_space=self.env.action_space, reward_scale=50.0, exe_single_group=self.exe_single_group, exe_group_num=self.exe_group_num)
            demos = common_utils.preprocess_demos(demos, resz=self.env.observation_space.shape[-1:-3:-1], use_bi_reward=self.use_bi_reward, goal_reward=self.reached_goal_reward)

            if self.use_n_step:
                demos, demos_n_step = common_utils.get_n_step_info_from_demo(
                    demos, self.hyper_params["N_STEP"], self.hyper_params["GAMMA"]
                )

                # replay memory for multi-steps
                self.memory_n = NStepTransitionBuffer(
                    buffer_size=self.hyper_params["BUFFER_SIZE"],
                    n_step=self.hyper_params["N_STEP"],
                    gamma=self.hyper_params["GAMMA"],
                    demo=demos_n_step,
                )

            # replay memory
            self.beta = self.hyper_params["PER_BETA"]
            self.memory = PrioritizedReplayBufferfD(
                self.hyper_params["BUFFER_SIZE"],
                self.hyper_params["BATCH_SIZE"],
                demo=demos,
                alpha=self.hyper_params["PER_ALPHA"],
                epsilon_d=self.hyper_params["PER_EPS_DEMO"],
            )

    def _add_transition_to_memory(self, transition):
        """Add 1 step and n step transitions to memory."""
        # add n-step transition
        if self.use_n_step:
            transition = self.memory_n.add(transition)

        # add a single step transition
        # if transition is not an empty tuple
        if transition:
            self.memory.add(*transition)

    # pylint: disable=too-many-statements
    def update_model(self, experiences):
        """Train the model after each episode."""
        with torch.autograd.set_detect_anomaly(True):
            states, actions, rewards, next_states, dones, weights, indices, eps_d = (
                experiences
            )
            simple_states = img2simpleStates(states, end=self.hyper_params["SIMPLE_STATES_SIZE"])
            next_simple_states = img2simpleStates(next_states, end=self.hyper_params["SIMPLE_STATES_SIZE"])

            new_actions, log_prob, pre_tanh_value, mu, std = self.actor(simple_states)

            # train alpha
            if self.hyper_params["AUTO_ENTROPY_TUNING"]:
                alpha_loss = torch.mean(
                    (-self.log_alpha * (log_prob + self.target_entropy).detach()) * weights
                )

                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                alpha = self.log_alpha.exp()
            else:
                alpha_loss = torch.zeros(1)
                alpha = self.hyper_params["W_ENTROPY"]

            # Q function loss
            masks = 1 - dones
            gamma = self.hyper_params["GAMMA"]
            q_1_pred = self.qf_1(simple_states, actions)
            q_2_pred = self.qf_2(simple_states, actions)
            v_target = self.vf_target(next_simple_states)
            q_target = rewards + self.hyper_params["GAMMA"] * v_target * masks
            qf_1_loss = torch.mean((q_1_pred - q_target.detach()).pow(2) * weights)
            qf_2_loss = torch.mean((q_2_pred - q_target.detach()).pow(2) * weights)

            if self.use_n_step:
                experiences_n = self.memory_n.sample(indices)
                _, _, rewards, next_states, dones = experiences_n
                next_simple_states = img2simpleStates(next_states, end=self.hyper_params["SIMPLE_STATES_SIZE"])


                gamma = gamma ** self.hyper_params["N_STEP"]
                lambda1 = self.hyper_params["LAMBDA1"]
                masks = 1 - dones

                v_target = self.vf_target(next_simple_states)
                q_target = rewards + gamma * v_target * masks
                qf_1_loss_n = torch.mean((q_1_pred - q_target.detach()).pow(2) * weights)
                qf_2_loss_n = torch.mean((q_2_pred - q_target.detach()).pow(2) * weights)

                # to update loss and priorities
                qf_1_loss = qf_1_loss + qf_1_loss_n * lambda1
                qf_2_loss = qf_2_loss + qf_2_loss_n * lambda1

            # V function loss
            v_pred = self.vf(simple_states)
            q_pred = torch.min(
                self.qf_1(simple_states, new_actions), self.qf_2(simple_states, new_actions)
            )
            v_target = (q_pred - alpha * log_prob).detach()
            vf_loss_element_wise = (v_pred - v_target).pow(2)
            vf_loss = torch.mean(vf_loss_element_wise * weights)

            if self.total_steps % self.hyper_params["DELAYED_UPDATE"] == 0:
                # actor loss
                advantage = q_pred - v_pred.detach()
                actor_loss_element_wise = alpha * log_prob - advantage
                actor_loss = torch.mean(actor_loss_element_wise * weights)

                # regularization
                mean_reg = self.hyper_params["W_MEAN_REG"] * mu.pow(2).mean()
                std_reg = self.hyper_params["W_STD_REG"] * std.pow(2).mean()
                pre_activation_reg = self.hyper_params["W_PRE_ACTIVATION_REG"] * (
                    pre_tanh_value.pow(2).sum(dim=-1).mean()
                )
                actor_reg = mean_reg + std_reg + pre_activation_reg

                # actor loss + regularization
                actor_loss += actor_reg

                # train actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # update target networks
                common_utils.soft_update(self.vf, self.vf_target, self.hyper_params["TAU"])

                # update priorities
                new_priorities = vf_loss_element_wise
                new_priorities += self.hyper_params[
                    "LAMBDA3"
                ] * actor_loss_element_wise.pow(2)
                new_priorities += self.hyper_params["PER_EPS"]
                new_priorities = new_priorities.data.cpu().numpy().squeeze()
                new_priorities += eps_d
                self.memory.update_priorities(indices, new_priorities)

                # increase beta
                fraction = min(float(self.i_episode) / self.args.episode_num, 1.0)
                self.beta = self.beta + fraction * (1.0 - self.beta)
            else:
                actor_loss = torch.zeros(1)


            # train Q functions
            self.qf_1_optimizer.zero_grad()
            qf_1_loss.backward()
            self.qf_1_optimizer.step()

            self.qf_2_optimizer.zero_grad()
            qf_2_loss.backward()
            self.qf_2_optimizer.step()

            # train V function
            self.vf_optimizer.zero_grad()
            vf_loss.backward()
            self.vf_optimizer.step()

            return (
                actor_loss.data,
                qf_1_loss.data,
                qf_2_loss.data,
                vf_loss.data,
                alpha_loss.data,
            )

    def pretrain(self):
        """Pretraining steps."""
        pretrain_loss = list()
        print("[INFO] Pre-Train %d steps." % self.hyper_params["PRETRAIN_STEP"])
        for i_step in range(1, self.hyper_params["PRETRAIN_STEP"] + 1):
            experiences = self.memory.sample(beta=1.0)
            loss = self.update_model(experiences)
            pretrain_loss.append(loss)  # for logging

            # logging
            if i_step == 1 or i_step % 100 == 0:
                avg_loss = np.vstack(pretrain_loss).mean(axis=0)
                pretrain_loss.clear()
                self.write_log(
                    0, avg_loss, 0)#, delayed_update=self.hyper_params["DELAYED_UPDATE"]
                # )
