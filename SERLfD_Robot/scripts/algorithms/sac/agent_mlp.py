# -*- coding: utf-8 -*-
"""SAC agent for episodic tasks in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: https://arxiv.org/pdf/1801.01290.pdf
         https://arxiv.org/pdf/1812.05905.pdf
"""

import os
import pickle
import datetime
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from collections import deque
from concurrent import futures
import algorithms.common.helper_functions as common_utils
from algorithms.common.abstract.agent import AbstractAgent
from algorithms.common.buffer.replay_buffer import ReplayBuffer
import cv2
from algorithms.fd.se_utils import img2simpleStates, simpleStates2img
from algorithms.common.load_config_utils import loadYAML
from algorithms.common.helper_functions import draw_predicates_on_img
import time
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(AbstractAgent):
    """SAC agent interacting with environment.

    Attrtibutes:
        memory (ReplayBuffer): replay memory
        actor (nn.Module): actor model to select actions
        actor_target (nn.Module): target actor model to select actions
        actor_optimizer (Optimizer): optimizer for training actor
        critic_1 (nn.Module): critic model to predict state values
        critic_2 (nn.Module): critic model to predict state values
        critic_target1 (nn.Module): target critic model to predict state values
        critic_target2 (nn.Module): target critic model to predict state values
        critic_optimizer1 (Optimizer): optimizer for training critic_1
        critic_optimizer2 (Optimizer): optimizer for training critic_2
        curr_state (np.ndarray): temporary storage of the current state
        target_entropy (int): desired entropy used for the inequality constraint
        alpha (torch.Tensor): weight for entropy
        alpha_optimizer (Optimizer): optimizer for alpha
        hyper_params (dict): hyper-parameters
        total_step (int): total step numbers
        episode_step (int): step number of the current episode
        i_episode (int): current episode number
        her (HER): hinsight experience replay

    """

    def __init__(self, env, args, hyper_params, models, optims, target_entropy, her, sparse=False):
        """Initialization.

        Args:
            env (gym.Env): openAI Gym environment
            args (argparse.Namespace): arguments including hyperparameters and training settings
            hyper_params (dict): hyper-parameters
            models (tuple): models including actor and critic
            optims (tuple): optimizers for actor and critic
            target_entropy (float): target entropy for the inequality constraint
            her (HER): hinsight experience replay

        """
        AbstractAgent.__init__(self, env, args)

        self.actor, self.vf, self.vf_target, self.qf_1, self.qf_2 = models[:5]
        self.actor_optimizer, self.vf_optimizer = optims[0:2]
        self.qf_1_optimizer, self.qf_2_optimizer = optims[2:4]
        self.hyper_params = hyper_params
        self.curr_state = np.zeros((1,))
        self.total_steps = 0
        self.episode_steps = 0
        self.sparse = sparse
        self.all_score = []
        self.i_episode = 0
        self.avg_scores_window = deque(maxlen=self.args.avg_score_window)

        ts = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.train_img_folder = '/data/outputs/SERL/fetch-' + self.args.algo + '_' + ts
        conf_str, self.conf_data = loadYAML(self.args.robot_env_config)
        self.reached_goal_reward = self.conf_data['env']['reached_goal_reward']
        self.exe_single_group = self.conf_data['fetch']['exe_single_group']
        self.exe_group_num = self.conf_data['fetch']['exe_group_num']
        self.use_shaping = self.conf_data['env']['use_shaping']
        self.use_bi_reward = not self.use_shaping
        self.avg_scores_window = deque(maxlen=self.args.avg_score_window)

        if not os.path.exists(self.train_img_folder):
            os.makedirs(self.train_img_folder)

        # automatic entropy tuning
        if self.hyper_params["AUTO_ENTROPY_TUNING"]:
            self.target_entropy = target_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam(
                [self.log_alpha], lr=self.hyper_params["LR_ENTROPY"]
            )

        # load the optimizer and model parameters
        if args.load_from is not None and os.path.exists(args.load_from) and not "LR_EXPLAINER" in self.hyper_params:
            self.load_params(args.load_from)

        self._initialize()

    def _initialize(self):
        """Initialize non-common things."""
        if not self.args.test:
            # replay memory
            self.memory = ReplayBuffer(
                self.hyper_params["BUFFER_SIZE"], self.hyper_params["BATCH_SIZE"]
            )

        # HER
        if self.hyper_params["USE_HER"]:
            # load demo replay memory
            with open(self.args.demo_path, "rb") as f:
                demo = pickle.load(f)

            if self.hyper_params["DESIRED_STATES_FROM_DEMO"]:
                self.her.fetch_desired_states_from_demo(demo)

            self.transitions_epi = list()
            self.desired_state = np.zeros((1,))
            demo = self.her.generate_demo_transitions(demo)

        if not self.args.test:
            # Replay buffers
            self.memory = ReplayBuffer(
                self.hyper_params["BUFFER_SIZE"], self.hyper_params["BATCH_SIZE"]
            )

    def _preprocess_state(self, state):
        """Preprocess state so that actor selects an action."""
        if self.hyper_params["USE_HER"]:
            self.desired_state = self.her.get_desired_state()
            state = np.concatenate((state, self.desired_state), axis=-1)
        state = torch.FloatTensor(state).to(device)
        return state

    def _add_transition_to_memory(self, transition):
        """Add 1 step and n step transitions to memory."""
        if self.hyper_params["USE_HER"]:
            self.transitions_epi.append(transition)
            done = transition[-1] or self.episode_step == self.args.max_episode_steps
            if done:
                # insert generated transitions if the episode is done
                transitions = self.her.generate_transitions(
                    self.transitions_epi,
                    self.desired_state,
                    self.hyper_params["SUCCESS_SCORE"],
                )
                self.memory.extend(transitions)
                self.transitions_epi = list()
        else:
            self.memory.add(*transition)

    def select_action(self, state):
        """Select an action from the input space."""
        self.curr_state = state
        state = self._preprocess_state(state)
        state = img2simpleStates(state, end=self.hyper_params["SIMPLE_STATES_SIZE"])

        # if initial random action should be conducted
        if (
            self.total_steps < self.hyper_params["INITIAL_RANDOM_ACTION"]
            and not self.args.test
        ):
            # unscaled_random_action = self.env.action_space.sample()
            # # Unscaled between true action_space bounds
            # return common_utils.reverse_action(unscaled_random_action, self.env.action_space)
            return self.env.action_space.sample()

        if self.args.test:
            _, _, _, selected_action, _ = self.actor(state)
        else:
            selected_action, _, _, _, _ = self.actor(state)

        return selected_action.detach().cpu().numpy()

    def step(self, action):
        """Take an action and return the response of the env."""
        self.total_steps += 1
        self.episode_step += 1

        next_state, reward, done, _ = self.env.step(action)

        if not self.args.test:
            done_bool = (
                False if reward != self.reached_goal_reward else True
            )
            transition = (self.curr_state, action, reward, next_state, done_bool)
            self._add_transition_to_memory(transition)

        return next_state, reward, done

    def update_model(self, experiences):
        """Train the model after each episode."""
        with torch.autograd.set_detect_anomaly(True):
            states, actions, rewards, next_states, dones = experiences
            states = img2simpleStates(states, end=self.hyper_params["SIMPLE_STATES_SIZE"])
            next_states = img2simpleStates(next_states, end=self.hyper_params["SIMPLE_STATES_SIZE"])

            new_actions, log_prob, pre_tanh_value, mu, std = self.actor(states)

            # train alpha
            if self.hyper_params["AUTO_ENTROPY_TUNING"]:
                alpha_loss = torch.mean(
                    -self.log_alpha * (log_prob + self.target_entropy).detach()
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
            q_1_pred = self.qf_1(states, actions)
            q_2_pred = self.qf_2(states, actions)
            v_target = self.vf_target(next_states)
            q_target = rewards + self.hyper_params["GAMMA"] * v_target * masks
            qf_1_loss = torch.mean((q_1_pred - q_target.detach()).pow(2))
            qf_2_loss = torch.mean((q_2_pred - q_target.detach()).pow(2))

            # V function loss
            v_pred = self.vf(states)
            q_pred = torch.min(
                self.qf_1(states, new_actions), self.qf_2(states, new_actions)
            )
            v_target = (q_pred - alpha * log_prob).detach()
            vf_loss = (v_pred - v_target).pow(2)
            vf_loss = torch.mean(vf_loss)

            # # train Q functions
            # self.qf_1_optimizer.zero_grad()
            # qf_1_loss.backward()
            # self.qf_1_optimizer.step()
            #
            # self.qf_2_optimizer.zero_grad()
            # qf_2_loss.backward()
            # self.qf_2_optimizer.step()
            #
            # # train V function
            # self.vf_optimizer.zero_grad()
            # vf_loss.backward()
            # self.vf_optimizer.step()

            if self.total_steps % self.hyper_params["DELAYED_UPDATE"] == 0:
                # actor loss
                advantage = q_pred - v_pred.detach()
                actor_loss = alpha * log_prob - advantage
                actor_loss = torch.mean(actor_loss)

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

    def load_params(self, path):
        """Load model and optimizer parameters."""
        if not os.path.exists(path):
            print("[ERROR] the input path does not exist. ->", path)
            return

        params = torch.load(path)
        self.actor.load_state_dict(params["actor"])
        self.qf_1.load_state_dict(params["qf_1"])
        self.qf_2.load_state_dict(params["qf_2"])
        self.vf.load_state_dict(params["vf"])
        self.vf_target.load_state_dict(params["vf_target"])
        self.actor_optimizer.load_state_dict(params["actor_optim"])
        self.qf_1_optimizer.load_state_dict(params["qf_1_optim"])
        self.qf_2_optimizer.load_state_dict(params["qf_2_optim"])
        self.vf_optimizer.load_state_dict(params["vf_optim"])

        if self.hyper_params["AUTO_ENTROPY_TUNING"]:
            self.alpha_optimizer.load_state_dict(params["alpha_optim"])

        print("[INFO] loaded the model and optimizer from", path)

    def save_params(self, n_episode, params=None):
        """Save model and optimizer parameters."""

        if not params:
            params = {
                "actor": self.actor.state_dict(),
                "qf_1": self.qf_1.state_dict(),
                "qf_2": self.qf_2.state_dict(),
                "vf": self.vf.state_dict(),
                "vf_target": self.vf_target.state_dict(),
                "actor_optim": self.actor_optimizer.state_dict(),
                "qf_1_optim": self.qf_1_optimizer.state_dict(),
                "qf_2_optim": self.qf_2_optimizer.state_dict(),
                "vf_optim": self.vf_optimizer.state_dict(),
            }

            if self.hyper_params["AUTO_ENTROPY_TUNING"]:
                params["alpha_optim"] = self.alpha_optimizer.state_dict()

        AbstractAgent.save_params(self, params, n_episode)

    def write_log(self, i, loss, score=0.0, delayed_update=1, avg_score_window=None):
        """Write log about loss and score"""
        total_loss = loss.sum()

        print(
            "[INFO] episode %d, episode_step %d, total step %d, total score: %d\n"
            "total loss: %.3f actor_loss: %.3f qf_1_loss: %.3f qf_2_loss: %.3f "
            "vf_loss: %.3f alpha_loss: %.3f\n"
            % (
                i,
                self.i_episode,
                self.total_steps,
                score,
                total_loss,
                loss[0] * delayed_update,  # actor loss
                loss[1],  # qf_1 loss
                loss[2],  # qf_2 loss
                loss[3],  # vf loss
                loss[4],  # alpha loss
            )
        )

        if self.args.log:
            wandb.log(
                {"total_steps": self.total_steps,
                    "score": score,
                    "total loss": total_loss,
                    "actor loss": loss[0] * delayed_update,
                    "qf_1 loss": loss[1],
                    "qf_2 loss": loss[2],
                    "vf loss": loss[3],
                    "alpha loss": loss[4],
                }, step=i
            )

            if avg_score_window:
                wandb.log({"avg_score_window": avg_score_window}, step=i)

    def train(self):
        """Train the agent."""
        # logger
        if self.args.log:
            wandb.init(dir='data_2/data')
            wandb.config.update(self.hyper_params)
            wandb.config.update(vars(self.args))
            wandb.watch([self.actor, self.vf, self.qf_1, self.qf_2], log="parameters")

        if self.hyper_params["IF_PRETRAIN_DEMO"]:
            self.pretrain()

        for self.i_episode in range(1, self.args.episode_num + 1):
            state = self.env.reset()
            done = False
            score = 0
            self.episode_step = 0
            loss_episode = list()

            while not done:
                if self.args.render and self.i_episode >= self.args.render_after:
                    self.env.render()

                # training
                if len(self.memory) >= self.hyper_params["BATCH_SIZE"]:
                    experiences = self.memory.sample()
                    loss = self.update_model(experiences)
                    loss_episode.append(loss)  # for logging

                action = self.select_action(state)
                next_state, reward, done = self.step(action)
                t_state = draw_predicates_on_img(state, self.hyper_params["SIMPLE_STATES_SIZE"], (640, 480), reward, done)
                cv2.imwrite(self.train_img_folder + '/color_img_' + str(self.i_episode) + '_' + str(self.episode_step) + '.jpg', t_state)
                print("epi and step: ", self.i_episode, self.episode_step, action, done, reward)
                state = next_state
                score += reward


            t_state = draw_predicates_on_img(state, self.hyper_params["SIMPLE_STATES_SIZE"], (640, 480), reward, done)
            cv2.imwrite(
                self.train_img_folder + '/color_img_' + str(self.i_episode) + '_' + str(self.episode_step) + '.jpg',
                t_state)

            # logging
            self.avg_scores_window.append(score)
            avg_score_window = float(np.mean(list(self.avg_scores_window)))

            if loss_episode:
                avg_loss = np.vstack(loss_episode).mean(axis=0)
                self.write_log(
                    self.i_episode, avg_loss, score, self.hyper_params["DELAYED_UPDATE"], avg_score_window
                )

            if self.i_episode % self.args.save_period == 0:
                self.save_params(self.i_episode)

        # termination
        self.env.close()
