# -*- coding: utf-8 -*-
"""TD3 agent from demonstration for episodic tasks in OpenAI Gym.

- Author: Seungjae Ryan Lee
- Contact: seungjaeryanlee@gmail.com
- Paper: https://arxiv.org/pdf/1802.09477.pdf (TD3)
         https://arxiv.org/pdf/1511.05952.pdf (PER)
         https://arxiv.org/pdf/1707.08817.pdf (DDPGfD)
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
from algorithms.td3.agent_mlp import Agent as TD3Agent
from algorithms.fd.se_utils import img2simpleStates, simpleStates2img
from algorithms.common.load_config_utils import loadYAML
from algorithms.common.helper_functions import draw_predicates_on_img

from kair_algorithms_draft.scripts.experiment_record_utils import ExperimentLogger

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(TD3Agent):
    """TD3 agent interacting with environment.

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
        self.avg_scores_window = deque(maxlen=self.args.avg_score_window)

        if not self.args.test:
            # load demo replay memory
            # TODO: should make new demo to set protocol 2
            #       e.g. pickle.dump(your_object, your_file, protocol=2)
            demo_files = glob.glob(self.args.demo_path + '/good*/traj*.pickle')
            demos = []
            self.demos = []
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

    def _get_critic_loss(self, experiences, gamma):
        """Return element-wise critic loss."""
        states, actions, rewards, next_states, dones = experiences[:5]

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        masks = 1 - dones

        # Added different noises to each action batch
        next_actions = self.actor_target(next_states)
        for i in range(self.hyper_params["BATCH_SIZE"]):
            noise = torch.FloatTensor(self.target_policy_noise.sample(self.total_steps)).to(device)
            clipped_noise = torch.clamp(
                noise,
                -self.hyper_params["TARGET_POLICY_NOISE_CLIP"],
                self.hyper_params["TARGET_POLICY_NOISE_CLIP"],
            )
            next_actions[i] = (next_actions[i] + clipped_noise).clamp(-1.0, 1.0)

        target_values1 = self.critic1_target(
            torch.cat((next_states, next_actions), dim=-1)
        )
        target_values2 = self.critic2_target(
            torch.cat((next_states, next_actions), dim=-1)
        )
        target_values = torch.min(target_values1, target_values2)
        target_values = rewards + (gamma * target_values * masks).detach()

        # train critic
        values1 = self.critic1(torch.cat((states, actions), dim=-1))
        critic1_loss_element_wise = (values1 - target_values.detach()).pow(2)

        values2 = self.critic2(torch.cat((states, actions), dim=-1))
        critic2_loss_element_wise = (values2 - target_values.detach()).pow(2)

        return critic1_loss_element_wise, critic2_loss_element_wise

    # pylint: disable=too-many-statements
    def update_model(self, experiences):
        """Train the model after each episode."""
        states, actions, rewards, next_states, dones, weights, indices, eps_d = (
            experiences
        )
        states = img2simpleStates(states, end=self.hyper_params["SIMPLE_STATES_SIZE"])
        next_states = img2simpleStates(next_states, end=self.hyper_params["SIMPLE_STATES_SIZE"])

        experiences = states, actions, rewards, next_states, dones, weights, indices, eps_d

        gamma = self.hyper_params["GAMMA"]
        critic1_loss_element_wise, critic2_loss_element_wise = self._get_critic_loss(
            experiences, gamma
        )
        critic_loss_element_wise = critic1_loss_element_wise + critic2_loss_element_wise
        critic1_loss = torch.mean(critic1_loss_element_wise * weights)
        critic2_loss = torch.mean(critic2_loss_element_wise * weights)
        critic_loss = critic1_loss + critic2_loss

        if self.use_n_step:
            experiences_n = self.memory_n.sample(indices)
            states_n, actions_n, rewards_n, next_states_n, dones_n = (
                experiences_n
            )
            states_n, next_states_n = img2simpleStates(states_n, end=self.hyper_params["SIMPLE_STATES_SIZE"]), img2simpleStates(next_states_n, end=self.hyper_params[
                    "SIMPLE_STATES_SIZE"])

            experiences_n = states_n, actions_n, rewards_n, next_states_n, dones_n


            gamma = self.hyper_params["GAMMA"] ** self.hyper_params["N_STEP"]
            critic1_loss_n_element_wise, critic2_loss_n_element_wise = self._get_critic_loss(
                experiences_n, gamma
            )
            critic_loss_n_element_wise = (
                critic1_loss_n_element_wise + critic2_loss_n_element_wise
            )
            critic1_loss_n = torch.mean(critic1_loss_n_element_wise * weights)
            critic2_loss_n = torch.mean(critic2_loss_n_element_wise * weights)
            critic_loss_n = critic1_loss_n + critic2_loss_n

            lambda1 = self.hyper_params["LAMBDA1"]
            critic_loss_element_wise += lambda1 * critic_loss_n_element_wise
            critic_loss += lambda1 * critic_loss_n

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        if self.episode_steps % self.hyper_params["POLICY_UPDATE_FREQ"] == 0:
            # train actor
            actions = self.actor(states)
            actor_loss_element_wise = -self.critic1(
                torch.cat((states, actions), dim=-1)
            )
            actor_loss = torch.mean(actor_loss_element_wise * weights)
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            # update target networks
            tau = self.hyper_params["TAU"]
            common_utils.soft_update(self.actor, self.actor_target, tau)
            common_utils.soft_update(self.critic1, self.critic1_target, tau)
            common_utils.soft_update(self.critic2, self.critic2_target, tau)

            # update priorities
            new_priorities = critic_loss_element_wise
            new_priorities += self.hyper_params[
                                  "LAMBDA3"
                              ] * actor_loss_element_wise.pow(2)
            new_priorities += self.hyper_params["PER_EPS"]
            new_priorities = new_priorities.data.cpu().numpy().squeeze()
            new_priorities += eps_d
            self.memory.update_priorities(indices, new_priorities)
        else:
            actor_loss = torch.zeros(1)

        return actor_loss.data, critic1_loss.data, critic2_loss.data

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

    def record(self):
        experiment_recorder = ExperimentLogger(self.args.demo_path, "push", save_trajectories=True)
        experiment_recorder.redirect_output_to_logfile_as_well()
        for i_episode in range(1):
            step = 0
            done = False
            state = self.env.reset()

            while not done:
                # action = trans[1] if self.exe_single_group else trans[1][self.exe_group_num:]
                while True:
                    try:
                        action = input("Enter the target pose for the end-effector:\n").split(' ')
                        action = [float(num) for num in action]
                        next_state, reward, done, _ = self.env.step(action)
                        break
                    except ValueError:
                        continue
                # print(state, action, reward, next_state, done)
                experiment_recorder.add_transition(state, action, reward, next_state, done)
                t_next_state = draw_predicates_on_img(next_state, 36, (640, 480), reward, done)
                cv2.imwrite('../data/fetch-1r/color_img_' + str(i_episode) + '_' + str(step) + '.jpg',
                            t_next_state)
                print("epi and step: ", i_episode, step, action, done, reward)
                step += 1
                state = next_state

        experiment_recorder.save_trajectories(self.hyper_params["SIMPLE_STATES_SIZE"], fps=1, is_separated_file=True, is_save_utility=False, show_predicates=True)

        self.env.close()

    def record_mouse(self):
        experiment_recorder = ExperimentLogger(self.args.demo_path, "push", save_trajectories=True)
        experiment_recorder.redirect_output_to_logfile_as_well()
        for i_episode in range(1):
            step = 0
            done = False
            state = self.env.reset()
            grasp = 0
            self.env.save(step)
            back_step = 0

            while not done:
                step += 1
                # action = trans[1] if self.exe_single_group else trans[1][self.exe_group_num:]
                while True:
                    try:
                        # action = input("Enter the target pose for the end-effector:\n").split(' ')
                        action = self.env.mouse_control(grasp, back_step) #[float(num) for num in action]
                        action = np.concatenate((action[:2], [action[5]]))
                        print(action)
                        action = common_utils.reverse_action(action, self.env.action_space)
                        next_state, reward, done, _ = self.env.step(action)
                        # grasp = action[-1]

                        while True:
                            try:
                                print(
                                    "Press Enter if you are happy with the next state, otherwise Press # steps that you want to go back (1, 2, or ... 10) and then Enter to repeat\n")
                                back_step = input()
                                assert back_step == "" or int(back_step) in range(1, 4), "wrong back_step"
                                back_step = 0 if back_step == "" else int(back_step)
                                break
                            except ValueError:
                                continue

                        good = True if back_step == 0 else False

                        if not good:
                            ####
                            save_trial = False
                            done = True
                            break
                            #### TODO remove

                        # TODO uncomment
                        # print("Restoring to the previous environment step..")
                        # if back_step == 1:
                        # 	self.restore(1)
                        # 	grasp = self._grasps_history[-1]
                        #
                        # else:
                        # 	step, traj = restore(back_step, step, traj)
                        # 	grasp = self._grasps_history[-back_step]
                        # continue

                        break
                    except KeyboardInterrupt:
                        continue

                self.env.save(step)

                # print(state, action, reward, next_state, done)
                experiment_recorder.add_transition(state, action, reward, next_state, done)
                t_next_state = draw_predicates_on_img(next_state, 36, (640, 480), reward, done)
                cv2.imwrite('../data/fetch-1r/color_img_' + str(i_episode) + '_' + str(step) + '.jpg',
                            t_next_state)
                print("epi and step: ", i_episode, step, action, done, reward)
                state = next_state

        experiment_recorder.save_trajectories(self.hyper_params["SIMPLE_STATES_SIZE"], fps=1, is_separated_file=True, is_save_utility=False, show_predicates=True)

        self.env.close()

    # def record_ig(self):
