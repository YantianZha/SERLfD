import pickle
import time

import numpy as np
import torch
import torch.nn as nn

from learning_agents.rl.ddpg.ddpg import DDPGAgent
from learning_agents.common.priortized_replay_buffer import PrioritizedReplayBuffer
from learning_agents.common.replay_buffer import ReplayBuffer
from utils.trajectory_utils import get_n_step_info_from_traj
from learning_agents.common import common_utils
from utils.trajectory_utils import read_expert_demo, get_flatten_trajectories, demo_discrete_actions_to_one_hot

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPGfDAgent(DDPGAgent):
    """
    Actor-Critic interacting with environment.
    Attributes:
        memory (PrioritizedReplayBuffer): replay memory
        per_beta (float): beta parameter for prioritized replay buffer
        use_n_step (bool): whether or not to use n-step returns
    """

    # pylint: disable=attribute-defined-outside-init
    def _initialize(self):
        """Initialize non-common things."""
        self.per_beta = self.hyper_params.per_beta
        self.use_n_step = self.hyper_params.n_step > 1

        if not self.args.test:
            # load demo replay memory
            demos = self._load_demos()

            if self.use_n_step:
                demos, demos_n_step = get_n_step_info_from_traj(
                    demos, self.hyper_params.n_step, self.hyper_params.gamma
                )

                # replay memory for multi-steps
                self.memory_n = ReplayBuffer(
                    buffer_size=self.hyper_params.buffer_size,
                    n_step=self.hyper_params.n_step,
                    gamma=self.hyper_params.gamma,
                    demo=demos_n_step,
                )

            # replay memory for a single step
            self.memory = PrioritizedReplayBuffer(
                self.hyper_params.buffer_size,
                self.hyper_params.batch_size,
                demo=demos,
                alpha=self.hyper_params.per_alpha,
                epsilon_d=self.hyper_params.per_eps_demo,
            )

    def _load_demos(self):
        demos = read_expert_demo(self.args.demo_path)
        if self.is_discrete:
            demo_discrete_actions_to_one_hot(demos, self.action_dim)
        return get_flatten_trajectories(demos)

    def _add_transition_to_memory(self, transition):
        """Add 1 step and n step transitions to memory."""
        # add n-step transition
        if self.use_n_step:
            transition = self.memory_n.add(transition)

        # when the n-step memory is not ready, memory_n.add will return ()
        # add a single step transition (check if transition is empty tuple)
        if transition:
            self.memory.add(transition)

    def _get_critic_loss(self, experiences, gamma):
        """
        Return element-wise critic loss.
        """
        states, actions, rewards, next_states, dones = experiences[:5]

        masks = 1 - dones
        next_actions = self.actor_target(next_states)
        next_states_actions = (next_states, next_actions)
        next_values = self.critic_target(next_states_actions)
        curr_returns = rewards + gamma * next_values * masks
        curr_returns = curr_returns.to(device).detach()

        # train critic
        values = self.critic((states, actions))
        critic_loss_element_wise = (values - curr_returns).pow(2)

        return critic_loss_element_wise

    def update_model(self):
        """Train the model after each episode."""
        experiences_1_step = self.memory.sample(self.per_beta)
        states, actions = experiences_1_step[:2]
        weights, indices, eps_d = experiences_1_step[-3:]
        gamma = self.hyper_params.gamma

        # train critic
        gradient_clip_ac = self.hyper_params.gradient_clip_ac
        gradient_clip_cr = self.hyper_params.gradient_clip_cr

        critic_loss_element_wise = self._get_critic_loss(experiences_1_step, gamma)
        critic_loss = torch.mean(critic_loss_element_wise * weights)

        if self.use_n_step:
            experiences_n_step = self.memory_n.sample(indices)
            gamma = gamma ** self.hyper_params.n_step
            critic_loss_n_element_wise = self._get_critic_loss(experiences_n_step, gamma)
            # to update loss and priorities
            critic_loss_element_wise += (
                critic_loss_n_element_wise * self.hyper_params.lambda1
            )
            critic_loss = torch.mean(critic_loss_element_wise * weights)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), gradient_clip_cr)
        self.critic_optim.step()

        # train actor
        actions = self.actor(states)
        actor_loss_element_wise = -self.critic((states, actions))
        actor_loss = torch.mean(actor_loss_element_wise * weights)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), gradient_clip_ac)
        self.actor_optim.step()

        # update target networks
        common_utils.soft_update(self.actor, self.actor_target, self.hyper_params.tau)
        common_utils.soft_update(self.critic, self.critic_target, self.hyper_params.tau)

        # update priorities
        new_priorities = critic_loss_element_wise
        new_priorities += self.hyper_params.lambda3 * actor_loss_element_wise.pow(2)
        new_priorities += self.hyper_params.per_eps
        new_priorities = new_priorities.data.cpu().numpy().squeeze()
        new_priorities += eps_d
        self.memory.update_priorities(indices, new_priorities)

        # increase beta
        fraction = min(float(self.i_episode) / self.args.episode_num, 1.0)
        self.per_beta = self.per_beta + fraction * (1.0 - self.per_beta)

        return actor_loss.item(), critic_loss.item()

    def pretrain(self):
        """
        Pretraining steps.
        """
        pretrain_loss = list()
        pretrain_step = self.hyper_params.pretrain_step
        print("[INFO] Pre-Train %d step." % pretrain_step)
        for i_step in range(1, pretrain_step + 1):
            t_begin = time.time()
            loss = self.update_model()
            t_end = time.time()
            pretrain_loss.append(loss)  # for logging

            # logging
            if i_step == 1 or i_step % 100 == 0:
                avg_loss = np.vstack(pretrain_loss).mean(axis=0)
                pretrain_loss.clear()
                log_value = (0, avg_loss, 0, t_end - t_begin)
                self.write_log(log_value)
        print("[INFO] Pre-Train Complete!\n")
