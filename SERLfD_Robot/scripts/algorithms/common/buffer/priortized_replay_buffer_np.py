# -*- coding: utf-8 -*-
"""Prioritized Replay buffer for baselines.

- Author: Kh Kim
- Contact: kh.kim@medipixel.io
- Paper: https://arxiv.org/pdf/1511.05952.pdf
         https://arxiv.org/pdf/1707.08817.pdf
"""

import random

import numpy as np
import torch

from algorithms.common.buffer.replay_buffer import ReplayBuffer
from algorithms.common.buffer.segment_tree import MinSegmentTree, SumSegmentTree

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PrioritizedReplayBuffer(ReplayBuffer):
    """Create Prioritized Replay buffer.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

    Attributes:
        buffer_size (int): size of replay buffer for experience
        alpha (float): alpha parameter for prioritized replay buffer
        tree_idx (int): next index of tree
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight
        _max_priority (float): max priority

        """

    def __init__(self, buffer_size, batch_size, alpha=0.6):
        """Initialization.

        Args:
            buffer_size (int): size of replay buffer for experience
            batch_size (int): size of a batched sampled from replay buffer for training
            alpha (float): alpha parameter for prioritized replay buffer

        """
        super(PrioritizedReplayBuffer, self).__init__(buffer_size, batch_size)
        assert alpha >= 0
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.tree_idx = 0

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.buffer_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        self._max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        """Add experience and priority."""
        idx = self.tree_idx
        self.tree_idx = (self.tree_idx + 1) % self.buffer_size
        super(PrioritizedReplayBuffer, self).add(
            state, action, reward, next_state, done
        )

        self.sum_tree[idx] = self._max_priority ** self.alpha
        self.min_tree[idx] = self._max_priority ** self.alpha

    def extend(self, transitions):
        """Add experiences to memory."""
        raise NotImplementedError

    def _sample_proportional(self, batch_size):
        """Sample indices based on proportional."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self.buffer) - 1)
        segment = p_total / batch_size
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
        return indices

    def sample(self, beta=0.4):
        """Sample a batch of experiences."""
        assert beta > 0

        indices = self._sample_proportional(self.batch_size)
        states, actions, rewards, next_states, dones, weights = [], [], [], [], [], []

        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self.buffer)) ** (-beta)

        for i in indices:
            s, a, r, n_s, d = self.buffer[i]
            states.append(np.array(s, copy=False))
            actions.append(np.array(a, copy=False))
            rewards.append(np.array(r, copy=False))
            next_states.append(np.array(n_s, copy=False))
            dones.append(np.array(float(d), copy=False))

            # calculate weights
            p_sample = self.sum_tree[i] / self.sum_tree.sum()
            weight = (p_sample * len(self.buffer)) ** (-beta)
            weights.append(weight / max_weight)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards).reshape(-1, 1)
        next_states = np.array(next_states)
        dones = np.array(dones).reshape(-1, 1)
        weights = np.array(weights).reshape(-1, 1)

        experiences = (states, actions, rewards, next_states, dones, weights, indices)

        return experiences

    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self.buffer)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self._max_priority = max(self._max_priority, priority)


class PrioritizedReplayBufferfD(PrioritizedReplayBuffer):
    """Create Prioritized Replay buffer with demo.
    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Attributes:
        demo (list): list of demo replay buffer
        buffer_size (int): size of replay buffer for experience
        demo_size (int): size of replay buffer for demonstration
        total_size (int): sum of demo size and number of samples of experience
        epsilon_d (float) : epsilon_d parameter to update priority using demo
        """

    def __init__(self, buffer_size, batch_size, demo=[], alpha=0.6, epsilon_d=1.0):
        """Initialization.
        Args:
            buffer_size (int): size of replay buffer for experience
            batch_size (int): size of a batched sampled from replay buffer for training
            demo (list): demonstration
            alpha (float): alpha parameter for prioritized replay buffer
            epsilon_d (float) : epsilon_d parameter to update priority using demo
        """
        super(PrioritizedReplayBufferfD, self).__init__(buffer_size, batch_size, alpha)
        self.demo = demo
        self.demo_size = len(demo)
        self.total_size = self.demo_size + len(self.buffer)
        self.epsilon_d = epsilon_d

        # for init priority of demo
        for _ in range(self.demo_size):
            self.sum_tree[self.tree_idx] = self._max_priority ** self.alpha
            self.min_tree[self.tree_idx] = self._max_priority ** self.alpha
            self.tree_idx += 1

    def add(self, state, action, reward, next_state, done):
        """Add experience and priority."""
        idx = self.tree_idx
        # buffer is full
        if (self.tree_idx + 1) % (self.buffer_size + self.demo_size) == 0:
            self.tree_idx = self.demo_size
        else:
            self.tree_idx = self.tree_idx + 1
        super(PrioritizedReplayBuffer, self).add(
            state, action, reward, next_state, done
        )

        self.sum_tree[idx] = self._max_priority ** self.alpha
        self.min_tree[idx] = self._max_priority ** self.alpha

        # update current total size
        self.total_size = self.demo_size + len(self.buffer)

    def sample(self, beta=0.4):
        """Sample a batch of experiences."""
        assert beta > 0

        indices = self._sample_proportional(self.batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        weights, eps_d = [], []

        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * self.total_size) ** (-beta)

        for i in indices:
            # sample from buffer
            if i < self.demo_size:
                s, a, r, n_s, d = self.demo[i]
                eps_d.append(self.epsilon_d)
            else:
                s, a, r, n_s, d = self.buffer[i - self.demo_size]
                eps_d.append(0.0)

            # append transition info
            states.append(np.array(s, copy=False))
            actions.append(np.array(a, copy=False))
            rewards.append(np.array(r, copy=False))
            next_states.append(np.array(n_s, copy=False))
            dones.append(np.array(float(d), copy=False))

            # calculate weights
            p_sample = self.sum_tree[i] / self.sum_tree.sum()
            weight = (p_sample * self.total_size) ** (-beta)
            weights.append(weight / max_weight)

        states_ = np.array(states)
        actions_ = np.array(actions)
        rewards_ = np.array(rewards).reshape(-1, 1)
        next_states_ = np.array(next_states)
        dones_ = np.array(dones).reshape(-1, 1)
        weights_ = np.array(weights).reshape(-1, 1)
        eps_d = np.array(eps_d)

        # if torch.cuda.is_available():
        #     states_ = states_.cuda(non_blocking=True)
        #     actions_ = actions_.cuda(non_blocking=True)
        #     rewards_ = rewards_.cuda(non_blocking=True)
        #     next_states_ = next_states_.cuda(non_blocking=True)
        #     dones_ = dones_.cuda(non_blocking=True)
        #     weights_ = weights_.cuda(non_blocking=True)

        experiences = (
            states_,
            actions_,
            rewards_,
            next_states_,
            dones_,
            weights_,
            indices,
            eps_d,
        )

        return experiences

    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < self.total_size

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self._max_priority = max(self._max_priority, priority)
