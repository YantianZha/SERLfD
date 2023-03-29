"""
Prioritized Replay buffer for algorithms.
"""

import random

import numpy as np
import torch

from learning_agents.common.replay_buffer import ReplayBuffer
from learning_agents.common.segment_tree import MinSegmentTree, SumSegmentTree

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PrioritizedReplayBuffer(ReplayBuffer):
    """Create Prioritized Replay buffer.
    Refer to OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Attributes:
        alpha (float): alpha parameter for prioritized replay buffer
        epsilon_d (float): small positive constants to add to the priorities of demo trajs
        epsilon_a (float): small positive constants to add to the priorities of sampled trajs
        tree_idx (int): next index of tree
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight
        _max_priority (float): max priority
    """

    def __init__(
        self,
        buffer_size,
        batch_size=32,
        gamma=0.99,
        n_step=1,
        alpha=0.6,
        epsilon_d=1.0,
        demo=None,
        demo_utility_info=None,
        epsilon_a=0.0
    ):
        """Initialize.
        Args:
            buffer_size (int): size of replay buffer for experience
            batch_size (int): size of a batched sampled from replay buffer for training
            alpha (float): alpha parameter for prioritized replay buffer
            demo: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]]
        """
        super(PrioritizedReplayBuffer, self).__init__(
            buffer_size, batch_size, gamma, n_step, demo, demo_utility_info=demo_utility_info
        )
        assert alpha >= 0
        self.alpha = alpha
        self.epsilon_d = epsilon_d
        self.epsilon_a = epsilon_a
        self.tree_idx = 0

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.buffer_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        self._max_priority = 1.0

        # for init priority of demo
        self.tree_idx = self.demo_size
        for i in range(self.demo_size):
            self.sum_tree[i] = self._max_priority ** self.alpha
            self.min_tree[i] = self._max_priority ** self.alpha

    def add(self, transition, utility_info=None):
        """
        Add experience and priority.
        transition: Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]
        """
        n_step_transition = super().add(transition, utility_info=utility_info)
        if n_step_transition:
            self.sum_tree[self.tree_idx] = self._max_priority ** self.alpha
            self.min_tree[self.tree_idx] = self._max_priority ** self.alpha

            self.tree_idx += 1
            if self.tree_idx % self.buffer_size == 0:
                self.tree_idx = self.demo_size

        return n_step_transition

    def _sample_proportional(self, batch_size):
        """Sample indices based on proportional."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
        return indices

    def sample(self, beta=0.4, is_to_tensor=True):
        """
        Sample a batch of experiences.
        :return: Tuple[torch.Tensor, ...]
        """
        n_sample = self.batch_size
        if len(self) < n_sample:
            n_sample = len(self)
        assert beta > 0

        indices = self._sample_proportional(n_sample)

        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        # calculate weights
        weights_, eps_d = [], []
        for i in indices:
            eps_d.append(self.epsilon_d if i < self.demo_size else self.epsilon_a)
            p_sample = self.sum_tree[i] / self.sum_tree.sum()
            weight = (p_sample * len(self)) ** (-beta)
            weights_.append(weight / max_weight)

        weights = np.array(weights_)
        eps_d = np.array(eps_d)

        # noinspection PyArgumentList
        if is_to_tensor:
            weights = torch.FloatTensor(weights.reshape(-1, 1)).to(device)
            if torch.cuda.is_available():
                weights = weights.cuda(non_blocking=True)
        else:
            weights = weights.reshape(-1, 1)

        states, actions, rewards, next_states, dones = super().sample(indices, is_to_tensor=is_to_tensor)

        return states, actions, rewards, next_states, dones, weights, indices, eps_d

    def update_priorities(self, indices, priorities):
        """
        Update priorities of sampled transitions.
        priorities (np.ndarray)
        """
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0, "[ERROR] expected positive priority but get this value: " + str(priority)
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self._max_priority = max(self._max_priority, priority)
