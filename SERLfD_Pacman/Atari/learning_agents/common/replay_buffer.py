from collections import deque

import numpy as np
import torch

from learning_agents.utils.utils import UtilityTransition
from utils.trajectory_utils import get_n_step_info, TRAJECTORY_INDEX, read_expert_demo, \
    demo_discrete_actions_to_one_hot, get_flatten_trajectories, stack_frames_in_traj, read_expert_demo_utility_info, \
    get_flatten_indexed_utility_info

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    Attributes:
        obs_buf (np.ndarray): observations
        acts_buf (np.ndarray): actions
        rewards_buf (np.ndarray): rewards
        next_obs_buf (np.ndarray): next observations
        done_buf (np.ndarray): dones
        n_step_buffer (deque): recent n transitions
        n_step (int): step size for n-step transition
        gamma (float): discount factor
        buffer_size (int): size of buffers
        batch_size (int): batch size for training
        demo_size (int): size of demo transitions
        length (int): amount of memory filled
        idx (int): memory index to add the next incoming transition
    """

    def __init__(self, buffer_size, batch_size=32, gamma=0.99, n_step=1, demo=None, demo_utility_info=None):
        """Initialize a ReplayBuffer object.
        Args:
            buffer_size (int): size of replay buffer for experience
            batch_size (int): size of a batched sampled from replay buffer for training
            gamma (float): discount factor
            n_step (int): step size for n-step transition
            demo (list): transitions of human play
        """
        assert 0 < batch_size <= buffer_size
        assert 0.0 <= gamma <= 1.0
        assert 1 <= n_step <= buffer_size

        # basic experience
        self.obs_buf = None
        self.acts_buf = None
        self.rewards_buf = None
        self.next_obs_buf = None
        self.done_buf = None
        # utility-related experience
        self.predicate_values_buf = None
        self.next_predicate_values_buf = None

        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step_utility_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.demo_size = len(demo) if demo else 0
        self.demo = demo
        self.length = 0
        self.idx = self.demo_size   # idx makes sure demo is always in demo

        # use self.demo[0] to make sure demo don't have empty tuple list [()]
        if self.demo and self.demo[0]:
            self.buffer_size += self.demo_size
            self.length += self.demo_size
            for idx, d in enumerate(self.demo):
                state, action, reward, next_state, done = d

                if idx == 0:
                    self._initialize_buffers(state, action)
                self.obs_buf[idx] = state
                self.acts_buf[idx] = np.array(action)
                self.rewards_buf[idx] = reward
                self.next_obs_buf[idx] = next_state
                self.done_buf[idx] = done

                if demo_utility_info is not None:
                    self._add_utility_transition(idx, demo_utility_info[idx])

    def _initialize_buffers(self, state, action):
        """
        Initialize buffers for state, action, rewards, next_state, done.
        state: np.ndarray
        action: np.ndarray
        """
        # init observation buffer
        self.obs_buf = np.zeros(
            [self.buffer_size] + list(state.shape), dtype=state.dtype
        )
        # init action buffer
        action = (np.array(action).astype(np.int64) if isinstance(action, int) else action)
        self.acts_buf = np.zeros(
            [self.buffer_size] + list(action.shape), dtype=action.dtype
        )
        # init reward buffer
        self.rewards_buf = np.zeros([self.buffer_size], dtype=float)
        # init next observation buffer
        self.next_obs_buf = np.zeros(
            [self.buffer_size] + list(state.shape), dtype=state.dtype
        )
        # init done buffer
        self.done_buf = np.zeros([self.buffer_size], dtype=float)

    def __len__(self):
        """Return the current size of internal memory."""
        return self.length

    def add(self, transition, utility_info=None):
        """
        Add a new experience to memory.
        If the buffer is empty, it is respectively initialized by size of arguments.
        transition: Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]

        :return: Tuple[Any, ...]
        """
        contain_utility_info = True if utility_info is not None else False
        self.n_step_buffer.append(transition)
        if contain_utility_info:
            self.n_step_utility_buffer.append(utility_info)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()

        if self.length == 0:
            state, action = transition[:2]
            self._initialize_buffers(state, action)

        # add a multi step transition
        if contain_utility_info:
            reward, next_state, done, n_step_util_info = get_n_step_info(self.n_step_buffer, self.gamma, self.n_step_utility_buffer)
        else:
            reward, next_state, done = get_n_step_info(self.n_step_buffer, self.gamma)
        curr_state, action = self.n_step_buffer[0][:2]
        self.obs_buf[self.idx] = curr_state
        self.acts_buf[self.idx] = action
        self.rewards_buf[self.idx] = reward
        self.next_obs_buf[self.idx] = next_state
        self.done_buf[self.idx] = done

        if len(self.n_step_utility_buffer) > 0 and contain_utility_info:
            # noinspection PyUnboundLocalVariable
            n_step_util_transition = UtilityTransition(self.n_step_utility_buffer[0][TRAJECTORY_INDEX.PREDICATE_VALUES.value],
                                                       n_step_util_info[TRAJECTORY_INDEX.NEXT_PREDICATE_VALUES.value])
            self._add_utility_transition(self.idx, n_step_util_transition.util_transition)

        self.idx += 1
        self.idx = self.demo_size if self.idx % self.buffer_size == 0 else self.idx
        self.length = min(self.length + 1, self.buffer_size)

        # return a single step transition to insert to replay buffer
        if utility_info is not None:
            return self.n_step_buffer[0], self.n_step_utility_buffer[0]
        else:
            return self.n_step_buffer[0]

    def _add_utility_transition(self, idx, utility_transition):
        predicate_value = utility_transition[TRAJECTORY_INDEX.PREDICATE_VALUES.value]
        if predicate_value is not None:
            if self.predicate_values_buf is None:
                self.predicate_values_buf = np.array([predicate_value for _ in range(self.buffer_size)])
            self.predicate_values_buf[idx] = predicate_value

        next_predicate_value = utility_transition[TRAJECTORY_INDEX.NEXT_PREDICATE_VALUES.value]
        if next_predicate_value is not None:
            if self.next_predicate_values_buf is None:
                self.next_predicate_values_buf = np.array([next_predicate_value for _ in range(self.buffer_size)])
            self.next_predicate_values_buf[idx] = next_predicate_value

    def extend(self, transitions, utility_info=None):
        """
        Add experiences to memory.
        transitions (List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]])
        """
        for i, transition in enumerate(transitions):
            utility_transition = utility_info[i] if utility_info is not None else None
            self.add(transition, utility_info=utility_transition)

    def sample(self, indices=None, batch_size=None, is_to_tensor=True):
        """
        Randomly sample a batch of experiences from memory.
        indices: List[int] = None)
        :return: Tuple[torch.Tensor, ...]
        """
        if batch_size is None:
            if indices is not None:
                batch_size = len(indices)
            else:
                batch_size = self.batch_size

        assert len(self) >= batch_size

        if indices is None:
            indices = np.random.choice(len(self), size=batch_size, replace=False)

        if is_to_tensor:
            states = torch.FloatTensor(self.obs_buf[indices]).to(device)
            actions = torch.FloatTensor(self.acts_buf[indices]).to(device)
            rewards = torch.FloatTensor(self.rewards_buf[indices].reshape(-1, 1)).to(device)
            next_states = torch.FloatTensor(self.next_obs_buf[indices]).to(device)
            dones = torch.FloatTensor(self.done_buf[indices].reshape(-1, 1)).to(device)

            if torch.cuda.is_available():
                states = states.cuda(non_blocking=True)
                actions = actions.cuda(non_blocking=True)
                rewards = rewards.cuda(non_blocking=True)
                next_states = next_states.cuda(non_blocking=True)
                dones = dones.cuda(non_blocking=True)
        else:
            states = self.obs_buf[indices]
            actions = self.acts_buf[indices]
            rewards = self.rewards_buf[indices].reshape(-1, 1)
            next_states = self.next_obs_buf[indices]
            dones = self.done_buf[indices].reshape(-1, 1)

        return states, actions, rewards, next_states, dones

    def get_utility_info(self, indices):
        predicates = self.predicate_values_buf[indices] if self.predicate_values_buf is not None else None
        next_predicates = self.next_predicate_values_buf[indices] if self.next_predicate_values_buf is not None else None

        return predicates, next_predicates


def load_stacked_demos_and_utility(demo_path, demo_util_path,
                                   is_discrete_action, discrete_to_one_hot, action_dim,
                                   demo_preprocessor, frame_stack):
    """
    read demo trajectories and related utility info from the specified path
    """
    # load basic trajectories
    demo_trajectories = read_expert_demo(demo_path)
    # it's not necessary to convert int action to np.array since the replay buffer can handle it
    if is_discrete_action and discrete_to_one_hot:
        demo_discrete_actions_to_one_hot(demo_trajectories, action_dim)
    flatten_demo_traj = get_flatten_trajectories(demo_trajectories)
    stacked_demo = stack_frames_in_traj(flatten_demo_traj, demo_preprocessor, frame_stack)

    # load utility info
    demo_utility_indexed_info = read_expert_demo_utility_info(demo_util_path, is_by_keys=True)
    flatten_indexed_utility_info = get_flatten_indexed_utility_info(demo_utility_indexed_info)
    demo_predicates = flatten_indexed_utility_info[TRAJECTORY_INDEX.PREDICATE_VALUES.value]
    demo_next_predicates = flatten_indexed_utility_info[TRAJECTORY_INDEX.NEXT_PREDICATE_VALUES.value]
    assert len(demo_predicates) == len(stacked_demo), 'Number of demo transitions and predicates should be the same'
    # process predicates
    demo_utility_info = []
    for i, demo_predicate in enumerate(demo_predicates):
        utility_transition = UtilityTransition(demo_predicates[i], demo_next_predicates[i])
        demo_utility_info.append(utility_transition.util_transition)

    return stacked_demo, demo_utility_info
