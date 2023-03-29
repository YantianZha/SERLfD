import argparse
import copy
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import gym
from torch.nn.utils import clip_grad_norm_

from learning_agents.agent import Agent
from learning_agents.architectures.cnn import Conv2d_MLP_Model
from learning_agents.common.replay_buffer import ReplayBuffer, load_stacked_demos_and_utility
from learning_agents.common.trajectory_buffer import TrajectoryBuffer
import learning_agents.common.common_utils as common_utils
from learning_agents.architectures.mlp import MLP
from utils.trajectory_utils import split_fixed_length_indexed_traj
from learning_agents.utils.utils import ConfigDict, UtilityTransition
from utils.trajectory_utils import TRAJECTORY_INDEX

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOG_REG = 1e-8
MAX_ENERGY = 50


# noinspection DuplicatedCode
class SEGL_Single_Update_Step(Agent):
    """
    Self-Explanation Guided Learning on single state and update controller_policy at every time step

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
        demo_preprocessor: preprocessor of frames in demonstrations
        ####################### attributes of the RL (controller_policy) ##############################
        policy_network (Agent): the pre-defined controller_policy network
    """

    def __init__(self, env, policy, args, log_cfg, hyper_params, network_cfg, optim_cfg, demo_preprocessor, encoder=None, logger=None):
        """
        Initialize.
        env: gym.Env,
        args: argparse.Namespace,
        log_cfg: ConfigDict,
        hyper_params: ConfigDict,
        network_cfg: ConfigDict,
        optim_cfg: ConfigDict,
        controller_policy (Agent): the controller_policy module
        demo_preprocessor: preprocessor of frames in demonstrations
        encoder(nn): if encoder is not None, the irl/explainer will use the encoder to preprocess the state
        logger (ExperimentLogger): the experiment logger
        """
        Agent.__init__(self, env, args, log_cfg)

        self.curr_state = np.zeros((1,))
        self.total_step = 0
        self.i_episode = 0
        self.avg_scores_window = deque(maxlen=self.args.avg_score_window)
        self.eval_scores_window = deque(maxlen=self.args.eval_score_window)
        self.i_iteration = 0
        self.is_doing_pretrain = False

        self.hyper_params = hyper_params
        self.network_cfg = network_cfg
        self.optim_cfg = optim_cfg
        self.encoder = encoder
        self.demo_preprocessor = demo_preprocessor
        self.logger = logger

        # get state space info
        if encoder is None:
            self.state_dim = self.env.observation_space.shape[0]
            self.state_channel = self.hyper_params.frame_stack
        else:
            self.state_dim = self.encoder.output_dim
            self.state_channel = self.encoder.output_channel

        # get action space info
        self.is_discrete = False
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.action_dim = self.env.action_space.n
            self.is_discrete = True
        else:
            self.action_dim = self.env.action_space.shape[0]

        # get predicate info
        self.predicate_keys = env.get_predicate_keys()

        # set the controller_policy module
        self.policy = policy
        # init memory
        self._initialize()

        # initialize the networks
        self._init_network()

    # pylint: disable=attribute-defined-outside-init
    def _initialize(self):
        """ Initialize non-common things """
        # human demonstration buffer
        if not self.args.test:
            demos, utility_info = self._load_demos()
            # human demonstration buffer
            self.good_buffer = ReplayBuffer(
                self.hyper_params.buffer_size,
                batch_size=int(self.hyper_params.batch_size/2)
            )
            self.good_buffer.extend(transitions=demos, utility_info=utility_info)
            # sampled trajectories buffer (In Justin Fu's implementation, they don't reuse past sample)
            self.bad_buffer = ReplayBuffer(
                self.hyper_params.buffer_size,
                batch_size=int(self.hyper_params.batch_size/2)
            )

    def _load_demos(self):
        return load_stacked_demos_and_utility(self.args.demo_path, self.args.demo_util_path,
                                              self.is_discrete, self.hyper_params.discrete_to_one_hot, self.action_dim,
                                              self.demo_preprocessor, self.hyper_params.frame_stack)

    def _set_to_default_MLP(self):
        n_predicate_key = len(self.predicate_keys)
        if self.hyper_params.predicate_one_hot:
            fc_output_size = n_predicate_key * 2 + 1 if self.hyper_params.bias_in_predicate else n_predicate_key * 2
        else:
            fc_output_size = n_predicate_key + 1 if self.hyper_params.bias_in_predicate else n_predicate_key

        # the explainer model
        self.explainer = MLP(
            input_size=self.state_dim,
            output_size=fc_output_size,
            hidden_sizes=self.network_cfg.hidden_sizes_irl,
            hidden_activation=torch.tanh
        ).to(device)

    def _set_to_default_CNN(self):
        n_predicate_key = len(self.predicate_keys)
        if self.hyper_params.predicate_one_hot:
            fc_output_size = n_predicate_key * 2 + 1 if self.hyper_params.bias_in_predicate else n_predicate_key * 2
        else:
            fc_output_size = n_predicate_key + 1 if self.hyper_params.bias_in_predicate else n_predicate_key

        self.explainer = Conv2d_MLP_Model(input_channels=self.state_channel,
                                          fc_input_size=self.network_cfg.fc_input_size,
                                          fc_output_size=fc_output_size,
                                          nonlinearity=self.network_cfg.nonlinearity,
                                          channels=self.network_cfg.channels,
                                          kernel_sizes=self.network_cfg.kernel_sizes,
                                          strides=self.network_cfg.strides,
                                          paddings=self.network_cfg.paddings,
                                          fc_hidden_sizes=self.network_cfg.fc_hidden_sizes,
                                          fc_hidden_activation=self.network_cfg.fc_hidden_activation).to(device)

    def _init_network(self):
        """ Initialize the explainer network """
        if self.network_cfg.use_cnn:
            self._set_to_default_CNN()
        else:
            self._set_to_default_MLP()

        # define the optimizer
        self.explainer_optim = optim.Adam(
            self.explainer.parameters(),
            lr=self.optim_cfg.lr_explainer,
            weight_decay=self.optim_cfg.weight_decay
        )
        # init network from file
        self._init_from_file()

    def _init_from_file(self):
        # load the optimizer and model parameters
        if self.args.explainer_load_from is not None:
            self.load_params(self.args.explainer_load_from)

    def _preprocess_state(self, state):
        """Preprocess state so that actor selects an action."""
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(device)
        # if state is a single state, we unsqueeze it
        if len(state.size()) == 3:
            state = state.unsqueeze(0)
        if self.encoder is not None:
            state = self.encoder(state)

        return state

    def to_predicate_vector(self, predicate_value):
        """
        Convert dict representation of predicate values to list (vector) representation of predicate values
        :param predicate_value: a dict storing predicate values
        :return (np.ndarray): predicate vector
        """
        add_bias = self.hyper_params.bias_in_predicate
        predicate_vector = [0.0 for _ in predicate_value]
        if self.hyper_params.predicate_one_hot:
            predicate_vector = predicate_vector + predicate_vector
        # expand one 1 element for bias
        if add_bias:
            predicate_vector = predicate_vector + [1.0]
        for idx, key in enumerate(self.predicate_keys):
            if self.hyper_params.predicate_one_hot:
                if predicate_value[key] == 1:
                    predicate_vector[idx*2+1] = 1
                else:
                    predicate_vector[idx*2] = 1
            else:
                predicate_vector[idx] = predicate_value[key]
        return np.array(predicate_vector, dtype=np.float64)

    def to_predicate_vectors(self, predicate_values):
        return np.array([self.to_predicate_vector(predicate_value)
                         for predicate_value in predicate_values])

    # noinspection PyMethodMayBeStatic
    def _get_manual_shaping(self, state, predicate_values, next_state, next_predicate_values):
        """
        For comparison
        """
        shaping_rewards = np.zeros(shape=(len(predicate_values), 1))
        for idx, predicate_value in enumerate(predicate_values):
            shaping_reward = 0
            next_predicate_value = next_predicate_values[idx]
            if predicate_value == {'is_eat_capsule': -1, 'is_ghost_nearby': -1} and \
                    next_predicate_value == {'is_eat_capsule': 1, 'is_ghost_nearby': -1}:
                shaping_reward = -0.5
            if predicate_value == {'is_eat_capsule': -1, 'is_ghost_nearby': 1} and \
                    next_predicate_value == {'is_eat_capsule': 1, 'is_ghost_nearby': 1}:
                shaping_reward = 0.5
            if predicate_value == {'is_eat_capsule': 1, 'is_ghost_nearby': 1} and \
                    next_predicate_value == {'is_eat_capsule': 1, 'is_ghost_nearby': -1}:
                shaping_reward = -0.5
            shaping_rewards[idx][0] = shaping_reward

        return shaping_rewards

    def get_shaping_reward(self, states, predicate_values, next_state, next_predicate_values):
        """
        Return shaping reward given the transition. The shaping reward is the biggest utility value change
            Note that shaping reward is given only when there is a predicate value change
        """
        if self.is_doing_pretrain or self.args.no_shaping:
            return 0
        if self.args.manual_shaping:
            return self._get_manual_shaping(states, predicate_values, next_state, next_predicate_values)

        predicate_vectors = self.to_predicate_vectors(predicate_values)
        states_util_values = self.get_states_utility(states, predicate_vectors)
        next_predicate_vectors = self.to_predicate_vectors(next_predicate_values)
        next_state_util_values = self.get_states_utility(next_state, next_predicate_vectors)

        shaping_rewards = next_state_util_values - states_util_values

        if self.args.print_shaping:
            for idx, _ in enumerate(shaping_rewards):
                shaping_reward = shaping_rewards[idx][0]
                if shaping_reward != 0:
                    print('\n[Explainer INFO] shaping reward: ', shaping_reward,
                          '\nold predicate vector: ', predicate_vectors[idx],
                          '\nnew predicate vector: ', next_predicate_vectors[idx],
                          '\nutility value: ', states_util_values[idx],
                          '\nnext utility: ', next_state_util_values[idx])

        # clip the shaping reward for stabilization consideration
        shaping_rewards = self.hyper_params.shaping_reward_weight * np.clip(shaping_rewards,
                                                                            self.hyper_params.shaping_reward_clip[0],
                                                                            self.hyper_params.shaping_reward_clip[1])
        if self.hyper_params.negative_reward_only:
            shaping_rewards = np.clip(shaping_rewards,
                                      self.hyper_params.shaping_reward_clip[0],
                                      0)
        return shaping_rewards

    def get_states_utility(self, states, predicate_vectors):
        """
        Compute single state utility based on predicate values and predicate weights
        :param states: (np.ndarray) the states (can be a single state or a nested list: [[state, state, ...], [state, state, ...], ...]
        :param predicate_vectors: np.array storing the predicate vectors
        """
        # if nested list of states (traj), sum up the utility values of states in the same trajectory
        energy = self.eval(states, predicate_vectors)
        utility_values = (-energy)
        return utility_values.detach().cpu().numpy()

    def get_utility_values_vectors(self, states, predicate_values):
        n_states = len(predicate_values)
        util_vector = np.zeros(shape=(n_states, len(self.predicate_keys)))
        if self.is_doing_pretrain:
            return util_vector

        states = self._preprocess_state(states)
        predicate_vectors = self.to_predicate_vectors(predicate_values)

        neg_predicate_weights = self.explainer(states)
        if torch.isnan(neg_predicate_weights).any():
            print('[ERROR] Predicate weight contains nan: ', neg_predicate_weights)
        predicate_weights = (-neg_predicate_weights).detach().cpu().numpy()
        predicate_weights = np.clip(predicate_weights, -100, 100)

        for i in range(n_states):
            for j, _ in enumerate(self.predicate_keys):
                idx = j
                if self.hyper_params.predicate_one_hot:
                    idx = 2*j
                    util_vector[i][j] = (predicate_vectors[i][idx] * predicate_weights[i][idx] +
                                         predicate_vectors[i][idx+1] * predicate_weights[i][idx+1])
                else:
                    util_vector[i][j] = predicate_vectors[i][idx] * predicate_weights[i][idx]

        return util_vector

    def get_utility_values_dicts(self, states, predicate_values):
        """
        Compute the utility value of each predicate and return a dict storing utility values
        """
        util_vectors = self.get_utility_values_vectors(states, predicate_values)

        util_value_dict = []
        for i in range(len(util_vectors)):
            util_dict = {}
            for idx, key in enumerate(self.predicate_keys):
                util_dict[key] = util_vectors[i][idx]
            util_value_dict.append(util_dict)
        return util_value_dict

    def eval(self, states, predicate_vectors):
        """
        Compute single state utility value based on predicate values and predicate weights
        :param states: (np.ndarray) the states (can be a single state or a nested list: [[state, state, ...], [state, state, ...], ...]
        :param predicate_vectors: torch.Tensor storing the predicate values
        """
        states = self._preprocess_state(states)
        neg_predicate_weights = self.explainer(states)
        
        if torch.isnan(neg_predicate_weights).any():
            print('[ERROR] Predicate weight contains nan: ', neg_predicate_weights)
        predicate_vectors = torch.from_numpy(predicate_vectors).type(torch.FloatTensor).to(device)
        negative_utility_values = torch.sum(neg_predicate_weights * predicate_vectors, dim=-1, keepdim=True)

        return negative_utility_values

    def eval_version2(self, states, predicate_vectors, rewards, next_states, next_predicate_vectors):
        negative_utility_values = self.eval(states, predicate_vectors)
        next_negative_utility_values = self.eval(next_states, next_predicate_vectors)
        return -rewards - next_negative_utility_values + negative_utility_values

    def augment_states(self, states, predicate_values):
        """ augment the original observation with utility map """
        if self.is_doing_pretrain:
            return states
        else:
            return states

    def sample_trajectories(self):
        """
        Use the controller_policy network to sample a list of trajectories
        :return (list of dict): list of trajectories (each trajectory is a dict)
        """
        max_traj_length = self.hyper_params.max_traj_length
        cnt_steps = 0
        while cnt_steps < self.hyper_params.n_samples_epoch:
            state = self.env.reset()
            state = np.squeeze(state, axis=0)
            states_queue = deque(maxlen=self.hyper_params.frame_stack)
            states_queue.extend([state for _ in range(self.hyper_params.frame_stack)])

            done = False
            is_good_trajectory = False
            i_episode_step = 0
            episode_score = 0
            episode_transitions = []
            episode_utility_transitions = []
            policy_losses = list()

            self.testing = (self.i_episode % self.args.eval_period == 0)
            while not done and i_episode_step < max_traj_length:
                if self.args.render and self.i_iteration % int(self.args.render_freq) == 0:
                    self.env.render()

                # get current predicate
                predicate_values = self.env.get_current_predicate()
                predicate_vector = self.to_predicate_vectors([predicate_values])

                stacked_states = np.copy(np.stack(list(states_queue), axis=0))
                is_random_action = (self.total_step < self.hyper_params.policy_random_actions)

                util_vector = self.get_utility_values_vectors(np.expand_dims(stacked_states, axis=0),
                                                              [predicate_values]) if self.hyper_params.augmented_feature else None
                action = self.policy.get_action(stacked_states, is_test=self.testing, is_random=is_random_action,
                                                info={TRAJECTORY_INDEX.PREDICATE_VECTOR.value: predicate_vector,
                                                      TRAJECTORY_INDEX.UTILITY_VECTOR.value: util_vector})

                # interact with the environment
                if self.is_discrete and self.hyper_params.discrete_to_one_hot:
                    action_id = common_utils.one_hot_to_discrete_action(action, is_softmax=False)
                    action = common_utils.discrete_action_to_one_hot(action_id, self.action_dim)
                    next_state, true_reward, done, info = self.env.step(action_id)
                else:
                    next_state, true_reward, done, info = self.env.step(action)
                next_predicate_values = self.env.get_current_predicate()

                if true_reward > 0 and done:
                    is_good_trajectory = True
                episode_score += true_reward

                # add next state into states queue
                next_state = np.squeeze(next_state, axis=0)
                states_queue.append(next_state)
                # save the new transition
                utility_transition = UtilityTransition(predicate_values, next_predicate_values)
                episode_utility_transitions.append(utility_transition.util_transition)
                transition = (stacked_states, action, true_reward, np.copy(np.stack(list(states_queue), axis=0)), done)
                episode_transitions.append(transition)

                # training
                self._add_transition_to_policy_memory(transition, utility_transition.util_transition)
                # update policy
                policy_loss_info = self.update_policy()
                if policy_loss_info is not None:
                    policy_losses.append(policy_loss_info)

                # logging
                if self.logger:
                    action_taken = action
                    if self.is_discrete and self.hyper_params.discrete_to_one_hot:
                        action_taken = common_utils.one_hot_to_discrete_action(action)
                    # log transition
                    self.logger.add_transition(state, action_taken, true_reward, next_state, done,
                                               is_save_utility=True, predicate_values=predicate_values,
                                               next_predicate_values=next_predicate_values,
                                               utility_map=None, utility_values=None)

                self.total_step += 1
                i_episode_step += 1

            # save the transitions to corresponding trajectories
            if is_good_trajectory:
                self.good_buffer.extend(episode_transitions, utility_info=episode_utility_transitions)
            else:
                self.bad_buffer.extend(episode_transitions, utility_info=episode_utility_transitions)
            if 'do_post_episode_update' in dir(self.policy):
                self.policy.do_post_episode_update(self.total_step, self.hyper_params.policy_random_actions)

            # write policy log
            if len(policy_losses) > 0:
                policy_loss_info = np.vstack(policy_losses).mean(axis=0)
                self.policy.write_policy_log(policy_loss_info)
                policy_log = self.policy.get_policy_wandb_log(policy_loss_info)
                self.policy.write_policy_wandb_log(policy_log, step=self.total_step)
            # write explainer log
            self.avg_scores_window.append(episode_score)
            avg_score_window = float(np.mean(list(self.avg_scores_window)))
            self.write_episode_log((episode_score, avg_score_window))

            if self.testing:
                self.eval_scores_window.append(episode_score)
                # noinspection PyStringFormat
                print('[EVAL INFO] episode: %d, total step %d, '
                      'evaluation score: %.3f, window avg: %.3f\n'
                      % (self.i_episode,
                         self.total_step,
                         episode_score,
                         np.mean(self.eval_scores_window)))

                if self.logger is not None:
                    self.logger.log_wandb({
                        'eval score': episode_score,
                        "eval window avg": np.mean(self.eval_scores_window),
                    }, step=self.total_step)

                self.testing = False

            cnt_steps += i_episode_step
            self.i_episode += 1

    def _add_transition_to_policy_memory(self, transition, utility_info):
        self.policy.add_transition_to_memory(transition, utility_info)

    def update_policy(self):
        """
        Update/optimize the controller_policy network
        :return: return training related information (may vary based on the controller_policy network algorithms)
        """
        policy_losses = []
        if self.total_step >= self.hyper_params.policy_update_starts_from:
            if self.total_step % self.hyper_params.policy_train_freq == 0:
                for _ in range(self.hyper_params.policy_multiple_update):
                    policy_loss = self.policy.update_policy(indexed_trajs=None, irl_model=self)
                    policy_losses.append(policy_loss)  # for logging
        if len(policy_losses) == 0:
            return None
        else:
            return np.vstack(policy_losses).mean(axis=0)

    def update_explainer(self):
        if self.total_step >= self.hyper_params.update_starts_from:
            return self._update_explainer()
        else:
            return None

    # noinspection PyArgumentList
    def _update_explainer(self):
        """
        update the discriminator
        :param: indexed_sampled_traj: the sampled trajectories from current iteration
        """
        # calculate the good batch size
        n_good = len(self.good_buffer)
        good_batch_size = min(n_good, int(self.hyper_params.batch_size/2))

        # sample batch_size
        n_bad = len(self.bad_buffer)
        bad_batch_size = min(n_bad, int(self.hyper_params.batch_size/2))

        losses_iteration = []
        good_predict_acc = []
        bad_predict_acc = []
        for it in range(self.hyper_params.multiple_update):
            # sample good trajectories
            good_indices = np.random.randint(low=0, high=n_good, size=good_batch_size)
            bad_indices = np.random.randint(low=0, high=n_bad, size=bad_batch_size)
            n_samples = min(good_indices.shape[0], bad_indices.shape[0])

            arrange = np.arange(n_samples)
            np.random.shuffle(arrange)
            mini_batch_size = int(self.hyper_params.mini_batch_size / 2)
            for i in range(arrange.shape[0] // mini_batch_size):
                start_idx = mini_batch_size * i
                end_idx = mini_batch_size * (i + 1)
                good_batch_index = good_indices[arrange[start_idx:end_idx]]
                bad_batch_index = bad_indices[arrange[start_idx:end_idx]]

                # get labels of each sample
                batch_good_labels = np.ones(shape=(good_batch_index.shape[0],), dtype=int)
                batch_bad_labels = np.zeros(shape=(bad_batch_index.shape[0],), dtype=int)
                batch_labels = torch.FloatTensor(
                    torch.from_numpy(np.concatenate((batch_good_labels, batch_bad_labels), axis=0)).float()).to(
                    device).unsqueeze(1)

                # get good samples
                good_states, good_actions, good_rewards, good_next_states, _ = self.good_buffer.sample(indices=good_batch_index, is_to_tensor=False)
                good_predicates, good_next_predicates = self.good_buffer.get_utility_info(indices=good_batch_index)
                # get bad samples
                bad_states, bad_actions, bad_rewards, bad_next_states, _ = self.bad_buffer.sample(indices=bad_batch_index, is_to_tensor=False)
                bad_predicates, bad_next_predicates = self.bad_buffer.get_utility_info(indices=bad_batch_index)

                # get states
                batch_states = np.concatenate((good_next_states, bad_next_states), axis=0)
                batch_next_states = np.concatenate((good_states, bad_states), axis=0)
                batch_actions = np.concatenate((good_actions, bad_actions), axis=0)
                batch_rewards = torch.from_numpy(np.concatenate((good_rewards, bad_rewards), axis=0)).type(torch.FloatTensor).to(device)
                batch_predicates = np.concatenate((good_predicates, bad_predicates), axis=0)
                batch_predicate_vectors = self.to_predicate_vectors(batch_predicates)
                batch_next_predicates = np.concatenate((good_next_predicates, bad_next_predicates), axis=0)
                batch_next_predicate_vectors = self.to_predicate_vectors(batch_next_predicates)

                # augment feature
                info = None
                if self.hyper_params.augmented_feature:
                    predicate_util_vectors = self.get_utility_values_vectors(batch_states, batch_predicates)
                    info = {TRAJECTORY_INDEX.PREDICATE_VECTOR.value: batch_predicate_vectors,
                            TRAJECTORY_INDEX.UTILITY_VECTOR.value: predicate_util_vectors}

                # compute log_q
                log_q = self.policy.evaluate_states_actions(batch_states, batch_actions, info=info)
                batch_log_q = torch.from_numpy(log_q).type(torch.FloatTensor).to(device)
                if torch.isnan(batch_log_q).any():
                    print('[ERROR] batch_log_q contains nan: ', batch_log_q)

                # compute energy (energy is negative utility)
                if self.hyper_params.eval_verion == 2:
                    energy = self.eval_version2(batch_states, batch_predicate_vectors, batch_rewards,
                                                batch_next_states, batch_next_predicate_vectors)
                else:
                    energy = self.eval(batch_states, batch_predicate_vectors)
                batch_log_p = torch.sum(-energy, dim=1, keepdim=True)
                batch_log_p = torch.clamp(batch_log_p, max=MAX_ENERGY)
                if torch.isnan(batch_log_p).any():
                    print('[ERROR] batch_log_p contains nan: ', batch_log_p)

                batch_log_p_q = (batch_log_q.exp() + batch_log_p.exp() + LOG_REG).log()
                if torch.isnan(batch_log_p_q).any():
                    print('[ERROR] batch_log_p_q contain nan: ', batch_log_p_q)
                    print('[ERROR] batch_log_q.exp(): ', batch_log_q.exp())
                    print('[ERROR] batch_log_p.exp(): ', batch_log_p.exp())

                # compute the loss
                loss = batch_labels * (batch_log_p - batch_log_p_q) + (1 - batch_labels) * (batch_log_q - batch_log_p_q)
                if torch.isnan(loss).any():
                    print('[ERROR] loss contain nan: ', loss)
                    print('[ERROR] batch_labels: ', batch_labels)
                    print('[ERROR] batch_log_p_q: ', batch_log_p_q)
                    print('[ERROR] batch_log_p: ', batch_log_p)
                    print('[ERROR] batch_log_q: ', batch_log_q)
                    print('[ERROR] batch_log_p - batch_log_p_q: ', batch_log_p - batch_log_p_q)
                    print('[ERROR] batch_log_q - batch_log_p_q: ', batch_log_q - batch_log_p_q)

                # maximize the log likelihood -> minimize mean loss
                mean_loss = -torch.mean(loss)
                self.explainer_optim.zero_grad()
                mean_loss.backward()
                if self.hyper_params.gradient_clip is not None:
                    clip_grad_norm_(self.explainer.parameters(), self.hyper_params.gradient_clip)
                self.explainer_optim.step()

                # for logging: get the discriminator output
                clone_log_p_tau = batch_log_p.clone().detach().cpu().numpy()
                clone_log_p_q = batch_log_p_q.clone().detach().cpu().numpy()
                n_batch_good = good_batch_index.shape[0]
                n_batch_bad = bad_batch_index.shape[0]
                good_prediction = np.exp(clone_log_p_tau[0:n_batch_good] - clone_log_p_q[0:n_batch_good])
                bad_prediction = np.exp(
                    clone_log_p_tau[n_batch_good:n_batch_good + n_batch_bad]
                    - clone_log_p_q[n_batch_good:n_batch_good + n_batch_bad])
                good_predict_acc.append(np.mean((good_prediction > 0.5).astype(float)))
                bad_predict_acc.append(np.mean((bad_prediction < 0.5).astype(float)))
                losses_iteration.append(mean_loss.item())

        return np.mean(losses_iteration), np.mean(good_predict_acc), np.mean(bad_predict_acc)

    def load_params(self, path):
        """Load model and optimizer parameters."""
        Agent.load_params(self, path)

        params = torch.load(path, map_location=device)
        self.explainer.load_state_dict(params["explainer_state_dict"])
        self.explainer_optim.load_state_dict(params["explainer_optim_state_dict"])
        print("[Explainer INFO] explainer loaded the model and optimizer from", path)

    # noinspection PyMethodOverriding
    def save_params(self, n_step):
        """Save model and optimizer parameters."""
        params = {
            "explainer_state_dict": self.explainer.state_dict(),
            "explainer_optim_state_dict": self.explainer_optim.state_dict(),
            "step": n_step
        }
        if self.logger is not None:
            self.logger.save_models(params, prefix='explainer', postfix=str(n_step), is_snapshot=True)

        Agent.save_params(self, params, n_step)

    def write_log(self, log_value):
        pass

    def write_episode_log(self, log_value):
        episode_score, window_avg_score = log_value
        print("\n[Explainer INFO] iteration %d, episode %d, total step: %d, score: %d, window avg: %f"
              % (self.i_iteration, self.i_episode, self.total_step, episode_score, window_avg_score))

        if self.logger is not None:
            self.logger.log_wandb({
                'episode': self.i_episode,
                'score': episode_score,
                'avg score window': window_avg_score,
                'total step': self.total_step
            }, step=self.total_step)

    def write_iteration_log(self, log_value):
        """
        Write log about loss and score
        """
        i_iteration, explainer_avg_loss, demo_avg_acc, sampled_avg_acc, iteration_time_cost = log_value

        print(
            "\n[Explainer INFO] iteration %d, total episode: %d, total step: %d\n"
            "explainer loss: %.6f, good prediction accuracy: %.3f%%, "
            "bad prediction accuracy: %.3f%%, iteration time cost: %.3f"
            % (
                self.i_iteration,
                self.i_episode,
                self.total_step,
                explainer_avg_loss,
                demo_avg_acc * 100,
                sampled_avg_acc * 100,
                iteration_time_cost
            )
        )

        if self.logger is not None:
            wandb_log = {
                'explainer_loss': explainer_avg_loss,
                'good accuracy': demo_avg_acc,
                'bad accuracy': sampled_avg_acc,
                'episode': self.i_episode,
            }
            self.logger.log_wandb(wandb_log, step=self.total_step)

    def get_wandb_watch_list(self):
        return [self.explainer]

    def train(self):
        """Train the agent."""
        if self.logger is not None:
            self.logger.watch_wandb(self.get_wandb_watch_list() + self.policy.get_wandb_watch_list())

        # pre-training if needed
        self.pretrain()

        for i_iteration in range(1, self.args.iteration_num):
            self.i_iteration = i_iteration
            t_begin = time.time()

            # sample trajectories
            self.sample_trajectories()
            explainer_loss_info = self._update_explainer()

            t_end = time.time()
            iteration_time_cost = t_end - t_begin

            # logging
            if self.logger and explainer_loss_info:
                discriminator_avg_loss, good_avg_acc, bad_avg_acc = explainer_loss_info
                discriminator_log_values = (
                    i_iteration,
                    discriminator_avg_loss,
                    good_avg_acc,
                    bad_avg_acc,
                    iteration_time_cost
                )
                self.write_iteration_log(discriminator_log_values)

            if i_iteration % self.args.save_period == 0:
                self.save_params(self.total_step)
                self.policy.save_params(self.total_step)

    # pylint: disable=no-self-use, unnecessary-pass
    def pretrain(self):
        """ Pre-training steps. """
        # pretrain the policy
        self.is_doing_pretrain = True
        self.policy.pretrain_policy(irl_model=self)
        self.is_doing_pretrain = False

    def select_action(self, state):
        pass

    def step(self, action):
        pass

    def update_model(self):
        pass





