import time

import numpy as np
import torch
import torch.optim as optim
from collections import deque

from learning_agents.architectures.cnn import Conv2d_MLP_Model
from torch.nn.utils import clip_grad_norm_
from learning_agents.common.priortized_replay_buffer import PrioritizedReplayBuffer
from learning_agents.common.replay_buffer import ReplayBuffer
from learning_agents.rl.dqn import dqn_utils
from learning_agents.common import common_utils
from learning_agents.agent import Agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQNAgent(Agent):
    """
    DQN Agent. (Here we assume that we are using stacked frames and grayscaled observation)
    Attribute:
        env (gym.Env): openAI Gym environment
        args (argparse.Namespace): arguments including hyperparameters and training settings
        hyper_params (ConfigDict): hyper-parameters
        network_cfg (ConfigDict): config of network for training agent
        optim_cfg (ConfigDict): config of optimizer
        state_dim (int): state size of env
        action_dim (int): action size of env
        memory (PrioritizedReplayBuffer): replay memory
        dqn (nn.Module): actor model to select actions
        dqn_target (nn.Module): target actor model to select actions
        dqn_optim (Optimizer): optimizer for training actor
        curr_state (np.ndarray): temporary storage of the current state
        total_step (int): total step number
        episode_step (int): step number of the current episode
        i_episode (int): current episode number
        epsilon (float): parameter for epsilon greedy controller_policy
        n_step_buffer (deque): n-size buffer to calculate n-step returns
        per_beta (float): beta parameter for prioritized replay buffer
        use_n_step (bool): whether or not to use n-step returns
    """

    def __init__(self, env, args, log_cfg, hyper_params, network_cfg, optim_cfg, encoder=None, logger=None):
        """Initialize."""
        Agent.__init__(self, env, args, log_cfg)

        self.episode_step = 0
        self.total_step = 0
        self.i_episode = 0
        self.logger = logger
        self.encoder = encoder

        self.hyper_params = hyper_params
        self.network_cfg = network_cfg
        self.optim_cfg = optim_cfg

        # get state space info
        self.state_dim = self.env.observation_space.shape[0]
        print('[INFO] state shape: ', self.env.observation_space.shape)
        # check if it's single channel or multi channel
        self.state_channel = self.hyper_params.frame_stack
        print('[INFO] stack state: ', self.state_channel)
        # get action space info (DQN only works on discrete domain)
        self.is_discrete = True
        self.action_dim = self.env.action_space.n
        print('[INFO] action dim: ', self.action_dim)

        self.per_beta = hyper_params.per_beta
        self.use_n_step = hyper_params.n_step > 1
        self.use_prioritized = hyper_params.use_prioritized

        if hyper_params.use_noisy_net:
            self.max_epsilon = 0.0
            self.min_epsilon = 0.0
            self.epsilon = 0.0
        else:
            self.max_epsilon = hyper_params.max_epsilon
            self.min_epsilon = hyper_params.min_epsilon
            self.epsilon = hyper_params.max_epsilon

        self._initialize()
        self._init_network()

        self.testing = self.args.test

    # pylint: disable=attribute-defined-outside-init
    def _initialize(self):
        """Initialize non-common things."""
        if not self.args.test:
            # replay memory for a single step
            if self.use_prioritized:
                self.memory = PrioritizedReplayBuffer(
                    self.hyper_params.buffer_size,
                    self.hyper_params.batch_size,
                    alpha=self.hyper_params.per_alpha,
                )
            # use ordinary replay buffer
            else:
                self.memory = ReplayBuffer(
                    self.hyper_params.buffer_size,
                    batch_size=self.hyper_params.batch_size,
                    gamma=self.hyper_params.gamma,
                )

            # replay memory for multi-steps
            if self.use_n_step:
                self.memory_n = ReplayBuffer(
                    self.hyper_params.buffer_size,
                    batch_size=self.hyper_params.batch_size,
                    n_step=self.hyper_params.n_step,
                    gamma=self.hyper_params.gamma,
                )

    # pylint: disable=attribute-defined-outside-init
    def _init_network(self):
        """Initialize networks and optimizers."""
        self.dqn = Conv2d_MLP_Model(input_channels=self.state_channel,
                                    fc_input_size=self.network_cfg.fc_input_size,
                                    fc_output_size=self.action_dim,
                                    nonlinearity=self.network_cfg.nonlinearity,
                                    channels=self.network_cfg.channels,
                                    kernel_sizes=self.network_cfg.kernel_sizes,
                                    strides=self.network_cfg.strides,
                                    paddings=self.network_cfg.paddings,
                                    fc_hidden_sizes=self.network_cfg.fc_hidden_sizes,
                                    fc_hidden_activation=self.network_cfg.fc_hidden_activation).to(device)
        self.dqn_target = Conv2d_MLP_Model(input_channels=self.state_channel,
                                           fc_input_size=self.network_cfg.fc_input_size,
                                           fc_output_size=self.action_dim,
                                           nonlinearity=self.network_cfg.nonlinearity,
                                           channels=self.network_cfg.channels,
                                           kernel_sizes=self.network_cfg.kernel_sizes,
                                           strides=self.network_cfg.strides,
                                           paddings=self.network_cfg.paddings,
                                           fc_hidden_sizes=self.network_cfg.fc_hidden_sizes,
                                           fc_hidden_activation=self.network_cfg.fc_hidden_activation).to(device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())

        # create optimizer
        self.dqn_optim = optim.Adam(
            self.dqn.parameters(),
            lr=self.optim_cfg.lr_dqn,
            weight_decay=self.optim_cfg.weight_decay,
            eps=self.optim_cfg.adam_eps,
        )

        # init network from file
        self._init_from_file()

    def _init_from_file(self):
        # load the optimizer and model parameters
        if self.args.load_from is not None:
            self.load_params(self.args.load_from)

    def select_action(self, state):
        """Select an action from the input space."""

        # epsilon greedy controller_policy
        # pylint: disable=comparison-with-callable
        if not self.testing and \
                (self.epsilon > np.random.random() or self.total_step < self.hyper_params.init_random_actions):
            selected_action = np.array(self.env.action_space.sample())
        else:
            state = self._preprocess_state(state)
            self.dqn.eval()
            with torch.no_grad():
                selected_action = self.dqn(state).argmax()
            self.dqn.train()
            selected_action = selected_action.detach().cpu().numpy()
        return selected_action

    # pylint: disable=no-self-use
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

    def step(self, action):
        """Take an action and return the response of the env."""
        next_state, reward, done, info = self.env.step(action)

        done = (
            True if self.episode_step == self.hyper_params.max_traj_length else done
        )
        return next_state, reward, done, info

    def _add_transition_to_memory(self, transition):
        """Add 1 step and n step transitions to memory."""
        # add n-step transition
        if self.use_n_step:
            transition = self.memory_n.add(transition)

        # add a single step transition
        # if transition is not an empty tuple
        if transition:
            self.memory.add(transition)

    def _get_dqn_loss(self, experiences, gamma):
        """Return element-wise dqn loss and Q-values."""
        return dqn_utils.calculate_dqn_loss(
            model=self.dqn,
            target_model=self.dqn_target,
            experiences=experiences,
            gamma=gamma,
            use_double_q_update=self.hyper_params.use_double_q_update,
            reward_clip=self.hyper_params.reward_clip
        )

    def update_model(self):
        """Train the model after each episode."""
        # 1 step loss
        if self.use_prioritized:
            experiences_one_step = self.memory.sample(self.per_beta)
            weights, indices = experiences_one_step[-3:-1]
            indices = np.array(indices)
            # re-normalize the weights such that they sum up to the value of batch_size
            weights = weights/torch.sum(weights)*float(self.hyper_params.batch_size)
        else:
            indices = np.random.choice(len(self.memory), size=self.hyper_params.batch_size, replace=False)
            weights = torch.from_numpy(np.ones(shape=(indices.shape[0], 1), dtype=np.float64)).type(torch.FloatTensor).to(device)
            experiences_one_step = self.memory.sample(indices=indices)

        dq_loss_element_wise, q_values = self._get_dqn_loss(experiences_one_step, self.hyper_params.gamma)
        dq_loss = torch.mean(dq_loss_element_wise * weights)

        # n step loss
        if self.use_n_step:
            experiences_n = self.memory_n.sample(indices)
            gamma = self.hyper_params.gamma ** self.hyper_params.n_step
            dq_loss_n_element_wise, q_values_n = self._get_dqn_loss(experiences_n, gamma)

            # to update loss and priorities
            q_values = 0.5 * (q_values + q_values_n)
            # mix of 1-step and n-step returns
            dq_loss_element_wise += dq_loss_n_element_wise * self.hyper_params.w_n_step
            dq_loss = torch.mean(dq_loss_element_wise * weights)

        # total loss
        loss = dq_loss

        # q_value regularization (not used when w_q_reg is set to 0)
        if self.optim_cfg.w_q_reg > 0:
            q_regular = torch.norm(q_values, 2).mean() * self.optim_cfg.w_q_reg
            loss = loss + q_regular

        self.dqn_optim.zero_grad()
        loss.backward()
        if self.hyper_params.gradient_clip is not None:
            clip_grad_norm_(self.dqn.parameters(), self.hyper_params.gradient_clip)
        self.dqn_optim.step()

        # update target networks
        common_utils.soft_update(self.dqn, self.dqn_target, self.hyper_params.tau)

        # update priorities in PER
        if self.use_prioritized:
            loss_for_prior = dq_loss_element_wise.detach().cpu().numpy().squeeze()
            new_priorities = loss_for_prior + self.hyper_params.per_eps
            if (new_priorities <= 0).any().item():
                print('[ERROR] new priorities less than 0. Loss info: ', str(loss_for_prior))

            # noinspection PyUnresolvedReferences
            self.memory.update_priorities(indices, new_priorities)

            # increase beta
            fraction = min(float(self.i_episode) / self.args.iteration_num, 1.0)
            self.per_beta = self.per_beta + fraction * (1.0 - self.per_beta)

        # whether to use noise net
        if self.hyper_params.use_noisy_net:
            self.dqn.reset_noise()
            self.dqn_target.reset_noise()

        return loss.item(), q_values.mean().item()

    def load_params(self, path):
        """Load model and optimizer parameters."""
        Agent.load_params(self, path)

        params = torch.load(path, map_location=device)
        self.dqn.load_state_dict(params["dqn_state_dict"])
        self.dqn_target.load_state_dict(params["dqn_target_state_dict"])
        self.dqn_optim.load_state_dict(params["dqn_optim_state_dict"])
        print("[INFO] loaded the model and optimizer from", path)

    def save_params(self, n_step):
        """Save model and optimizer parameters."""
        params = {
            "dqn_state_dict": self.dqn.state_dict(),
            "dqn_target_state_dict": self.dqn_target.state_dict(),
            "dqn_optim_state_dict": self.dqn_optim.state_dict(),
        }

        Agent.save_params(self, params, n_step)

        if self.logger is not None:
            self.logger.save_models(params, postfix=str(n_step), is_snapshot=True)

    def write_log(self, log_value):
        """Write log about loss and score"""
        i, loss, score, avg_time_cost, avg_score_window = log_value
        print(
            "[INFO] episode %d, episode step: %d, total step: %d, total score: %f\n"
            "epsilon: %f, loss: %f, avg q-value: %f , avg score window: %f (spent %.6f sec/step)\n"
            % (
                i,
                self.episode_step,
                self.total_step,
                score,
                self.epsilon,
                loss[0],
                loss[1],
                avg_score_window,
                avg_time_cost,
            )
        )

        if self.logger is not None:
            self.logger.log_wandb({
                "score": score,
                "episode": self.i_episode,
                "episode step": self.episode_step,
                "total step": self.total_step,
                "epsilon": self.epsilon,
                "dqn loss": loss[0],
                "avg q values": loss[1],
                "time per each step": avg_time_cost,
                "avg score window": avg_score_window,
            }, step=self.total_step)

    # pylint: disable=no-self-use, unnecessary-pass
    def pretrain(self):
        """Pretraining steps."""
        pass

    def train(self):
        """Train the agent."""
        # logger
        if self.logger is not None:
            self.logger.watch_wandb([self.dqn, self.dqn_target])

        # pre-training if needed
        self.pretrain()

        avg_scores_window = deque(maxlen=self.args.avg_score_window)
        eval_scores_window = deque(maxlen=self.args.eval_score_window)

        for i_episode in range(1, self.args.iteration_num + 1):
            self.i_episode = i_episode
            self.testing = (self.i_episode % self.args.eval_period == 0)

            state = np.squeeze(self.env.reset(), axis=0)
            self.episode_step = 0
            losses = list()
            done = False
            score = 0

            states_queue = deque(maxlen=self.hyper_params.frame_stack)
            states_queue.extend([state for _ in range(self.hyper_params.frame_stack)])

            t_begin = time.time()

            while not done:
                if self.args.render \
                        and self.i_episode >= self.args.render_after \
                        and self.i_episode % self.args.render_freq == 0:
                    self.env.render()

                stacked_states = np.copy(np.stack(list(states_queue), axis=0))
                action = self.select_action(stacked_states)
                next_state, reward, done, _ = self.step(action)
                # add next state into states queue
                next_state = np.squeeze(next_state, axis=0)
                states_queue.append(next_state)
                # save the new transition
                transition = (stacked_states, action, reward, np.copy(np.stack(list(states_queue), axis=0)), done)
                self._add_transition_to_memory(transition)

                self.total_step += 1
                self.episode_step += 1

                if len(self.memory) >= self.hyper_params.update_starts_from:
                    if self.total_step % self.hyper_params.train_freq == 0:
                        for _ in range(self.hyper_params.multiple_update):
                            loss = self.update_model()
                            losses.append(loss)  # for logging
                score += reward

            self.do_post_episode_update()
            t_end = time.time()
            avg_time_cost = (t_end - t_begin) / self.episode_step
            avg_scores_window.append(score)

            if self.testing:
                eval_scores_window.append(score)
                # noinspection PyStringFormat
                print('[EVAL INFO] episode: %d, total step %d, '
                      'evaluation score: %.3f, window avg: %.3f\n'
                      % (self.i_episode,
                         self.total_step,
                         score,
                         np.mean(eval_scores_window)))

                if self.logger is not None:
                    self.logger.log_wandb({
                        'eval score': score,
                        "eval window avg": np.mean(eval_scores_window),
                    }, step=self.total_step)

                self.testing = False

            if losses:
                avg_loss = np.vstack(losses).mean(axis=0)
                log_value = (self.i_episode, avg_loss, score, avg_time_cost, np.mean(avg_scores_window))
                self.write_log(log_value)

            if self.i_episode % self.args.save_period == 0:
                self.save_params(self.total_step)

        # termination
        self.env.close()
        self.save_params(self.total_step)

    def do_post_episode_update(self, *argv):
        if self.total_step >= self.hyper_params.init_random_actions:
            # decrease epsilon
            self.epsilon = max(self.min_epsilon, self.hyper_params.epsilon_decay * self.epsilon)

    def test(self, n_episode=1e3, verbose=True):
        avg_scores_window = deque(maxlen=self.args.avg_score_window)
        self.testing = True

        for i_episode in range(int(n_episode)):
            state = np.squeeze(self.env.reset(), axis=0)
            done = False
            score = 0
            self.episode_step = 0

            states_queue = deque(maxlen=self.hyper_params.frame_stack)
            states_queue.extend([state for _ in range(self.hyper_params.frame_stack)])

            while not done:
                self.env.render()

                stacked_stats = np.copy(np.stack(list(states_queue), axis=0))
                action = self.select_action(stacked_stats)
                next_state, reward, done, _ = self.step(action)
                # add next state into states queue
                next_state = np.squeeze(next_state, axis=0)
                states_queue.append(next_state)

                self.total_step += 1
                self.episode_step += 1

                score += reward

            avg_scores_window.append(score)
            if verbose:
                print('Episode: %d, Score: %f, Avg Score Window: %f, Episode Steps: %d, Total Steps: %d\n'
                      % (i_episode, score, float(np.mean(list(avg_scores_window))), self.episode_step, self.total_step))

        # termination
        self.env.close()
        return list(avg_scores_window)

