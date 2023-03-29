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
from utils.trajectory_utils import read_expert_demo, get_flatten_trajectories,  \
    stack_frames_in_traj, get_n_step_info_from_traj

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQfDAgent(Agent):
    """
    DQNfD Agent. (Here we assume that we are using stacked frames and grayscaled observation)
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
        demo_preprocessor: preprocessor of frames in demonstrations
    """

    def __init__(self, env, args, log_cfg, hyper_params, network_cfg, optim_cfg, demo_preprocessor, encoder=None, logger=None):
        """Initialize."""
        Agent.__init__(self, env, args, log_cfg)

        self.episode_step = 0
        self.total_step = 0
        self.i_episode = 0
        self.logger = logger
        self.encoder = encoder
        self.demo_preprocessor = demo_preprocessor

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
        self.lambda2 = hyper_params.max_lambda2

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
            # get flatten demos
            demos = self._load_demos()
            print('[INFO] demo observation shape: ', demos[0][0].shape)
            # replay memory for multi-steps
            if self.use_n_step:
                demos, demos_n_step = get_n_step_info_from_traj(demos, self.hyper_params.n_step, self.hyper_params.gamma)
                self.memory_n = ReplayBuffer(
                    self.hyper_params.buffer_size,
                    batch_size=self.hyper_params.batch_size,
                    n_step=self.hyper_params.n_step,
                    gamma=self.hyper_params.gamma,
                    demo=demos_n_step,
                )

            # replay memory for a single step
            if self.use_prioritized:
                self.memory = PrioritizedReplayBuffer(
                    self.hyper_params.buffer_size,
                    self.hyper_params.batch_size,
                    demo=demos,
                    alpha=self.hyper_params.per_alpha,
                    epsilon_d=self.hyper_params.per_eps_demo,
                    epsilon_a=self.hyper_params.per_eps
                )
            # use ordinary replay buffer
            else:
                self.memory = ReplayBuffer(
                    self.hyper_params.buffer_size,
                    batch_size=self.hyper_params.batch_size,
                    demo=demos,
                    gamma=self.hyper_params.gamma,
                )

    def _load_demos(self):
        demos = read_expert_demo(self.args.demo_path)
        flatten_demo_traj = get_flatten_trajectories(demos)
        return stack_frames_in_traj(flatten_demo_traj, self.demo_preprocessor, self.hyper_params.frame_stack)

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
        n_sample = min(len(self.memory), self.hyper_params.batch_size)
        if self.use_prioritized:
            experiences_one_step = self.memory.sample(self.per_beta)
            weights, indices, eps_demo = experiences_one_step[-3:]
            indices = np.array(indices)
            actions = experiences_one_step[1]
            # re-normalize the weights such that they sum up to the value of batch_size
            weights = weights/torch.sum(weights)*float(n_sample)
        else:
            indices = np.random.choice(len(self.memory), size=n_sample, replace=False)
            eps_demo = np.zeros_like(indices)
            eps_demo[np.where(indices < self.memory.demo_size)] = self.hyper_params.per_eps_demo
            weights = torch.from_numpy(np.ones(shape=(indices.shape[0], 1), dtype=np.float64)).type(torch.FloatTensor).to(device)
            experiences_one_step = self.memory.sample(indices=indices)
            actions = experiences_one_step[1]

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
            dq_loss_element_wise += dq_loss_n_element_wise * self.hyper_params.lambda1
            dq_loss = torch.mean(dq_loss_element_wise * weights)

        # q_value regularization (not used when w_q_reg is set to 0)
        q_regular = torch.norm(q_values, 2).mean() * self.optim_cfg.w_q_reg

        # supervised loss using demo for only demo transitions
        demo_idxs = np.where(indices < self.memory.demo_size)
        n_demo = demo_idxs[0].size
        if n_demo != 0:  # if 1 or more demos are sampled
            # get margin for each demo transition
            action_idxs = actions[demo_idxs].long()
            margin = torch.ones(q_values.size()) * self.hyper_params.margin
            margin[demo_idxs, action_idxs] = 0.0  # demo actions have 0 margins
            margin = margin.to(device)

            # calculate supervised loss
            demo_q_values = q_values[demo_idxs, action_idxs].squeeze()
            supervised_loss = torch.max(q_values + margin, dim=-1)[0]
            supervised_loss = supervised_loss[demo_idxs] - demo_q_values
            supervised_loss = torch.mean(supervised_loss) * self.lambda2
        else:  # no demo sampled
            supervised_loss = torch.zeros(1, device=device)

        # total loss
        loss = dq_loss + supervised_loss + q_regular
        # train dqn
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
            new_priorities += eps_demo

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

        return loss.item(), dq_loss.item(), supervised_loss.item(), q_values.mean().item(), n_demo

    def load_params(self, path):
        """Load model and optimizer parameters."""
        Agent.load_params(self, path)

        params = torch.load(path, map_location=device)
        self.dqn.load_state_dict(params["dqn_state_dict"])
        self.dqn_target.load_state_dict(params["dqn_target_state_dict"])
        self.dqn_optim.load_state_dict(params["dqn_optim_state_dict"])
        print("[INFO] loaded the model and optimizer from", path)

    def save_params(self, n_step):  # type: ignore
        """Save model and optimizer parameters."""
        params = {
            "dqn_state_dict": self.dqn.state_dict(),
            "dqn_target_state_dict": self.dqn_target.state_dict(),
            "dqn_optim_state_dict": self.dqn_optim.state_dict(),
        }

        Agent.save_params(self, params, n_step)

        if self.logger is not None:
            self.logger.save_models(params, prefix='dqfd', postfix=str(n_step), is_snapshot=True)

    def write_log(self, log_value):
        """Write log about loss and score"""
        i, loss_info, score, avg_time_cost, avg_score_window = log_value
        print(
            "[INFO] episode %d, episode step: %d, total step: %d\n"
            "total score: %f, avg score window: %f\n"
            "epsilon: %f, total loss: %f, dq loss: %f, supervised loss: %f\n"
            "avg q values: %f, demo num in mini-batch: %d (spent %.6f sec/step)\n"
            % (
                i,
                self.episode_step,
                self.total_step,
                score,
                avg_score_window,
                self.epsilon,
                loss_info[0],
                loss_info[1],
                loss_info[2],
                loss_info[3],
                loss_info[4],
                avg_time_cost,
            )
        )

        if self.logger is not None:
            self.logger.log_wandb({
                "score": score,
                "episode": self.i_episode,
                "total step": self.total_step,
                "epsilon": self.epsilon,
                "total loss": loss_info[0],
                "dqn loss": loss_info[1],
                "supervised loss": loss_info[2],
                "avg q values": loss_info[3],
                "demo num": loss_info[4],
                "time per each step": avg_time_cost,
                "avg score window": avg_score_window,
            }, step=self.total_step)

    # pylint: disable=no-self-use, unnecessary-pass
    def pretrain(self):
        """ Pretraining steps. """
        pretrain_loss = list()
        pretrain_step = self.hyper_params.pretrain_step
        print("[INFO] Start pre-training for %d step." % pretrain_step)

        for i_step in range(1, pretrain_step + 1):
            loss = self.update_model()
            pretrain_loss.append(loss)  # for logging

            # logging
            if i_step == 1 or i_step % 100 == 0:
                loss_info = np.vstack(pretrain_loss).mean(axis=0)
                pretrain_loss.clear()
                print("[INFO] Pre-Train step %d, total loss: %f, dq loss: %f, supervised loss: %f\n"
                      "avg q values: %f, demo num in mini-batch: %d"
                      % (
                          i_step,
                          loss_info[0],
                          loss_info[1],
                          loss_info[2],
                          loss_info[3],
                          loss_info[4]
                      ))
        print("[INFO] Pre-Train Complete!\n")

    def test(self, n_episode=1e3, verbose=True):
        avg_scores_window = deque(maxlen=self.args.avg_score_window)
        self.testing = True

        for i_episode in range(int(n_episode)):
            state = np.squeeze(self.env.reset(), axis=0)
            done = False
            score = 0

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
                loss_info = np.vstack(losses).mean(axis=0)
                log_value = (self.i_episode, loss_info, score, avg_time_cost, np.mean(avg_scores_window))
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
        self.lambda2 = max(self.hyper_params.min_lambda2, self.hyper_params.lambda2_decay * self.lambda2)
