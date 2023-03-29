import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import gym

from learning_agents.agent import Agent
from learning_agents.common.trajectory_buffer import TrajectoryBuffer
import learning_agents.common.common_utils as common_utils
from learning_agents.architectures.mlp import MLP
from utils.trajectory_utils import read_expert_demo, get_indexed_trajs, demo_discrete_actions_to_one_hot
from learning_agents.utils.utils import ConfigDict, IndexedTraj
from utils.trajectory_utils import TRAJECTORY_INDEX, extract_experiences_from_indexed_trajs, split_fixed_length_indexed_traj
from utils.utils import concat_nested_list

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOG_REG = 1e-8


class GAIL(Agent):
    """
    Generative Adversarial Imitation Learning
    Attributes:
        ####################### attributes of the Discriminator ##########################
        env (gym.Env): openAI Gym environment
        irl_model (torch.nn): the irl/explainer model
        irl_optim (torch.optim): the irl/explainer network's optimizer
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
        ####################### attributes of the RL (controller_policy) ##############################
        policy_network (Agent): the pre-defined controller_policy network
    """

    def __init__(self, env, policy, args, log_cfg, hyper_params, network_cfg, optim_cfg, encoder=None, logger=None):
        """
        Initialize.
        env: gym.Env,
        args: argparse.Namespace,
        log_cfg: ConfigDict,
        hyper_params: ConfigDict,
        network_cfg: ConfigDict,
        optim_cfg: ConfigDict,
        controller_policy (Agent): the controller_policy module
        encoder(nn): if encoder is not None, the irl/explainer will use the encoder to preprocess the state
        logger (ExperimentLogger): the experiment logger
        """

        Agent.__init__(self, env, args, log_cfg)

        self.curr_state = np.zeros((1,))
        self.total_step = 0
        self.i_episode = 0
        self.i_iteration = 0

        self.hyper_params = hyper_params
        self.network_cfg = network_cfg
        self.optim_cfg = optim_cfg
        self.encoder = encoder
        self.logger = logger

        # get state space info
        if encoder is None:
            self.state_dim = self.env.observation_space.shape[0]
            # check if it's single channel or multi channel
            self.state_channel = 1 if len(self.env.observation_space.shape) == 2 else self.env.observation_space.shape[
                0]
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

        # set the controller_policy module
        self.policy = policy
        # init memory
        self._initialize()
        # init buffer
        self._init_demo_buff()

        # initialize the networks
        self._init_network()

    # pylint: disable=attribute-defined-outside-init
    def _initialize(self):
        """ Initialize non-common things """
        # human demonstration buffer
        if not self.args.test:
            # human demonstration buffer
            self.demo_buf = TrajectoryBuffer(
                self.hyper_params.buffer_size, self.hyper_params.batch_size
            )
            # sampled trajectories buffer (In Justin Fu's implementation, they don't reuse past sample)
            self.sampled_traj_buf = TrajectoryBuffer(
                self.hyper_params.buffer_size, self.hyper_params.batch_size
            )

    def _init_demo_buff(self):
        # store the expert demonstration
        demo_trajectories = read_expert_demo(self.args.demo_path)
        if self.is_discrete:
            demo_discrete_actions_to_one_hot(demo_trajectories, self.action_dim)
        indexed_demo = split_fixed_length_indexed_traj(get_indexed_trajs(demo_trajectories), self.hyper_params.traj_fixed_length)
        self.demo_buf.extend(indexed_demo)

    def _set_to_default_MLP(self):
        # the irl model
        self.irl_model = MLP(
            input_size=self.state_dim + self.action_dim,
            output_size=1,
            hidden_sizes=self.network_cfg.hidden_sizes_irl,
            hidden_activation=torch.tanh,
            output_activation=torch.sigmoid
        ).to(device)

    def _init_network(self):
        """ Initialize the IRL (explainer) network """
        self._set_to_default_MLP()

        # define optimizer
        self.irl_model_optim = optim.Adam(
            self.irl_model.parameters(),
            lr=self.optim_cfg.lr_discrim,
            weight_decay=self.optim_cfg.weight_decay
        )

        # load the optimizer and model parameters
        if self.args.load_from is not None:
            self.load_params(self.args.load_from)

    def load_params(self, path):
        """Load model and optimizer parameters."""
        Agent.load_params(self, path)

        params = torch.load(path, map_location=device)
        self.irl_model.load_state_dict(params["irl_state_dict"])
        self.irl_model_optim.load_state_dict(params["irl_optim_state_dict"])
        print("[INFO] loaded the model and optimizer from", path)

    # noinspection PyMethodOverriding
    def save_params(self, n_episode):
        """Save model and optimizer parameters."""
        params = {
            "irl_state_dict": self.irl_model.state_dict(),
            "irl_optim_state_dict": self.irl_model_optim.state_dict(),
            "epoch": n_episode
        }
        if self.logger is not None:
            self.logger.save_models(params, postfix=str(n_episode), is_snapshot=True)
        Agent.save_params(self, params, n_episode)

    def _preprocess_state(self, state):
        """Preprocess state so that actor selects an action."""
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(device)
        if self.encoder is not None:
            state = self.encoder(state)
        return state

    def get_reward(self, state, action=None):
        """
        Evaluate a single state-action pair. Will return the scalar value.

        :param states: numpy.ndarray
        :param actions: numpy.ndarray (in discrete case, use one-hot representation)
        :return: reward function, i.e. the log of energy function (if want cost, use the negation of the reward)
        """
        reward = np.log(torch.sum(self.eval(state, action), dim=1, keepdim=False).detach().cpu().numpy() + LOG_REG)
        return reward

    def eval(self, states, actions=None):
        """
        Evaluate the state action pairs (compute the energy value)
        Note this will return a Variable rather than the data in the numpy format.

        :param states: numpy.ndarray
        :param actions: numpy.ndarray (in discrete case, use one-hot representation)
        :return: reward function, i.e. the log of energy function (if want cost, use the negation of the reward)
        """
        states = self._preprocess_state(states)
        if not isinstance(actions, torch.Tensor):
            if isinstance(actions, np.ndarray):
                actions = Variable(torch.from_numpy(actions)).type(torch.FloatTensor).to(device)
            else:
                actions = Variable(torch.Tensor(actions)).type(torch.FloatTensor).to(device)

        return self.irl_model(torch.cat((states, actions), 1))

    def eval_trajectories(self, indexed_trajs):
        """
        Compute the reward/cost/utility of each transition in the trajectory list
        Will override the existing reward value in the trajectories list
        :param indexed_trajs: list of dict (in discrete case, the actions in trajectories all use one-hot representation)
        """
        for traj_idx, traj in enumerate(indexed_trajs):
            states = traj[TRAJECTORY_INDEX.STATE.value]
            actions = traj[TRAJECTORY_INDEX.ACTION.value]
            rewards = self.get_reward(states, actions)
            traj[TRAJECTORY_INDEX.REWARD.value] = rewards

    def sample_trajectories(self):
        """
        Use the controller_policy network to sample a list of trajectories
        :return (list of dict): list of trajectories (each trajectory is a dict)
        """
        max_traj_length = self.hyper_params.max_traj_length
        sampled_indexed_traj = []

        iteration_score = 0
        cnt_steps = 0
        while cnt_steps < self.hyper_params.n_samples_epoch:
            state = self.env.reset()
            sampled_traj = IndexedTraj()
            done = False

            i_episode_step = 0
            episode_score = 0
            while not done and i_episode_step < max_traj_length:
                if self.args.render and self.i_iteration % int(self.args.render_freq) == 0:
                    self.env.render()

                # get current predicate
                predicate_values = None
                if 'get_current_predicate' in dir(self.env):
                    predicate_values = self.env.get_current_predicate()

                # get the controller_policy network's action in the format of a vector, e.g. [0.0, 1.0, 0.0]
                action = self.policy.select_action(state)

                # interact with the environment
                if self.is_discrete:
                    action_id = common_utils.one_hot_to_discrete_action(action)
                    next_state, true_reward, done, info = self.env.step(action_id)
                else:
                    next_state, true_reward, done, info = self.env.step(action)
                # in guided cost learning, there is no true reward, so we ignore it
                sampled_traj.add_transition([state, action, true_reward, next_state, done])

                episode_score += true_reward

                # logging
                if self.logger:
                    action_taken = action
                    if self.is_discrete:
                        action_taken = common_utils.one_hot_to_discrete_action(action)
                    # log transition
                    self.logger.add_transition(state, action_taken, true_reward, next_state, done,
                                               is_save_utility=False, predicate_values=predicate_values,
                                               utility_map=None, utility_values=None)
                state = next_state
                self.total_step += 1
                i_episode_step += 1

            cnt_steps += i_episode_step
            iteration_score += episode_score
            sampled_indexed_traj.append(sampled_traj.traj_dict)
            # write log (episode score)
            if self.logger:
                self.write_episode_log(episode_score)
            self.i_episode += 1

        return sampled_indexed_traj, float(iteration_score)/len(sampled_indexed_traj)

    def update_policy(self, indexed_traj):
        """
        Update/optimize the controller_policy network
        :return: return training related information (may vary based on the controller_policy network algorithms)
        """
        return self.policy.update_policy(indexed_traj, self)

    def update_irl(self, indexed_sampled_trajs):
        """
        update the discriminator
        :param: indexed_sampled_traj: the sampled trajectories from current iteration
        """
        # calculate the expert batch size
        n_expert = len(self.demo_buf)
        if self.hyper_params.traj_batch_size > 0:
            expert_batch_size = min(n_expert, self.hyper_params.traj_batch_size)
        else:
            expert_batch_size = n_expert

        # sample batch_size
        n_sampled_trajs = len(self.sampled_traj_buf)
        if self.hyper_params.is_fusion:
            past_batch_size = min(n_sampled_trajs, max(0, self.hyper_params.traj_batch_size-len(indexed_sampled_trajs)))
        else:
            past_batch_size = 0

        losses_iteration = []
        demo_predict_acc = []
        sampled_predict_acc = []
        n_sample_trained = 0
        for it in range(self.hyper_params.num_iteration_update):
            # sampled expert trajectories
            demo_indices = np.random.randint(low=0, high=n_expert, size=expert_batch_size)
            demo_states, demo_actions, _, _, _ = self.demo_buf.get_experiences_from_trajs(indices=demo_indices)

            # sampled sample trajectories
            # if traj_batch_size is negative, use all sampled trajectories
            if self.hyper_params.traj_batch_size < 0:
                sample_indices = np.random.randint(low=0, high=n_sampled_trajs, size=n_sampled_trajs)
                sample_states, sample_actions, _, _, _ = self.sampled_traj_buf.get_experiences_from_trajs(indices=sample_indices)
            else:
                past_trajs = []
                if past_batch_size > 0:
                    past_trajs = self.sampled_traj_buf.sample(batch_size=past_batch_size)
                sampled_trajs = indexed_sampled_trajs + past_trajs
                sample_states, sample_actions, _, _, _ = extract_experiences_from_indexed_trajs(sampled_trajs)

            # update discriminator
            criterion = torch.nn.BCELoss()
            n_demo_states = demo_states.shape[0]
            n_sampled_states = sample_states.shape[0]

            arrange = np.arange(min(n_demo_states, n_sampled_states))
            np.random.shuffle(arrange)
            n_sample_trained = len(arrange)
            batch_size = int(self.hyper_params.batch_size/2)
            for i in range(arrange.shape[0] // batch_size):
                start_idx = batch_size * i
                end_idx = batch_size * (i + 1)
                batch_index = arrange[start_idx:end_idx]

                # demo states
                batch_demo_states = demo_states[batch_index]
                batch_demo_actions = demo_actions[batch_index]
                # sample states
                batch_sample_states = sample_states[batch_index]
                batch_sample_actions = sample_actions[batch_index]

                demo_states_actions = Variable(torch.from_numpy(np.concatenate((batch_demo_states, batch_demo_actions), axis=1))).type(
                    torch.FloatTensor).to(device)
                sampled_states_actions = Variable(torch.from_numpy(np.concatenate((batch_sample_states, batch_sample_actions), axis=1))).type(
                    torch.FloatTensor).to(device)

                demo_prediction = self.irl_model(demo_states_actions)
                sampled_prediction = self.irl_model(sampled_states_actions)
                demo_pred_labels = torch.ones((demo_states_actions.shape[0], 1)).to(device)
                sampled_pred_labels = torch.zeros((sampled_states_actions.shape[0], 1)).to(device)

                discrim_loss = criterion(demo_prediction, demo_pred_labels) \
                               + criterion(sampled_prediction, sampled_pred_labels)

                demo_predict_acc.append(((demo_prediction > 0.5).float()).mean())
                sampled_predict_acc.append(((sampled_prediction < 0.5).float()).mean())

                losses_iteration.append(discrim_loss.item())

                self.irl_model_optim.zero_grad()
                discrim_loss.backward()
                self.irl_model_optim.step()

        avg_loss = sum(losses_iteration)/float(len(losses_iteration))
        demo_avg_acc = sum(demo_predict_acc)/float(len(demo_predict_acc))
        sampled_avg_acc = sum(sampled_predict_acc)/float(len(sampled_predict_acc))
        return avg_loss, demo_avg_acc, sampled_avg_acc, n_sample_trained

    def write_log(self, log_value):
        pass

    def write_episode_log(self, log_value):
        episode_score = log_value
        if self.args.print_episode_log:
            print("[INFO] iteration %d, episode %d, total step: %d, score: %d"
                  % (self.i_iteration, self.i_episode, self.total_step, episode_score))

        if self.logger is not None:
            self.logger.log_wandb({
                'score': episode_score
            })

    def write_iteration_log(self, log_value, policy_update_info):
        """
        Write log about loss and score
        """
        i_iteration, iteration_avg_score, discriminator_avg_loss, demo_avg_acc, sampled_avg_acc, iteration_time_cost, n_sample_trained = log_value

        print(
            "\n[INFO] iteration %d, total episode: %d, total step: %d, avg score: %d\n"
            "discriminator loss: %.6f, demo prediction accuracy: %.3f%%, samples prediction accuracy: %.3f%%\n"
            "iteration time cost: %.3f, number of samples (expert + sampled) used to train (per iteration): %d"
            % (
                self.i_iteration,
                self.i_episode,
                self.total_step,
                iteration_avg_score,
                discriminator_avg_loss,
                demo_avg_acc*100,
                sampled_avg_acc*100,
                iteration_time_cost,
                n_sample_trained
            )
        )

        self.policy.write_policy_log(policy_update_info)

        if self.logger is not None:
            wandb_log = {
                'score': iteration_avg_score,
                'discriminator_loss': discriminator_avg_loss,
                'demo accuracy': demo_avg_acc,
                'sampled accuracy': sampled_avg_acc
            }
            wandb_log.update(self.policy.get_policy_wandb_log(policy_update_info))
            self.logger.log_wandb(wandb_log)

    # pylint: disable=no-self-use, unnecessary-pass
    def pretrain(self):
        """ Pre-training steps. GCL doesn't do pre-train """
        pass

    def select_action(self, state):
        pass

    def step(self, action):
        pass

    def update_model(self):
        pass

    def get_wandb_watch_list(self):
        return [self.irl_model]

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
            indexed_sampled_trajs, iteration_avg_score = self.sample_trajectories()
            indexed_sampled_trajs = split_fixed_length_indexed_traj(indexed_sampled_trajs, self.hyper_params.traj_fixed_length)
            self.sampled_traj_buf.extend(indexed_sampled_trajs)

            # update discriminator
            discriminator_avg_loss, demo_avg_acc, sampled_avg_acc, n_sample_trained = self.update_irl(indexed_sampled_trajs)
            # re-evaluate trajectories and update the controller_policy
            policy_losses = self.update_policy(indexed_sampled_trajs)

            t_end = time.time()
            iteration_time_cost = t_end - t_begin

            # logging
            if self.logger:
                discriminator_log_values = (
                    i_iteration,
                    iteration_avg_score,
                    discriminator_avg_loss,
                    demo_avg_acc,
                    sampled_avg_acc,
                    iteration_time_cost,
                    n_sample_trained
                )
                self.write_iteration_log(discriminator_log_values, policy_losses)

            if i_iteration % self.args.save_period == 0:
                self.save_params(i_iteration)




