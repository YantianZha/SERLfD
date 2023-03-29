import argparse
import time

import gym
import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.distributions import Independent, Normal

from learning_agents.utils.math_utils import normal_log_density
from learning_agents.utils.utils import IndexedTraj
from learning_agents.common.trajectory_buffer import TrajectoryBuffer
from learning_agents.common import common_utils
from learning_agents.rl.trpo.trpo_models import TrpoActor, TrpoCritic, TrpoActorCNN, TrpoCriticCNN
from learning_agents.agent import Agent
from utils.trajectory_utils import split_fixed_length_indexed_traj, extract_experiences_from_indexed_trajs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# noinspection PyArgumentList
class PPOAgent(Agent):
    """
    PPO agent interacting with environment.
    Attrtibutes:
        env (gym.Env): openAI Gym environment
        args (argparse.Namespace): arguments including hyperparameters and training settings
        hyper_params (ConfigDict): hyper-parameters
        network_cfg (ConfigDict): config of network for training agent
        optim_cfg (ConfigDict): config of optimizer
        state_dim (int): state size of env
        action_dim (int): action size of env
        memory (ReplayBuffer): replay memory
        actor (nn.Module): actor model to select actions
        actor_optim (Optimizer): optimizer for training actor
        v_critic_1 (nn.Module): critic model to predict state values
        v_critic_optim1 (Optimizer): optimizer for training critic_1
        total_step (int): total step numbers
        i_episode (int): current episode number
    """

    def __init__(self, env, args, log_cfg, hyper_params, network_cfg, optim_cfg, encoder=None, logger=None):
        """
        Initialize.
        Args:
            env (gym.Env): openAI Gym environment
            args (argparse.Namespace): arguments including hyperparameters and training settings
        """
        Agent.__init__(self, env, args, log_cfg)

        self.curr_state = np.zeros((1,))
        self.total_step = 0
        self.i_episode = 0
        self.logger = logger
        self.encoder = encoder

        self.hyper_params = hyper_params
        self.network_cfg = network_cfg
        self.optim_cfg = optim_cfg

        # get state space info
        self.state_dim = self.env.observation_space.shape[0]
        # check if it's single channel or multi channel
        self.state_channel = 1 if len(self.env.observation_space.shape) <= 2 else self.env.observation_space.shape[0]

        # get action space info
        self.is_discrete = False
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.action_dim = self.env.action_space.n
            self.is_discrete = True
        else:
            self.action_dim = self.env.action_space.shape[0]

        self.new_sampled_trajs = None
        self.used_as_policy = False     # if used as controller_policy, no initial random action
        self._initialize()
        self._init_network()

    # pylint: disable=attribute-defined-outside-init
    def _initialize(self):
        """Initialize non-common things."""
        if not self.args.test:
            # replay memory
            self.sampled_traj_buf = TrajectoryBuffer(
                self.hyper_params.buffer_size, self.hyper_params.traj_batch_size
            )

    def _set_to_default_MLP(self):
        # create actor
        self.actor = TrpoActor(self.state_dim, self.action_dim).to(device)
        # create v_critic
        self.v_critic = TrpoCritic(self.state_dim).to(device)

    def _set_to_default_CNN(self):
        # create actor
        self.actor = TrpoActorCNN(input_channels=self.state_channel, num_outputs=self.action_dim, fc_input_size=768).to(device)
        # create v_critic
        self.v_critic = TrpoCriticCNN(input_channels=self.state_channel, num_outputs=1, fc_input_size=768).to(device)

    # pylint: disable=attribute-defined-outside-init
    def _init_network(self):
        """Initialize networks and optimizers."""
        if self.network_cfg.use_cnn:
            self._set_to_default_CNN()
        else:
            self._set_to_default_MLP()

        # create optimizers
        self.actor_optim = optim.Adam(
            self.actor.parameters(),
            lr=self.optim_cfg.lr_actor,
            # weight_decay=self.optim_cfg.weight_decay,
        )

        self.v_critic_optim = optim.Adam(
            self.v_critic.parameters(),
            lr=self.optim_cfg.lr_vf,
            weight_decay=self.optim_cfg.weight_decay,
        )

        # load the optimizer and model parameters
        if self.args.load_from is not None:
            self.load_params(self.args.load_from)

    def select_action(self, state):
        """
        Select an action from the input space.
        state: np.ndarray
        """
        self.curr_state = state
        state = self._preprocess_state(state)

        # if initial random action should be conducted
        if self.total_step < self.hyper_params.initial_random_action and not self.args.test and not self.used_as_policy:
            if self.is_discrete:
                return np.array(
                    common_utils.discrete_action_to_one_hot(self.env.action_space.sample(), self.action_dim))
            else:
                return np.array(self.env.action_space.sample())
        mean, std = self.actor(state)
        selected_action = torch.normal(mean, std).squeeze(0).detach().cpu().numpy()
        return selected_action

    # pylint: disable=no-self-use
    def _preprocess_state(self, state):
        """
        Preprocess state so that actor selects an action.
        state: np.ndarray
        """
        state = torch.FloatTensor(state).to(device)
        if self.encoder is not None:
            state = self.encoder(state)
        # TODO: unsqueeze when single state
        return state

    def step(self, action):
        """
        Take an action and return the response of the env.
        action: np.ndarray
        """
        next_state, reward, done, info = self.env.step(action)
        # TODO: max traj length
        return next_state, reward, done, info

    def _add_transition_to_memory(self, transition):
        """Add 1 step and n step transitions to memory."""
        pass

    def update_model(self):
        """Train the model after each episode."""
        return self._update_ppo(self.new_sampled_trajs)

    def _to_device(self, device, *args):
        return [x.to(device) for x in args]

    def _estimate_advantages(self, rewards, masks, values, gamma, tau, device):
        rewards, masks, values = self._to_device(device, rewards, masks, values)
        tensor_type = type(rewards)
        deltas = tensor_type(rewards.size(0), 1).to(device)
        advantages = tensor_type(rewards.size(0), 1).to(device)

        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(rewards.size(0))):
            deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
            advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

            prev_value = values[i, 0]
            prev_advantage = advantages[i, 0]

        returns = values + advantages
        advantages = (advantages - advantages.mean()) / advantages.std()

        advantages, returns = self._to_device(device, advantages, returns)
        return advantages, returns

    def _log_prob_density(self, states, actions):
        """
        return a torch.Tensor storing the log probs of all the state-action pairs
        """
        means, stds = self.actor(states)
        dist = Independent(Normal(means, stds), 1)
        return dist.log_prob(actions).unsqueeze(1)

    # noinspection PyTypeChecker
    def _update_ppo(self, indexed_sampled_trajs):
        actor_losses = []
        critic_losses = []

        # sample batch_size
        n_sampled_trajs = len(self.sampled_traj_buf)
        # if use past trajectories, calculate how many past trajectories it uses
        if self.hyper_params.is_fusion:
            past_batch_size = min(n_sampled_trajs,
                                  max(0, self.hyper_params.traj_batch_size - len(indexed_sampled_trajs)))
        else:
            past_batch_size = 0

        for it in range(self.hyper_params.num_iteration_update):
            # sampled sample trajectories
            # if traj_batch_size is negative, use all sampled trajectories
            if self.hyper_params.traj_batch_size < 0:
                sample_indices = np.random.randint(low=0, high=n_sampled_trajs, size=n_sampled_trajs)
                # use irl model's sampled_traj_buf, so we don't need two copies of trajectories
                states, actions, rewards, next_states, dones = self.sampled_traj_buf.get_experiences_from_trajs(
                    indices=sample_indices)
            # if it doesn't use all past-sampled trajectories
            else:
                past_trajs = []
                if past_batch_size > 0:
                    past_trajs = self.sampled_traj_buf.sample(batch_size=past_batch_size)
                sampled_trajs = indexed_sampled_trajs + past_trajs
                states, actions, rewards, next_states, dones = extract_experiences_from_indexed_trajs(sampled_trajs)

            masks = 1.0 - dones
            torch_states = self._preprocess_state(states)
            torch_actions = torch.from_numpy(actions).type(torch.FloatTensor).to(device)
            torch_masks = torch.from_numpy(masks).type(torch.FloatTensor).to(device)
            torch_rewards = torch.from_numpy(rewards).type(torch.FloatTensor).to(device)

            self._ppo_step(torch_states, torch_actions, torch_masks, torch_rewards, actor_losses, critic_losses)

            return np.mean(actor_losses), np.mean(critic_losses)

    def _ppo_step(self, torch_states, torch_actions, torch_masks, torch_rewards, actor_losses, critic_losses):
        with torch.no_grad():
            old_values = self.v_critic(torch_states)
            fixed_log_prob = self._log_prob_density(Variable(torch_states), Variable(torch_actions)).data.clone()
        # get advantage estimation from the trajectories
        torch_advantages, torch_returns = self._estimate_advantages(torch_rewards, torch_masks, old_values, self.hyper_params.gamma, self.hyper_params.tau, device)

        # perform mini-batch PPO update
        arrange = np.arange(torch_states.shape[0])
        np.random.shuffle(arrange)
        perm = torch.LongTensor(arrange).to(device)
        torch_states, torch_actions, torch_returns, torch_advantages, fixed_log_prob = \
            torch_states[perm], torch_actions[perm].clone(), torch_returns[perm].clone(), torch_advantages[perm].clone(), fixed_log_prob[perm].clone()

        batch_size = int(self.hyper_params.batch_size)
        for i in range(arrange.shape[0] // batch_size):
            start_idx = batch_size * i
            end_idx = batch_size * (i + 1)
            batch_index = torch.LongTensor(np.arange(start_idx, end_idx)).to(device)

            batch_states, batch_actions, batch_returns, batch_advantages, batch_fixed_log_probs = \
                torch_states[batch_index], torch_actions[batch_index], torch_returns[batch_index], torch_advantages[batch_index], fixed_log_prob[batch_index]

            # update critic
            for _ in range(self.hyper_params.num_critic_update):
                values_pred = self.v_critic(batch_states)
                value_loss = (values_pred - batch_returns).pow(2).mean()
                # weight decay
                for param in self.v_critic.parameters():
                    value_loss += param.pow(2).sum() * self.optim_cfg.weight_decay
                self.v_critic_optim.zero_grad()
                value_loss.backward()
                self.v_critic_optim.step()

                critic_losses.append(value_loss.item())

            # update controller_policy
            for _ in range(self.hyper_params.num_policy_update):
                log_probs = self._log_prob_density(batch_states, batch_actions)
                ratio = torch.exp(log_probs - batch_fixed_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.hyper_params.clip_epsilon, 1.0 + self.hyper_params.clip_epsilon) * batch_advantages
                policy_surr_loss = -torch.min(surr1, surr2).mean()

                self.actor_optim.zero_grad()
                policy_surr_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                self.actor_optim.step()
                actor_losses.append(policy_surr_loss.item())

    def load_params(self, path):
        """Load model and optimizer parameters."""
        Agent.load_params(self, path)

        params = torch.load(path, map_location=device)
        self.actor.load_state_dict(params["actor"])
        self.v_critic.load_state_dict(params["v_critic"])
        self.actor_optim.load_state_dict(params["actor_optim"])
        self.v_critic_optim.load_state_dict(params["v_critic_optim"])

        print("[INFO] loaded the model and optimizer from", path)

    # noinspection PyMethodOverriding
    def save_params(self, n_episode):  # type: ignore
        """Save model and optimizer parameters."""
        params = {
            "actor": self.actor.state_dict(),
            "v_critic": self.v_critic.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "v_critic_optim": self.v_critic_optim.state_dict(),
        }

        Agent.save_params(self, params, n_episode)
        if self.logger is not None:
            self.logger.save_models(params, postfix=str(n_episode), is_snapshot=True)

    def write_log(self, log_value):
        """Write log about loss and score"""
        i_episode, actor_loss, critic_loss, avg_score, time_cost = log_value

        print(
            "[INFO] episode %d, total step %d, total score: %.4f\n"
            "actor loss: %.3f critic loss: %.3f (spent %.6f)\n"
            % (
                i_episode,
                self.total_step,
                avg_score,
                actor_loss,
                critic_loss,
                time_cost
            )
        )

        if self.logger is not None:
            self.logger.log_wandb({
                'score': avg_score,
                'actor loss': actor_loss,
                'critic loss': critic_loss,
                'time cost': time_cost,
            })

    # pylint: disable=no-self-use, unnecessary-pass
    def pretrain(self):
        """ Pretraining steps."""
        pass

    def train(self):
        """ Train the agent."""
        if self.logger is not None:
            self.logger.watch_wandb([self.actor, self.v_critic])

        # pre-training if needed
        self.pretrain()

        for it in range(1, self.args.iteration_num + 1):
            n_samples = 0
            scores = []

            t_begin = time.time()
            new_sampled_trajs = []
            while n_samples < self.hyper_params.sample_size_iter:
                state = self.env.reset()
                done = False
                episode_score = 0
                sampled_traj = IndexedTraj()

                while not done:
                    if self.args.render \
                            and self.i_episode >= self.args.render_after \
                            and it % self.args.render_freq == 0:
                        self.env.render()

                    # get current predicate
                    predicate_values = {}
                    if 'get_current_predicate' in dir(self.env):
                        predicate_values = self.env.get_current_predicate()

                    action = self.select_action(state)
                    if self.is_discrete:
                        action_to_take = common_utils.one_hot_to_discrete_action(action, is_softmax=True)
                        action = common_utils.discrete_action_to_one_hot(action_to_take, self.action_dim)
                    else:
                        action_to_take = action
                    next_state, reward, done, _ = self.step(action_to_take)
                    sampled_traj.add_transition([state, action, reward, next_state, done])

                    # logging
                    if self.logger is not None:
                        action_taken = action
                        if self.is_discrete:
                            action_taken = common_utils.one_hot_to_discrete_action(action)
                        # log transition
                        self.logger.add_transition(state, action_taken, reward, next_state, done,
                                                   is_save_utility=True, predicate_values=predicate_values,
                                                   utility_map=None, utility_values=None)

                    self.total_step += 1
                    n_samples += 1
                    state = next_state
                    episode_score += reward

                fixed_len_indexed_trajs = split_fixed_length_indexed_traj([sampled_traj.traj_dict], self.hyper_params.traj_fixed_length)
                self.sampled_traj_buf.extend(fixed_len_indexed_trajs)
                new_sampled_trajs = new_sampled_trajs + fixed_len_indexed_trajs
                scores.append(float(episode_score))
                self.i_episode += 1

            t_end = time.time()
            time_cost = t_end - t_begin
            avg_score = np.mean(scores, dtype=np.float64)

            # train the agent
            self.new_sampled_trajs = new_sampled_trajs
            actor_loss, critic_loss = self.update_model()

            # logging
            if self.logger:
                log_value = (
                    self.i_episode,
                    actor_loss,
                    critic_loss,
                    avg_score,
                    time_cost,
                )
                self.write_log(log_value)

            if self.i_episode % self.args.save_period == 0:
                self.save_params(self.i_episode)

        # termination
        self.env.close()
        self.save_params(self.i_episode)

