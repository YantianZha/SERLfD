import argparse
import time

import gym
import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
import scipy.optimize

from learning_agents.rl.trpo.trpo_utils import set_flat_params_to, get_flat_grad_from, get_flat_params_from, \
    normal_log_density
from learning_agents.utils.utils import IndexedTraj
from learning_agents.common.trajectory_buffer import TrajectoryBuffer
from learning_agents.common import common_utils
from learning_agents.rl.trpo.trpo_models import TrpoActor, TrpoCritic, TrpoActorCNN, TrpoCriticCNN
from learning_agents.agent import Agent
from utils.trajectory_utils import split_fixed_length_indexed_traj, extract_experiences_from_indexed_trajs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# noinspection PyArgumentList
class TRPOAgent(Agent):
    """
    TRPO agent interacting with environment.
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
        return self._update_trpo(self.new_sampled_trajs)

    def _conjugate_gradients(self, Avp, b, nsteps, residual_tol=1e-10):
        x = torch.zeros(b.size()).to(device)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for i in range(nsteps):
            _Avp = Avp(p)
            alpha = rdotr / torch.dot(p, _Avp)
            x += alpha * p
            r -= alpha * _Avp
            new_rdotr = torch.dot(r, r)
            betta = new_rdotr / rdotr
            p = r + betta * p
            rdotr = new_rdotr
            if rdotr < residual_tol:
                break
        return x

    def _line_search(self, model, f, x, fullstep, expected_improve_rate, max_backtracks=10, accept_ratio=.1):
        fval = f(True).data
        for (_n_backtracks, stepfrac) in enumerate(.5 ** np.arange(max_backtracks)):
            xnew = x + stepfrac * fullstep
            set_flat_params_to(model, xnew)
            newfval = f(True).data
            actual_improve = fval - newfval
            expected_improve = expected_improve_rate * stepfrac
            ratio = actual_improve / expected_improve

            if ratio.item() > accept_ratio and actual_improve.item() > 0:
                return True, xnew
        return False, x

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

    # noinspection PyTypeChecker
    def _update_trpo(self, indexed_sampled_trajs):
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

            with torch.no_grad():
                old_values = self.v_critic(torch_states)
            # get advantage estimation from the trajectories
            torch_advantages, torch_returns = self._estimate_advantages(torch_rewards, torch_masks, old_values, self.hyper_params.gamma, self.hyper_params.tau, device)

            # update critic
            def get_value_loss(flat_params):
                set_flat_params_to(self.v_critic, torch.Tensor(flat_params))
                for param in self.v_critic.parameters():
                    if param.grad is not None:
                        param.grad.data.fill_(0)

                values_ = self.v_critic(Variable(torch_states))

                value_loss = (values_ - torch_returns).pow(2).mean()
                critic_losses.append(value_loss.item())

                # weight decay
                for param in self.v_critic.parameters():
                    value_loss += param.pow(2).sum() * self.optim_cfg.weight_decay
                value_loss.backward()
                return value_loss.data.double().cpu().numpy(), get_flat_grad_from(self.v_critic).data.double().cpu().numpy()

            flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(self.v_critic).detach().cpu().numpy(), maxiter=2)
            set_flat_params_to(self.v_critic, torch.tensor(flat_params))

            # update controller_policy
            action_means, action_stds = self.actor(Variable(torch_states))
            action_log_stds = action_stds.log()
            fixed_log_prob = normal_log_density(Variable(torch_actions), action_means, action_log_stds,
                                                action_stds).data.clone()

            def get_loss(volatile=False):
                if volatile:
                    with torch.no_grad():
                        action_means, action_stds = self.actor(Variable(torch_states))
                        action_log_stds = action_stds.log()
                else:
                    action_means, action_stds = self.actor(Variable(torch_states))
                    action_log_stds = action_stds.log()

                log_prob = normal_log_density(Variable(torch_actions), action_means, action_log_stds, action_stds)
                action_loss = - Variable(torch_advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
                return action_loss.mean()

            # noinspection PyUnresolvedReferences
            def get_kl():
                mean1, std1 = self.actor(Variable(torch_states))
                log_std1 = std1.log()

                mean0 = Variable(mean1.data)
                log_std0 = Variable(log_std1.data)
                std0 = Variable(std1.data)
                kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
                return kl.sum(1, keepdim=True)

            # do trpo step
            policy_loss = get_loss()
            actor_losses.append(float(policy_loss))
            grads = torch.autograd.grad(policy_loss, self.actor.parameters())
            loss_grad = torch.cat([grad.view(-1) for grad in grads]).data

            def Fvp(v):
                kl = get_kl()
                kl = kl.mean()

                grads = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
                flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

                kl_v = (flat_grad_kl * Variable(v)).sum()
                grads = torch.autograd.grad(kl_v, self.actor.parameters())
                flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

                return flat_grad_grad_kl + v * self.hyper_params.damping

            stepdir = self._conjugate_gradients(Fvp, -loss_grad, 10)

            shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)

            lm = torch.sqrt(shs / self.hyper_params.max_kl)
            fullstep = stepdir / lm[0]

            neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)

            prev_params = get_flat_params_from(self.actor)
            success, new_params = self._line_search(self.actor, get_loss, prev_params, fullstep,
                                             neggdotstepdir / lm[0])
            set_flat_params_to(self.actor, new_params)

        return np.mean(actor_losses), np.mean(critic_losses)

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

