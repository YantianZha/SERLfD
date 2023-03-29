import argparse

import gym
import numpy as np
import torch
from torch.distributions import Independent, Normal

from learning_agents.rl.trpo.trpo_utils import set_flat_params_to, get_flat_grad_from, get_flat_params_from, \
    normal_log_density
from learning_agents.rl.trpo.trpo import TRPOAgent
from learning_agents.controller_policy.base_policy import BasePolicy
from torch.autograd import Variable
import scipy.optimize
from utils.trajectory_utils import extract_experiences_from_indexed_trajs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# noinspection DuplicatedCode
class TRPOPolicy(BasePolicy, TRPOAgent):
    """
    SAC agent interacting with environment.

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
        actor_target (nn.Module): target actor model to select actions
        actor_optim (Optimizer): optimizer for training actor
        critic_1 (nn.Module): critic model to predict state values
        critic_2 (nn.Module): critic model to predict state values
        critic_target1 (nn.Module): target critic model to predict state values
        critic_target2 (nn.Module): target critic model to predict state values
        critic_optim1 (Optimizer): optimizer for training critic_1
        critic_optim2 (Optimizer): optimizer for training critic_2
        curr_state (np.ndarray): temporary storage of the current state
        total_step (int): total step numbers
        episode_step (int): step number of the current episode
        update_step (int): step number of updates
        i_episode (int): current episode number
        target_entropy (int): desired entropy used for the inequality constraint
        log_alpha (torch.Tensor): weight for entropy
        alpha_optim (Optimizer): optimizer for alpha
    """

    def __init__(self, env, args, log_cfg, hyper_params, network_cfg, optim_cfg, encoder=None, logger=None):
        """
        Initialize.
        Args:
            env (gym.Env): openAI Gym environment
            args (argparse.Namespace): arguments including hyperparameters and training settings
        """
        BasePolicy.__init__(self, env, args, log_cfg, hyper_params, network_cfg, optim_cfg, logger=logger)
        TRPOAgent.__init__(self, env, args, log_cfg, hyper_params, network_cfg, optim_cfg, encoder=encoder, logger=logger)
        self.used_as_policy = True

    def evaluate_states_actions(self, states, actions, info=None):
        """
        get the log probability of actions
        :param: states: np.ndarray
        :param: actions: np.ndarray
        :return: a numpy array of log probability of each action
        """
        return self.evaluate_state_action(states, actions).detach().cpu().numpy()

    def evaluate_state_action(self, state, action, info=None):
        """
        Compute the probability of the state, action pair
        :param action: vector of the action (if discrete, it should be the one-hot representation)
        :return: the log probability (torch.Tensor) of the state, action pair
        """
        if not isinstance(action, torch.Tensor):
            if not isinstance(action, np.ndarray):
                action = torch.Tensor(action).type(torch.FloatTensor).to(device)
            else:
                action = torch.from_numpy(action).type(torch.FloatTensor).to(device)
        state = self._preprocess_state(state)
        mean, std = self.actor(state)

        dist = Independent(Normal(mean, std), 1)
        return dist.log_prob(action)

    def update_policy(self, indexed_sampled_trajs, irl_model):
        """ Train the model after each episode. """
        actor_losses = []
        critic_losses = []

        # sample batch_size
        n_sampled_trajs = len(irl_model.sampled_traj_buf)
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
                states, actions, rewards, next_states, dones = irl_model.sampled_traj_buf.get_experiences_from_trajs(
                    indices=sample_indices)
            # if it doesn't use all past-sampled trajectories
            else:
                past_trajs = []
                if past_batch_size > 0:
                    past_trajs = irl_model.sampled_traj_buf.sample(batch_size=past_batch_size)
                sampled_trajs = indexed_sampled_trajs + past_trajs
                states, actions, rewards, next_states, dones = extract_experiences_from_indexed_trajs(sampled_trajs)

            # let the irl module re-evaluate the reward
            rewards = irl_model.get_reward(states, actions)

            masks = 1.0 - dones
            torch_states = self._preprocess_state(states)
            torch_actions = torch.from_numpy(actions).type(torch.FloatTensor).to(device)
            torch_masks = torch.from_numpy(masks).type(torch.FloatTensor).to(device)
            torch_rewards = torch.from_numpy(rewards).type(torch.FloatTensor).to(device)

            with torch.no_grad():
                old_values = self.v_critic(torch_states)
            # get advantage estimation from the trajectories
            torch_advantages, torch_returns = self._estimate_advantages(torch_rewards, torch_masks, old_values,
                                                                        self.hyper_params.gamma, self.hyper_params.tau,
                                                                        device)

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
                return value_loss.data.double().cpu().numpy(), get_flat_grad_from(
                    self.v_critic).data.double().cpu().numpy()

            flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(
                self.v_critic).detach().cpu().numpy(), maxiter=25)
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

    def write_policy_log(self, log_value):
        actor_loss, critic_loss = log_value
        print(
            "controller_policy actor loss: %.3f, critic loss: %.3f"
            % (
                actor_loss,
                critic_loss
            )
        )

    def get_policy_wandb_log(self, log_value):
        actor_loss, critic_loss = log_value
        wandb_log = {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss
        }
        return wandb_log

    def load_params(self, path):
        """Load model and optimizer parameters."""
        TRPOAgent.load_params(self, path)

    def save_params(self, n_episode):
        TRPOAgent.save_params(self, n_episode)

    def get_wandb_watch_list(self):
        return [self.actor, self.v_critic]



