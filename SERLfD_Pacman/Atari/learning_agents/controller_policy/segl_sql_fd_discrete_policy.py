import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_

from learning_agents.common.priortized_replay_buffer import PrioritizedReplayBuffer
from learning_agents.common.replay_buffer import ReplayBuffer, load_stacked_demos_and_utility
from learning_agents.common import common_utils
from learning_agents.controller_policy.segl_base_policy import SEGL_BasePolicy
from learning_agents.rl.soft_q_learning.soft_dqfd_discrete import SQLfD_Discrete_Agent
from utils.trajectory_utils import get_n_step_info_from_traj

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LOG_REG = 1e-8


class SEGL_SQLfD_Discrete_Policy(SQLfD_Discrete_Agent, SEGL_BasePolicy):
    """
    SAC from Demonstration Discrete Policy used in SEGL
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
        SEGL_BasePolicy.__init__(self, env, args, log_cfg, hyper_params, network_cfg, optim_cfg, logger=None)
        SQLfD_Discrete_Agent.__init__(self, env, args, log_cfg, hyper_params, network_cfg, optim_cfg, demo_preprocessor,
                                      encoder=encoder, logger=logger)
        self.update_step = 0

    # pylint: disable=attribute-defined-outside-init
    def _initialize(self):
        """Initialize non-common things."""
        if not self.args.test:
            # get flatten demos
            demos, utility_info = self._load_demos()
            print('[Controller INFO] demo observation shape: ', demos[0][0].shape)
            # replay memory for multi-steps
            if self.use_n_step:
                demos, demos_n_step, utility_info, n_step_utility_info = get_n_step_info_from_traj(demos, self.hyper_params.n_step,
                                                                                                   self.hyper_params.gamma, utility_info=utility_info)
                self.memory_n = ReplayBuffer(
                    self.hyper_params.buffer_size,
                    batch_size=self.hyper_params.batch_size,
                    n_step=self.hyper_params.n_step,
                    gamma=self.hyper_params.gamma,
                    demo=demos_n_step,
                    demo_utility_info=n_step_utility_info
                )

            # replay memory for a single step
            if self.use_prioritized:
                self.memory = PrioritizedReplayBuffer(
                    self.hyper_params.buffer_size,
                    self.hyper_params.batch_size,
                    demo=demos,
                    demo_utility_info=utility_info,
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
                    demo_utility_info=utility_info,
                    gamma=self.hyper_params.gamma,
                )

    def _init_from_file(self):
        # load the optimizer and model parameters
        if self.args.policy_load_from is not None:
            self.load_params(self.args.policy_load_from)

    def _load_demos(self):
        return load_stacked_demos_and_utility(self.args.demo_path, self.args.demo_util_path,
                                              self.is_discrete, self.hyper_params.discrete_to_one_hot, self.action_dim,
                                              self.demo_preprocessor, self.hyper_params.frame_stack)

    def add_transition_to_memory(self, transition, utility_info=None):
        """Add 1 step and n step transitions to memory."""
        # add n-step transition
        if self.use_n_step:
            transition_to_save = self.memory_n.add(transition, utility_info=utility_info)
            if transition_to_save:
                transition, utility_info = transition_to_save

        # add a single step transition
        # if transition is not an empty tuple
        if transition:
            self.memory.add(transition, utility_info=utility_info)

    def get_action(self, state, is_test=False, is_random=False, info=None):
        if (not is_test and self.epsilon > np.random.random()) or is_random:
            selected_action = np.array(self.env.action_space.sample())
        else:
            state = self._preprocess_state(state)
            self.dqn.eval()
            with torch.no_grad():
                if is_test or self.args.deterministic:
                    selected_action = self.dqn(state).argmax()
                else:
                    q_values = ((1 / self.alpha) * self.dqn(state))
                    action_probs = F.log_softmax(q_values, dim=-1).exp()
                    dist = Categorical(action_probs)
                    selected_action = dist.sample()
            self.dqn.train()
            selected_action = selected_action.detach().cpu().numpy()
        return selected_action

    # noinspection DuplicatedCode
    def update_policy(self, indexed_trajs, irl_model):
        """ DQfD policy doesn't use indexed_trajs. Instead it uses transitions inside its own buffer """
        # 1 step loss
        n_sample = min(len(self.memory), self.hyper_params.batch_size)
        if self.use_prioritized:
            experiences_one_step = self.memory.sample(self.per_beta, is_to_tensor=False)
            torch_weights, indices, eps_d = experiences_one_step[-3:]
            indices = np.array(indices)
            # re-normalize the weights such that they sum up to the value of batch_size
            torch_weights = self._to_float_tensor(torch_weights)
            torch_weights = torch_weights / torch.sum(torch_weights) * float(n_sample)
        else:
            indices = np.random.choice(len(self.memory), size=n_sample, replace=False)
            eps_d = np.zeros_like(indices)
            eps_d[np.where(indices < self.memory.demo_size)] = self.hyper_params.per_eps_demo
            torch_weights = torch.from_numpy(np.ones(shape=(indices.shape[0], 1), dtype=np.float64)).type(
                torch.FloatTensor).to(device)
            experiences_one_step = self.memory.sample(indices=indices, is_to_tensor=False)
        predicates_one_step, next_predicates_one_step = self.memory.get_utility_info(indices)
        torch_actions = self._to_float_tensor(experiences_one_step[1])
        torch_dones = self._to_float_tensor(experiences_one_step[4])

        # incorporate information from explainer
        torch_states, torch_rewards, torch_next_states = self.augment_experience(irl_model, experiences_one_step,
                                                                                 predicates_one_step,
                                                                                 next_predicates_one_step,
                                                                                 is_to_tensor=True)
        experiences_one_step = [torch_states, torch_actions, torch_rewards, torch_next_states, torch_dones]

        dq_loss_element_wise, q_values = self._get_dqn_loss(experiences_one_step, self.hyper_params.gamma)
        dq_loss = torch.mean(dq_loss_element_wise * torch_weights)

        # n step loss
        if self.use_n_step:
            experiences_n = self.memory_n.sample(indices, is_to_tensor=False)
            predicates_n_step, next_predicates_n_step = self.memory_n.get_utility_info(indices)
            torch_n_step_actions = self._to_float_tensor(experiences_n[1])
            torch_n_step_dones = self._to_float_tensor(experiences_n[4])
            torch_n_step_states, torch_n_step_rewards, torch_n_step_next_states = self.augment_experience(irl_model, experiences_n,
                                                                                                         predicates_n_step,
                                                                                                         next_predicates_n_step,
                                                                                                         is_to_tensor=True)
            experiences_n = [torch_n_step_states, torch_n_step_actions, torch_n_step_rewards, torch_n_step_next_states, torch_n_step_dones]
            gamma = self.hyper_params.gamma ** self.hyper_params.n_step
            dq_loss_n_element_wise, q_values_n = self._get_dqn_loss(experiences_n, gamma)

            # to update loss and priorities
            q_values = 0.5 * (q_values + q_values_n)
            # mix of 1-step and n-step returns
            dq_loss_element_wise += dq_loss_n_element_wise * self.hyper_params.lambda1
            dq_loss = torch.mean(dq_loss_element_wise * torch_weights)

        # q_value regularization (not used when w_q_reg is set to 0)
        q_regular = torch.norm(q_values, 2).mean() * self.optim_cfg.w_q_reg

        # supervised loss using demo for only demo transitions
        demo_idxs = np.where(indices < self.memory.demo_size)
        n_demo = demo_idxs[0].size
        if n_demo != 0:  # if 1 or more demos are sampled
            # get margin for each demo transition
            action_idxs = torch_actions[demo_idxs].long()
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
            new_priorities += eps_d
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

    # pylint: disable=no-self-use, unnecessary-pass
    def pretrain_policy(self, irl_model):
        """ Pretraining steps."""
        pretrain_loss = list()
        pretrain_step = self.hyper_params.pretrain_step
        print("\n[Controller INFO] Start pre-training for %d step." % pretrain_step)

        for i_step in range(1, pretrain_step + 1):
            loss = self.update_policy(indexed_trajs=None, irl_model=irl_model)
            pretrain_loss.append(loss)  # for logging

            # logging
            if i_step == 1 or i_step % 100 == 0:
                loss_info = np.vstack(pretrain_loss).mean(axis=0)
                pretrain_loss.clear()
                print("\n[Controller INFO] Pre-Train step %d, total loss: %f, dq loss: %f, supervised loss: %f\n"
                      "avg q values: %f, demo num in mini-batch: %d"
                      % (
                          i_step,
                          loss_info[0],
                          loss_info[1],
                          loss_info[2],
                          loss_info[3],
                          loss_info[4]
                      ))
        print("\n[Controller INFO] Pre-Train Complete!\n")

    def evaluate_states_actions(self, states, actions, info=None):
        return self.evaluate_state_action(states, actions).detach().cpu().numpy()

    def evaluate_state_action(self, state, action, info=None):
        """
        Compute the probabilities/densities of the state-action pair.
        state: np.ndarray
        action: np.ndarray
        """
        torch_actions = torch.FloatTensor(action).to(device).unsqueeze(-1)
        torch_states = self._preprocess_state(state)

        self.dqn.eval()
        with torch.no_grad():
            q_values = ((1 / self.alpha) * self.dqn(torch_states))
            action_probs = F.log_softmax(q_values, dim=-1).exp()
            action_probs = action_probs.gather(1, torch_actions.long())
            action_probs = torch.clamp(action_probs, min=LOG_REG, max=1.0)
            log_probs = action_probs.log()
        self.dqn.train()

        return log_probs

    def get_wandb_watch_list(self):
        return [self.dqn, self.dqn_target]

    def write_policy_log(self, log_value):
        loss_info = log_value
        print(
            "\n[Controller INFO] epsilon: %f, total loss: %f, dq loss: %f, supervised loss: %f\n"
            "avg q values: %f, alpha: %.3f, demo num in mini-batch: %d"
            % (
                self.epsilon,
                loss_info[0],
                loss_info[1],
                loss_info[2],
                loss_info[3],
                self.alpha,
                loss_info[4]
            )
        )

    def get_policy_wandb_log(self, log_value):
        loss_info = log_value
        wandb_log = {
            "epsilon": self.epsilon,
            "total loss": loss_info[0],
            "dqn loss": loss_info[1],
            "supervised loss": loss_info[2],
            "avg q values": loss_info[3],
            "demo num": loss_info[4],
            "supervised weight": self.lambda2,
            "alpha": self.alpha,
        }
        return wandb_log

    def do_post_episode_update(self, *argv):
        total_step, init_random_actions = argv
        if total_step >= init_random_actions:
            # decrease epsilon
            self.epsilon = max(self.min_epsilon, self.hyper_params.epsilon_decay * self.epsilon)
            # decrease alpha
            self.alpha = max(self.hyper_params.min_alpha, self.hyper_params.alpha_decay * self.alpha)
        self.lambda2 = max(self.hyper_params.min_lambda2, self.hyper_params.lambda2_decay * self.lambda2)







