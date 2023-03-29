import numpy as np
import torch

from torch.nn.utils import clip_grad_norm_
from learning_agents.common.priortized_replay_buffer import PrioritizedReplayBuffer
from learning_agents.common.replay_buffer import ReplayBuffer, load_stacked_demos_and_utility
from learning_agents.common import common_utils
from learning_agents.controller_policy.segl_base_policy import SEGL_BasePolicy
from learning_agents.rl.sac.sac_fd_discrete import SACfD_Discrete_Agent
from utils.trajectory_utils import get_n_step_info_from_traj

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LOG_REG = 1e-8


class SEGL_SACfD_Discrete_Policy(SACfD_Discrete_Agent, SEGL_BasePolicy):
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
        SACfD_Discrete_Agent.__init__(self, env, args, log_cfg, hyper_params, network_cfg, optim_cfg, demo_preprocessor,
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
        if is_random and not is_test:
            selected_action = np.array(self.env.action_space.sample())
        else:
            state = self._preprocess_state(state)
            deterministic = True if is_test else self.hyper_params.deterministic_action
            self.actor.eval()
            with torch.no_grad():
                # actor returns squashed action, log_probs, z, mean, std
                selected_action, _, _, _ = self.actor(state, deterministic=deterministic)
            self.actor.train()
            selected_action = selected_action.squeeze(0).squeeze(0).detach().cpu().numpy()
        return selected_action

    # noinspection DuplicatedCode
    def update_policy(self, indexed_trajs, irl_model):
        """ SAC policy doesn't use indexed_trajs. Instead it uses transitions inside its own buffer """
        self.update_step += 1

        # 1 step loss
        n_sample = min(len(self.memory), self.hyper_params.batch_size)
        if self.use_prioritized:
            experiences_one_step = self.memory.sample(self.per_beta, is_to_tensor=False)
            torch_weights, indices, eps_demo = experiences_one_step[-3:]
            indices = np.array(indices)
            # re-normalize the weights such that they sum up to the value of batch_size
            torch_weights = self._to_float_tensor(torch_weights)
            torch_weights = torch_weights / torch.sum(torch_weights) * float(n_sample)
        else:
            indices = np.random.choice(len(self.memory), size=n_sample, replace=False)
            eps_demo = np.zeros_like(indices)
            eps_demo[np.where(indices < self.memory.demo_size)] = self.hyper_params.per_eps_demo
            torch_weights = torch.from_numpy(np.ones(shape=(indices.shape[0], 1), dtype=np.float64)).type(
                torch.FloatTensor).to(device)
            experiences_one_step = self.memory.sample(indices=indices, is_to_tensor=False)

        predicates_one_step, next_predicates_one_step = self.memory.get_utility_info(indices)
        torch_actions = self._to_float_tensor(experiences_one_step[1])
        torch_actions = torch_actions.unsqueeze(-1)
        torch_dones = self._to_float_tensor(experiences_one_step[4])

        # incorporate information from explainer
        torch_states, torch_rewards, torch_next_states = self.augment_experience(irl_model, experiences_one_step,
                                                                                 predicates_one_step,
                                                                                 next_predicates_one_step,
                                                                                 is_to_tensor=True)
        _, _, log_probs, _ = self.actor(torch_states)
        curr_action_probs = log_probs.exp()
        curr_policy_actions = torch.argmax(curr_action_probs, dim=-1, keepdim=True)

        ###################
        ### Train Alpha ###
        ###################
        if self.hyper_params.auto_entropy_tuning:
            # alpha loss (equation 11 in the paper)
            log_new_action_probs = torch.sum(curr_action_probs * log_probs, dim=-1)
            alpha_loss = (
                    (-self.log_alpha * ((log_new_action_probs + self.target_entropy).detach())) * torch_weights
            ).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.zeros(1)
            alpha = self.hyper_params.w_entropy

        ##########################
        ### Policy 1-Step Loss ###
        ##########################
        actor_loss_info = self._get_policy_loss(sample_indices=indices,
                                                obs=torch_states, actions=torch_actions,
                                                policy_actions=curr_policy_actions,
                                                log_probs=log_probs, action_probs=curr_action_probs, alpha=alpha)
        actor_loss, actor_supervised_loss = actor_loss_info

        ###############################
        ### Q functions 1-Step Loss ###
        ###############################
        loss_info = self._get_qf_loss(obs=torch_states, actions=torch_actions, rewards=torch_rewards,
                                      next_obs=torch_next_states, dones=torch_dones,
                                      alpha=alpha, gamma=self.hyper_params.gamma,
                                      reward_clip=self.hyper_params.reward_clip)
        q1_loss_element_wise, q2_loss_element_wise, q1_values, q2_values = loss_info
        q1_loss = torch.mean(q1_loss_element_wise * torch_weights)
        q2_loss = torch.mean(q2_loss_element_wise * torch_weights)

        ###############################
        ### Q functions n-Step Loss ###
        ###############################
        if self.use_n_step:
            experiences_n = self.memory_n.sample(indices, is_to_tensor=False)
            predicates_n_step, next_predicates_n_step = self.memory_n.get_utility_info(indices)
            torch_n_step_actions = self._to_float_tensor(experiences_n[1]).unsqueeze(-1)
            torch_n_step_dones = self._to_float_tensor(experiences_n[4])
            torch_n_step_states, torch_n_step_rewards, torch_n_step_next_states = self.augment_experience(irl_model,
                                                                                                          experiences_n,
                                                                                                          predicates_n_step,
                                                                                                          next_predicates_n_step,
                                                                                                          is_to_tensor=True)
            gamma_n = self.hyper_params.gamma ** self.hyper_params.n_step

            loss_info_n = self._get_qf_loss(obs=torch_n_step_states, actions=torch_n_step_actions,
                                            rewards=torch_n_step_rewards, next_obs=torch_n_step_next_states,
                                            dones=torch_n_step_dones, alpha=alpha, gamma=gamma_n,
                                            reward_clip=self.hyper_params.reward_clip)
            q1_loss_element_wise_n, q2_loss_element_wise_n, q1_values_n, q2_values_n = loss_info_n
            q1_values = 0.5 * (q1_values + q1_values_n)
            q2_values = 0.5 * (q2_values + q2_values_n)

            # mix of 1-step and n-step returns
            q1_loss_element_wise += q1_loss_element_wise_n * self.hyper_params.w_n_step
            q2_loss_element_wise += q2_loss_element_wise_n * self.hyper_params.w_n_step
            q1_loss = torch.mean(q1_loss_element_wise * torch_weights)
            q2_loss = torch.mean(q2_loss_element_wise * torch_weights)

        ###################################
        ### Q functions Supervised Loss ###
        ###################################
        q1_supervised_loss, q2_supervised_loss, n_demo = self._get_qf_supervised_loss(sample_indices=indices,
                                                                                      actions=torch_actions,
                                                                                      q1_values=q1_values,
                                                                                      q2_values=q2_values)
        # total loss
        q1_loss = q1_loss + q1_supervised_loss
        q2_loss = q2_loss + q2_supervised_loss

        #######################
        ### Update Networks ###
        #######################
        # train Q functions
        self.qf1_optim.zero_grad()
        q1_loss.backward()
        if self.hyper_params.gradient_clip is not None:
            clip_grad_norm_(self.qf1.parameters(), self.hyper_params.gradient_clip)
        self.qf1_optim.step()

        self.qf2_optim.zero_grad()
        q2_loss.backward()
        if self.hyper_params.gradient_clip is not None:
            clip_grad_norm_(self.qf2.parameters(), self.hyper_params.gradient_clip)
        self.qf2_optim.step()

        # train actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        if self.hyper_params.gradient_clip is not None:
            clip_grad_norm_(self.actor.parameters(), self.hyper_params.gradient_clip)
        self.actor_optim.step()

        if self.update_step % self.hyper_params.target_update_freq == 0:
            # soft update target network
            common_utils.soft_update(self.qf1, self.target_qf1, self.hyper_params.tau)
            common_utils.soft_update(self.qf2, self.target_qf2, self.hyper_params.tau)

        #########################
        ### Update Priorities ###
        #########################
        # update priorities in PER
        if self.use_prioritized:
            qf_loss_element_wise = 0.5 * (q1_loss_element_wise + q2_loss_element_wise)
            loss_for_prior = qf_loss_element_wise.detach().cpu().numpy().squeeze()
            new_priorities = loss_for_prior + self.hyper_params.per_eps
            new_priorities += eps_demo
            # noinspection PyUnresolvedReferences
            self.memory.update_priorities(indices, new_priorities)

            # increase beta
            fraction = min(float(self.i_episode) / self.args.iteration_num, 1.0)
            self.per_beta = self.per_beta + fraction * (1.0 - self.per_beta)

        return (actor_loss.item(), actor_supervised_loss.mean().item(),
                q1_loss.item(), q2_loss.item(), alpha_loss.item(),
                q1_values.mean().item(), q2_values.mean().item(), n_demo)

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
                print("[Controller INFO] Pre-Train step %d, actor loss: %.3f, actor supervised loss: %.3f\n"
                      "qf1 loss: %.3f, qf2 loss: %.3f\n"
                      "alpha loss: %.3f, avg qf1: %.3f, avg qf2: %.3f, demo num in mini-batch: %d"
                      % (
                          i_step,
                          loss_info[0],  # actor loss
                          loss_info[1],  # actor bc loss
                          loss_info[2],  # qf1 loss
                          loss_info[3],  # qf2 loss
                          loss_info[4],  # alpha loss
                          loss_info[5],  # avg qf 1
                          loss_info[6],  # avg qf 2
                          loss_info[7],  # n demo
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
        _, _, log_probs, _ = self.actor(torch_states)
        log_probs = torch.clamp(log_probs.gather(1, torch_actions.long()), min=np.log(LOG_REG), max=0.0)
        return log_probs

    def get_wandb_watch_list(self):
        return [self.actor, self.qf1, self.qf2, self.target_qf1, self.target_qf2]

    def write_policy_log(self, log_value):
        loss_info = log_value
        print(
            "[Controller INFO] actor_loss: %.3f, actor_supervised_loss: %.3f, qf1_loss: %.3f, qf2_loss: %.3f\n"
            "alpha_loss: %.3f, alpha: %.3f, avg qf 1: %.3f, avg qf 2: %.3f, n demo: %d\n"
            % (
                loss_info[0],  # actor loss
                loss_info[1],  # actor bc loss
                loss_info[2],  # qf1 loss
                loss_info[3],  # qf2 loss
                loss_info[4],  # alpha loss
                self.log_alpha.exp().item(),
                loss_info[5],  # avg qf 1
                loss_info[6],  # avg qf 2
                loss_info[7],  # n demo
            )
        )

    def get_policy_wandb_log(self, log_value):
        avg_loss_info = log_value
        wandb_log = {
            "alpha": self.log_alpha.exp().item(),
            'actor loss': avg_loss_info[0],
            'actor supervised loss': avg_loss_info[1],
            'qf1 loss': avg_loss_info[2],
            'qf2 loss': avg_loss_info[3],
            'alpha loss': avg_loss_info[4],
            'demo num': avg_loss_info[7],
            'supervised weight': self.lambda2,
            'actor supervised weight': self.policy_bc_weight,
            'avg q1 values': avg_loss_info[5],
            'avg q2 values': avg_loss_info[6],
            'avg q values': 0.5 * (avg_loss_info[5] + avg_loss_info[6])
        }
        return wandb_log

    def do_post_episode_update(self, *argv):
        self.lambda2 = max(self.hyper_params.min_lambda2, self.hyper_params.lambda2_decay * self.lambda2)
        self.policy_bc_weight = max(self.hyper_params.min_policy_bc_weight,
                                    self.hyper_params.policy_bc_weight_decay * self.policy_bc_weight)








