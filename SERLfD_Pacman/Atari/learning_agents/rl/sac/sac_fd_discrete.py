# sac_fd_discrete.py

import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from learning_agents.common.priortized_replay_buffer import PrioritizedReplayBuffer
from learning_agents.common.replay_buffer import ReplayBuffer
from learning_agents.rl.sac.sac_discrete import SAC_Discrete_Agent
from utils.trajectory_utils import get_n_step_info_from_traj, stack_frames_in_traj
from learning_agents.common import common_utils
from utils.trajectory_utils import read_expert_demo, get_flatten_trajectories, demo_discrete_actions_to_one_hot


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SACfD_Discrete_Agent(SAC_Discrete_Agent):
    """
    SAC agent interacting with environment.
    Attrtibutes:
        memory (PrioritizedReplayBuffer): replay memory
        beta (float): beta parameter for prioritized replay buffer
        use_n_step (bool): whether or not to use n-step returns
    """

    def __init__(self, env, args, log_cfg, hyper_params, network_cfg, optim_cfg, demo_preprocessor, encoder=None,
                 logger=None):
        self.demo_preprocessor = demo_preprocessor
        self.lambda2 = hyper_params.max_lambda2
        self.policy_bc_weight = hyper_params.max_policy_bc_weight

        SAC_Discrete_Agent.__init__(self, env, args, log_cfg, hyper_params, network_cfg, optim_cfg,
                                    encoder=encoder, logger=logger)

    # pylint: disable=attribute-defined-outside-init
    def _initialize(self):
        """ Initialize non-common things. """
        if not self.args.test:
            # get flatten demos
            demos = self._load_demos()
            print('[INFO] demo observation shape: ', demos[0][0].shape)
            # replay memory for multi-steps
            if self.use_n_step:
                demos, demos_n_step = get_n_step_info_from_traj(demos, self.hyper_params.n_step,
                                                                self.hyper_params.gamma)
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

    # noinspection DuplicatedCode
    def _get_qf_supervised_loss(self, sample_indices, actions, q1_values, q2_values):
        demo_idxs = np.where(sample_indices < self.memory.demo_size)
        n_demo = demo_idxs[0].size
        if n_demo != 0:  # if 1 or more demos are sampled
            # get margin for each demo transition
            action_idxs = actions[demo_idxs].long()
            q1_margin = torch.ones(q1_values.size()) * self.hyper_params.margin
            q2_margin = torch.ones(q2_values.size()) * self.hyper_params.margin

            q1_margin[demo_idxs, action_idxs] = 0.0  # demo actions have 0 margins
            q1_margin = q1_margin.to(device)
            q2_margin[demo_idxs, action_idxs] = 0.0  # demo actions have 0 margins
            q2_margin = q2_margin.to(device)

            # calculate supervised loss
            demo_q1_values = q1_values[demo_idxs, action_idxs].squeeze()
            q1_supervised_loss = torch.max(q1_values + q1_margin, dim=-1)[0]
            q1_supervised_loss = q1_supervised_loss[demo_idxs] - demo_q1_values
            q1_supervised_loss = torch.mean(q1_supervised_loss) * self.lambda2

            demo_q2_values = q2_values[demo_idxs, action_idxs].squeeze()
            q2_supervised_loss = torch.max(q2_values + q2_margin, dim=-1)[0]
            q2_supervised_loss = q2_supervised_loss[demo_idxs] - demo_q2_values
            q2_supervised_loss = torch.mean(q2_supervised_loss) * self.lambda2
        else:  # no demo sampled
            q1_supervised_loss = torch.zeros(1, device=device)
            q2_supervised_loss = torch.zeros(1, device=device)

        return q1_supervised_loss, q2_supervised_loss, n_demo

    def _get_policy_loss(self, sample_indices, obs, actions, policy_actions, log_probs, action_probs, alpha):
        curr_actions_q_values = torch.min(
            self.qf1(obs),
            self.qf2(obs)
        )

        # actor loss (equation 12 in the paper)
        actor_loss = alpha * log_probs - curr_actions_q_values
        actor_loss = (action_probs * actor_loss).mean()

        # bc supervised loss
        demo_idxs = np.where(sample_indices < self.memory.demo_size)
        n_demo = demo_idxs[0].size
        if n_demo != 0:  # if 1 or more demos are sampled
            demo_actions = actions[demo_idxs]
            policy_demo_actions = policy_actions[demo_idxs]
            action_diff = torch.eq(demo_actions.long(), policy_demo_actions.long()).float() * self.hyper_params.policy_bc_margin
            supervised_loss = torch.mean(action_diff) * self.policy_bc_weight
        else:  # no demo sampled
            supervised_loss = torch.zeros(1, device=device)

        actor_loss = actor_loss + supervised_loss
        return actor_loss, supervised_loss

    # noinspection DuplicatedCode
    def update_model(self):
        """ Train the model after each episode. """
        self.update_step += 1

        # 1 step loss
        n_sample = min(len(self.memory), self.hyper_params.batch_size)
        if self.use_prioritized:
            experiences_one_step = self.memory.sample(self.per_beta)
            weights, indices, eps_demo = experiences_one_step[-3:]
            indices = np.array(indices)
            # re-normalize the weights such that they sum up to the value of batch_size
            weights = weights / torch.sum(weights) * float(n_sample)
        else:
            indices = np.random.choice(len(self.memory), size=n_sample, replace=False)
            eps_demo = np.zeros_like(indices)
            eps_demo[np.where(indices < self.memory.demo_size)] = self.hyper_params.per_eps_demo
            weights = torch.from_numpy(np.ones(shape=(indices.shape[0], 1), dtype=np.float64)).type(
                torch.FloatTensor).to(device)
            experiences_one_step = self.memory.sample(indices=indices)

        obs, actions, rewards, next_obs, dones = experiences_one_step[:5]
        actions = actions.unsqueeze(-1)

        _, _, log_probs, _ = self.actor(obs)
        curr_action_probs = log_probs.exp()
        curr_policy_actions = torch.argmax(curr_action_probs, dim=-1, keepdim=True)

        ###################
        ### Train Alpha ###
        ###################
        if self.hyper_params.auto_entropy_tuning:
            # alpha loss (equation 11 in the paper)
            log_new_action_probs = torch.sum(curr_action_probs * log_probs, dim=-1)
            alpha_loss = (
                    (-self.log_alpha * ((log_new_action_probs + self.target_entropy).detach())) * weights
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
                                                obs=obs, actions=actions, policy_actions=curr_policy_actions,
                                                log_probs=log_probs, action_probs=curr_action_probs, alpha=alpha)
        actor_loss, actor_supervised_loss = actor_loss_info

        ###############################
        ### Q functions 1-Step Loss ###
        ###############################
        loss_info = self._get_qf_loss(obs=obs, actions=actions, rewards=rewards, next_obs=next_obs, dones=dones,
                                      alpha=alpha, gamma=self.hyper_params.gamma,
                                      reward_clip=self.hyper_params.reward_clip)
        q1_loss_element_wise, q2_loss_element_wise, q1_values, q2_values = loss_info
        q1_loss = torch.mean(q1_loss_element_wise * weights)
        q2_loss = torch.mean(q2_loss_element_wise * weights)

        ###############################
        ### Q functions n-Step Loss ###
        ###############################
        if self.use_n_step:
            experiences_n = self.memory_n.sample(indices)
            obs_n, actions_n, rewards_n, next_obs_n, dones_n = experiences_n
            actions_n = actions_n.unsqueeze(-1)
            gamma_n = self.hyper_params.gamma ** self.hyper_params.n_step

            loss_info_n = self._get_qf_loss(obs=obs_n, actions=actions_n, rewards=rewards_n, next_obs=next_obs_n,
                                            dones=dones_n, alpha=alpha, gamma=gamma_n,
                                            reward_clip=self.hyper_params.reward_clip)
            q1_loss_element_wise_n, q2_loss_element_wise_n, q1_values_n, q2_values_n = loss_info_n
            q1_values = 0.5 * (q1_values + q1_values_n)
            q2_values = 0.5 * (q2_values + q2_values_n)

            # mix of 1-step and n-step returns
            q1_loss_element_wise += q1_loss_element_wise_n * self.hyper_params.w_n_step
            q2_loss_element_wise += q2_loss_element_wise_n * self.hyper_params.w_n_step
            q1_loss = torch.mean(q1_loss_element_wise * weights)
            q2_loss = torch.mean(q2_loss_element_wise * weights)

        ###################################
        ### Q functions Supervised Loss ###
        ###################################
        q1_supervised_loss, q2_supervised_loss, n_demo = self._get_qf_supervised_loss(sample_indices=indices, actions=actions,
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

    def write_log(self, log_value):
        """
        Write log about loss and score
        log_value should be in the form of i_episode, avg_loss_info, score, avg_time_cost, avg_scores_window
        """
        i_episode, avg_loss_info, score, avg_time_cost, avg_scores_window, alpha_value = log_value

        print(
            "[INFO] episode %d, episode_step %d, total step %d, total score: %d\n"
            "actor_loss: %.3f, actor_supervised_loss: %.3f, qf1_loss: %.3f, qf2_loss: %.3f\n"
            "alpha_loss: %.3f, alpha: %.3f, avg window score: %.3f\n"
            "avg qf 1: %.3f, avg qf 2: %.3f, n demo: %d (spent %.6f sec/step)\n"
            % (
                i_episode,
                self.episode_step,
                self.total_step,
                score,
                avg_loss_info[0],  # actor loss
                avg_loss_info[1],  # actor bc loss
                avg_loss_info[2],  # qf1 loss
                avg_loss_info[3],  # qf2 loss
                avg_loss_info[4],  # alpha loss
                alpha_value,
                avg_scores_window,
                avg_loss_info[5],   # avg qf 1
                avg_loss_info[6],   # avg qf 2
                avg_loss_info[7],   # n demo
                avg_time_cost,
            )
        )

        if self.logger is not None:
            self.logger.log_wandb({
                'score': score,
                "episode": self.i_episode,
                "episode step": self.episode_step,
                "total step": self.total_step,
                "alpha": alpha_value,
                'actor loss': avg_loss_info[0],
                'actor supervised loss': avg_loss_info[1],
                'qf1 loss': avg_loss_info[2],
                'qf2 loss': avg_loss_info[3],
                'alpha loss': avg_loss_info[4],
                'time per each step': avg_time_cost,
                "avg score window": avg_scores_window,
                'demo num': avg_loss_info[7],
                'avg q1 values': avg_loss_info[5],
                'avg q2 values': avg_loss_info[6],
                'avg q values': 0.5 * (avg_loss_info[5] + avg_loss_info[6])
            }, step=self.total_step)

    def do_post_episode_update(self, *argv):
        if self.total_step >= self.hyper_params.init_random_actions:
            self.lambda2 = max(self.hyper_params.min_lambda2, self.hyper_params.lambda2_decay * self.lambda2)
            self.policy_bc_weight = max(self.hyper_params.min_policy_bc_weight,
                                        self.hyper_params.policy_bc_weight_decay * self.policy_bc_weight)

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
                print("[INFO] Pre-Train step %d, actor loss: %.3f, actor supervised loss: %.3f\n"
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

        print("[INFO] Pre-Train Complete!\n")



