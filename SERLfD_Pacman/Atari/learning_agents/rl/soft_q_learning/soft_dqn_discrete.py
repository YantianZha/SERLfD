import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical

from learning_agents.rl.dqn import dqn_utils
from learning_agents.rl.dqn.dqn import DQNAgent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SQL_Discrete_Agent(DQNAgent):
    """
    Soft Deep Q-Learning Agent. (Here we assume that we are using stacked frames and grayscaled observation)
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
        DQNAgent.__init__(self, env, args, log_cfg, hyper_params, network_cfg, optim_cfg,
                          encoder=encoder, logger=logger)
        self.alpha = self.hyper_params.max_alpha

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
                if self.testing or self.args.deterministic:
                    selected_action = self.dqn(state).argmax()
                else:
                    q_values = ((1 / self.alpha) * self.dqn(state))
                    action_probs = F.log_softmax(q_values, dim=-1).exp()
                    dist = Categorical(action_probs)
                    selected_action = dist.sample()
            self.dqn.train()
            selected_action = selected_action.detach().cpu().numpy()
        return selected_action

    def _get_dqn_loss(self, experiences, gamma):
        """Return element-wise dqn loss and Q-values."""
        return dqn_utils.calculate_soft_dqn_loss(
            model=self.dqn,
            target_model=self.dqn_target,
            experiences=experiences,
            gamma=gamma,
            alpha=self.alpha,
            use_double_q_update=self.hyper_params.use_double_q_update,
            reward_clip=self.hyper_params.reward_clip
        )

    def write_log(self, log_value):
        """Write log about loss and score"""
        i, loss, score, avg_time_cost, avg_score_window = log_value
        print(
            "[INFO] episode %d, episode step: %d, total step: %d, total score: %f\n"
            "epsilon: %f, alpha: %f, loss: %f, avg q-value: %f , avg score window: %f (spent %.6f sec/step)\n"
            % (
                i,
                self.episode_step,
                self.total_step,
                score,
                self.epsilon,
                self.alpha,
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
                "alpha": self.alpha,
                "dqn loss": loss[0],
                "avg q values": loss[1],
                "time per each step": avg_time_cost,
                "avg score window": avg_score_window,
            }, step=self.total_step)

    def do_post_episode_update(self, *argv):
        if self.total_step >= self.hyper_params.init_random_actions:
            # decrease epsilon
            self.epsilon = max(self.min_epsilon, self.hyper_params.epsilon_decay * self.epsilon)
            # decrease alpha
            self.alpha = max(self.hyper_params.min_alpha, self.hyper_params.alpha_decay * self.alpha)
