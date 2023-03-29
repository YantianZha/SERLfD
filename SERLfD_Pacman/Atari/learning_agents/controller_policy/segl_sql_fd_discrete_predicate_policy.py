import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical

from learning_agents.architectures.cnn import Conv2d_Flatten_MLP
from learning_agents.controller_policy.segl_sql_fd_discrete_policy import SEGL_SQLfD_Discrete_Policy
from utils.trajectory_utils import TRAJECTORY_INDEX

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LOG_REG = 1e-8


class SEGL_SQLfD_Discrete_Predicate_Policy(SEGL_SQLfD_Discrete_Policy):
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
        SEGL_SQLfD_Discrete_Policy.__init__(self, env, args, log_cfg, hyper_params, network_cfg, optim_cfg, demo_preprocessor, encoder=encoder, logger=logger)

    # pylint: disable=attribute-defined-outside-init
    def _init_network(self):
        """Initialize networks and optimizers."""
        self.dqn = Conv2d_Flatten_MLP(input_channels=self.state_channel,
                                      fc_input_size=self.network_cfg.fc_input_size,
                                      fc_output_size=self.action_dim,
                                      nonlinearity=self.network_cfg.nonlinearity,
                                      channels=self.network_cfg.channels,
                                      kernel_sizes=self.network_cfg.kernel_sizes,
                                      strides=self.network_cfg.strides,
                                      paddings=self.network_cfg.paddings,
                                      fc_hidden_sizes=self.network_cfg.fc_hidden_sizes,
                                      fc_hidden_activation=self.network_cfg.fc_hidden_activation,
                                      ).to(device)
        self.dqn_target = Conv2d_Flatten_MLP(input_channels=self.state_channel,
                                             fc_input_size=self.network_cfg.fc_input_size,
                                             fc_output_size=self.action_dim,
                                             nonlinearity=self.network_cfg.nonlinearity,
                                             channels=self.network_cfg.channels,
                                             kernel_sizes=self.network_cfg.kernel_sizes,
                                             strides=self.network_cfg.strides,
                                             paddings=self.network_cfg.paddings,
                                             fc_hidden_sizes=self.network_cfg.fc_hidden_sizes,
                                             fc_hidden_activation=self.network_cfg.fc_hidden_activation,
                                             ).to(device)
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

    def _preprocess_predicates_states(self, states, predicates):
        """Preprocess state so that actor selects an action."""
        if not isinstance(states, torch.Tensor):
            states = torch.FloatTensor(states).to(device)
        # if state is a single state, we unsqueeze it
        if len(states.size()) == 3:
            states = states.unsqueeze(0)
        n_state = states.size(0)
        if self.encoder is not None:
            states = self.encoder(states)

        # process predicate vector
        if not isinstance(predicates, torch.Tensor):
            predicates = torch.FloatTensor(predicates).to(device)
        if len(predicates.size()) == 1:
            predicates = predicates.unsqueeze(0)

        if self.hyper_params.use_flatten_input:
            # concatenate states and predicates
            flatten_states = states.view(n_state, -1)
            states_predicates = torch.cat((flatten_states, predicates), dim=1)
        else:
            states_predicates = (states, predicates)

        return states_predicates

    def get_action(self, state, is_test=False, is_random=False, info=None):
        predicate_vector = info[TRAJECTORY_INDEX.PREDICATE_VECTOR.value]
        predicate_utility = info[TRAJECTORY_INDEX.UTILITY_VECTOR.value]
        predicate_util_vector = np.concatenate((predicate_vector, predicate_utility), axis=1)

        if (not is_test and self.epsilon > np.random.random()) or is_random:
            selected_action = np.array(self.env.action_space.sample())
        else:
            state_predicate = self._preprocess_predicates_states(state, predicate_util_vector)
            self.dqn.eval()
            with torch.no_grad():
                if is_test or self.args.deterministic:
                    selected_action = self.dqn(*state_predicate).argmax()
                else:
                    q_values = ((1 / self.alpha) * self.dqn(*state_predicate))
                    action_probs = F.log_softmax(q_values, dim=-1).exp()
                    dist = Categorical(action_probs)
                    selected_action = dist.sample()
            self.dqn.train()
            selected_action = selected_action.detach().cpu().numpy()
        return selected_action

    def augment_experience(self, explainer, experiences, predicates, next_predicates, is_to_tensor=True):
        states, _, rewards, next_states, _ = experiences[:5]
        assert len(states) == len(predicates), 'the number of states and predicates should be the same'

        augmented_states = explainer.augment_states(states, predicates)
        predicate_vectors = explainer.to_predicate_vectors(predicates)
        predicate_util_vectors = explainer.get_utility_values_vectors(states, predicates)
        states_predicates_util = self._preprocess_predicates_states(augmented_states,
                                                                    np.concatenate((predicate_vectors,
                                                                                   predicate_util_vectors), axis=1))

        augmented_next_states = explainer.augment_states(next_states, next_predicates)
        next_predicate_vectors = explainer.to_predicate_vectors(next_predicates)
        next_predicate_util_vectors = explainer.get_utility_values_vectors(next_states, next_predicates)
        next_states_predicates_util = self._preprocess_predicates_states(augmented_next_states,
                                                                         np.concatenate((next_predicate_vectors,
                                                                                         next_predicate_util_vectors), axis=1))

        shaping_rewards = explainer.get_shaping_reward(states, predicates, next_states, next_predicates)
        augmented_rewards = shaping_rewards + rewards

        if is_to_tensor:
            augmented_rewards = self._to_float_tensor(augmented_rewards)
        else:
            states_predicates_util = states_predicates_util.detach().cpu().numpy()
            next_states_predicates_util = next_states_predicates_util.detach().cpu().numpy()

        return states_predicates_util, augmented_rewards, next_states_predicates_util

    def evaluate_states_actions(self, states, actions, info=None):
        return self.evaluate_state_action(states, actions, info=info).detach().cpu().numpy()

    def evaluate_state_action(self, state, action, info=None):
        """
        Compute the probabilities/densities of the state-action pair.
        state: np.ndarray
        action: np.ndarray
        """
        torch_actions = torch.FloatTensor(action).to(device).unsqueeze(-1)

        predicate_vector = info[TRAJECTORY_INDEX.PREDICATE_VECTOR.value]
        predicate_utility = info[TRAJECTORY_INDEX.UTILITY_VECTOR.value]
        predicate_util_vector = np.concatenate((predicate_vector, predicate_utility), axis=1)

        torch_states_predicates = self._preprocess_predicates_states(state, predicate_util_vector)

        self.dqn.eval()
        with torch.no_grad():
            q_values = ((1 / self.alpha) * self.dqn(*torch_states_predicates))
            action_probs = F.log_softmax(q_values, dim=-1).exp()
            action_probs = action_probs.gather(1, torch_actions.long())
            action_probs = torch.clamp(action_probs, min=LOG_REG, max=1.0)
            log_probs = action_probs.log()
        self.dqn.train()

        return log_probs


class SEGL_SQLfD_No_Shaping_Policy(SEGL_SQLfD_Discrete_Predicate_Policy):
    def __init__(self, env, args, log_cfg, hyper_params, network_cfg, optim_cfg, demo_preprocessor, encoder=None,
                 logger=None):
        super().__init__(env, args, log_cfg, hyper_params, network_cfg, optim_cfg, demo_preprocessor,
                         encoder=encoder, logger=logger)

    def augment_experience(self, explainer, experiences, predicates, next_predicates, is_to_tensor=True):
        states, _, rewards, next_states, _ = experiences[:5]
        assert len(states) == len(predicates), 'the number of states and predicates should be the same'

        augmented_states = explainer.augment_states(states, predicates)
        predicate_vectors = explainer.to_predicate_vectors(predicates)
        predicate_util_vectors = explainer.get_utility_values_vectors(states, predicates)
        states_predicates_util = self._preprocess_predicates_states(augmented_states,
                                                                    np.concatenate((predicate_vectors,
                                                                                   predicate_util_vectors), axis=1))

        augmented_next_states = explainer.augment_states(next_states, next_predicates)
        next_predicate_vectors = explainer.to_predicate_vectors(next_predicates)
        next_predicate_util_vectors = explainer.get_utility_values_vectors(next_states, next_predicates)
        next_states_predicates_util = self._preprocess_predicates_states(augmented_next_states,
                                                                         np.concatenate((next_predicate_vectors,
                                                                                         next_predicate_util_vectors), axis=1))

        # shaping_rewards = explainer.get_shaping_reward(states, predicates, next_states, next_predicates)
        augmented_rewards = rewards

        if is_to_tensor:
            augmented_rewards = self._to_float_tensor(augmented_rewards)
        else:
            states_predicates_util = states_predicates_util.detach().cpu().numpy()
            next_states_predicates_util = next_states_predicates_util.detach().cpu().numpy()

        return states_predicates_util, augmented_rewards, next_states_predicates_util



