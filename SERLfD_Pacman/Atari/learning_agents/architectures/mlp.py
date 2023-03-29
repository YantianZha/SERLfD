import torch
from torch.distributions import Categorical, Normal
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from learning_agents.common.common_utils import identity
from learning_agents.utils.math_utils import TanhNormal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def init_layer_uniform(layer, init_w=3e-3, init_b=0.1):
    """
    Init uniform parameters on the single layer
    layer: nn.Linear
    init_w: float = 3e-3
    :return: nn.Linear
    """
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_b, init_b)

    return layer


class MLP(nn.Module):
    """
    Baseline of Multilayer perceptron. The layer-norm is not implemented here
    Attributes:
        input_size (int): size of input
        output_size (int): size of output layer
        hidden_sizes (list): sizes of hidden layers
        hidden_activation (function): activation function of hidden layers
        output_activation (function): activation function of output layer
        hidden_layers (list): list containing linear layers
        use_output_layer (bool): whether or not to use the last layer
        n_category (int): category number (-1 if the action is continuous)
    """

    def __init__(
        self,
        input_size,
        output_size,
        hidden_sizes,
        hidden_activation=F.relu,
        output_activation=identity,
        linear_layer=nn.Linear,
        use_output_layer=True,
        n_category=-1,
        init_fn=init_layer_uniform
    ):
        """
        Initialize.
        Args:
            input_size (int): size of input
            output_size (int): size of output layer
            hidden_sizes (list): number of hidden layers
            hidden_activation (function): activation function of hidden layers
            output_activation (function): activation function of output layer
            linear_layer (nn.Module): linear layer of mlp
            use_output_layer (bool): whether or not to use the last layer
            n_category (int): category number (-1 if the action is continuous)
            init_fn (Callable): weight initialization function bound for the last layer
        """
        super(MLP, self).__init__()

        self.hidden_sizes = hidden_sizes
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.linear_layer = linear_layer
        self.use_output_layer = use_output_layer
        self.n_category = n_category

        # set hidden layers
        self.hidden_layers = []
        in_size = self.input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = self.linear_layer(in_size, next_size)
            in_size = next_size
            self.__setattr__("hidden_fc{}".format(i), fc)
            self.hidden_layers.append(fc)

        # set output layers
        if self.use_output_layer:
            self.output_layer = self.linear_layer(in_size, output_size)
            self.output_layer = init_fn(self.output_layer)
        else:
            self.output_layer = identity
            self.output_activation = identity

    def forward(self, x):
        """
        Forward method implementation.
        x: torch.Tensor
        :return: torch.Tensor
        """
        for hidden_layer in self.hidden_layers:
            x = self.hidden_activation(hidden_layer(x))
        x = self.output_activation(self.output_layer(x))

        return x


class FlattenMLP(MLP):
    """
    Baseline of Multi-layered perceptron for Flatten input.
    """

    def forward(self, *args):
        """
        Forward method implementation.
        states is assume to be Tensor containing list of flatten states
        """
        states, actions = args

        if len(states.size()) == 1:
            states = states.unsqueeze(0)
        if len(actions.size()) == 1:
            actions = actions.unsqueeze(0)
        flat_inputs = torch.cat((states, actions), dim=-1)
        return super(FlattenMLP, self).forward(flat_inputs)


class GaussianDist(MLP):
    """
    Multilayer perceptron with Gaussian distribution output.
    the Mean

    Attributes:
        mean_activation (function): bounding function for mean
        log_std_min (float): lower bound of log std
        log_std_max (float): upper bound of log std
        mean_layer (nn.Linear): output layer for mean
        log_std_layer (nn.Linear): output layer for log std
    """

    def __init__(
        self,
        input_size,
        output_size,
        hidden_sizes,
        hidden_activation=F.relu,
        mean_activation=torch.tanh,
        std_activation=torch.tanh,
        log_std_min=-20,
        log_std_max=2,
        init_fn=init_layer_uniform,
        std=None
    ):
        """
        Initialize
        If std is not None, then use fixed std value given by argument std, otherwise use std layer
        """
        super(GaussianDist, self).__init__(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            hidden_activation=hidden_activation,
            use_output_layer=False,
        )
        self.std_activation = std_activation
        self.mean_activation = mean_activation
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        in_size = hidden_sizes[-1]  # std layer is the last layer

        # set log_std layer
        self.std = std
        self.log_std = None
        if std is None:
            self.log_std_layer = nn.Linear(in_size, output_size)
            self.log_std_layer = init_fn(self.log_std_layer)
        else:
            self.log_std = np.log(std)
            assert log_std_min <= self.log_std <= log_std_max

        # set mean layer
        self.mean_layer = nn.Linear(in_size, output_size)
        self.mean_layer = init_fn(self.mean_layer)

    def get_dist_params(self, x):
        """
        Return gaussian distribution parameters.
        x (torch.Tensor)
        :return: mu, log_std, std
        """
        hidden = super(GaussianDist, self).forward(x)

        # get mean
        mu = self.mean_activation(self.mean_layer(hidden))

        # get std
        if self.std is None:
            uncentered_log_std = self.std_activation(self.log_std_layer(hidden))
            log_std = uncentered_log_std.clamp(min=self.log_std_min, max=self.log_std_max)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        return mu, log_std, std

    def forward(self, x):
        """
        Forward method implementation.
        x (torch.Tensor)
        :return: action (torch.Tensor) and dist
        """
        mu, _, std = self.get_dist_params(x)

        # get normal distribution and action
        dist = Normal(mu, std)
        action = dist.sample()

        return action, dist


class TanhGaussianDistParams(GaussianDist):
    """
    Multilayer perceptron with Gaussian distribution output.
    """

    def __init__(self, **kwargs):
        """Initialize."""
        super(TanhGaussianDistParams, self).__init__(**kwargs, mean_activation=identity)

    def forward(self, x, epsilon=1e-6, deterministic=False, reparameterize=True):
        """
        Forward method implementation.
        x: torch.Tensor
        epsilon: float = 1e-6
        deterministic: bool = False, if deterministic is True, action = tanh(mean).
        :return: Tuple[torch.Tensor, ...]
        """
        mean, _, std = super(TanhGaussianDistParams, self).get_dist_params(x)

        # sampling actions
        if deterministic:
            action = torch.tanh(mean)
            return action, None, None, mean, std
        else:
            tanh_normal = TanhNormal(mean, std, epsilon=epsilon)
            if reparameterize:
                action, z = tanh_normal.rsample()
                log_prob = tanh_normal.log_prob(value=action, pre_tanh_value=z)
            else:
                action, z = tanh_normal.sample()
                log_prob = tanh_normal.log_prob(value=action, pre_tanh_value=z)

            return action, log_prob, z, mean, std


class CategoricalDist(MLP):
    """
    Multilayer perceptron with categorical distribution output (for discrete domains)
    Attributes:
        last_layer (nn.Linear): output layer for softmax
    """

    def __init__(
        self,
        input_size,
        output_size,
        hidden_sizes,
        hidden_activation=F.relu,
        init_fn=init_layer_uniform,
    ):
        """Initialize."""
        super(CategoricalDist, self).__init__(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            hidden_activation=hidden_activation,
            use_output_layer=False,
        )

        in_size = hidden_sizes[-1]

        self.last_layer = nn.Linear(in_size, output_size)
        self.last_layer = init_fn(self.last_layer)

    def forward(self, x):
        """ Forward method implementation."""
        hidden = super(CategoricalDist, self).forward(x)
        action_probs = F.softmax(self.last_layer(hidden), dim=-1)

        dist = Categorical(action_probs)
        selected_action = dist.sample()
        selected_action = selected_action.unsqueeze(-1)

        return selected_action, action_probs, dist


class CategoricalDistParams(CategoricalDist):
    """ Multilayer perceptron with Categorical distribution output."""

    def __init__(self, epsilon=1e-8, **kwargs):
        """Initialize."""
        super(CategoricalDistParams, self).__init__(**kwargs)
        self.epsilon = epsilon

    def forward(self, x, deterministic=False):
        """ Forward method implementation."""
        selected_action, action_probs, dist = super(CategoricalDistParams, self).forward(x)

        if deterministic:
            selected_action = torch.argmax(action_probs, dim=-1, keepdim=True)

        z = (action_probs == 0.0)
        z = z.float() * self.epsilon
        log_probs = torch.log(action_probs + z)

        return selected_action, action_probs, log_probs, dist
