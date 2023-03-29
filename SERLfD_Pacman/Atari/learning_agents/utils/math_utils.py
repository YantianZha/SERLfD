import numpy as np
import time

import torch
from torch.distributions import Independent, Normal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TanhNormal:
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)
    Note: this is not very numerically stable.
    Source: https://github.com/vitchyr/rlkit/blob/f136e140a57078c4f0f665051df74dffb1351f33/rlkit/torch/distributions.py
    """
    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def sample_n(self, n):
        z = self.normal.sample_n(n)
        return torch.tanh(z), z

    def log_prob(self, value, pre_tanh_value=None):
        """
        :param value (torch.Tensor): some value, x
        :param pre_tanh_value (torch.Tensor): arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            # arctanh(x) = 1/2*log((1+x)/(1-x))
            pre_tanh_value = torch.log(
                (1+value) / (1-value)
            ) / 2.0

        if value is None:
            value = torch.tanh(pre_tanh_value)

        action = value
        z = pre_tanh_value
        log_prob = self.normal.log_prob(z) - torch.log(1 - action.pow(2) + self.epsilon)
        return log_prob.sum(-1, keepdim=True)

    def sample(self):
        """
        Gradients will and should not pass through this operation.
        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()
        return torch.tanh(z), z

    def rsample(self):
        """
        Sampling in the reparameterization case.
        """
        z = self.normal.rsample()
        z.requires_grad_()
        return torch.tanh(z), z


def normal_log_density(means, stds, actions):
    dist = Independent(Normal(means, stds), 1)
    return dist.log_prob(actions)


# noinspection PyPep8Naming
def gaussian_log_pdf(params, x):
    mean, log_diag_std = params
    N, d = mean.shape
    cov = np.square(np.exp(log_diag_std))
    diff = x-mean
    exp_term = -0.5 * np.sum(np.square(diff)/cov, axis=1)
    norm_term = -0.5*d*np.log(2*np.pi)
    var_term = -0.5 * np.sum(np.log(cov), axis=1)
    log_probs = norm_term + var_term + exp_term
    return log_probs


def categorical_log_pdf(params, x, one_hot=True):
    if not one_hot:
        raise NotImplementedError()
    probs = params[0]
    return np.log(np.max(probs * x, axis=1))


