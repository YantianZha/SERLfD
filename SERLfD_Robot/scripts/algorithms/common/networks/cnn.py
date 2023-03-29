import numpy as np
import torch
import torch.nn as nn

from learning_agents.common.common_utils import identity
from learning_agents.architectures.mlp import MLP, GaussianDist, CategoricalDistParams, TanhGaussianDistParams

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CNNLayer(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        stride=1,
        padding=0,
        pre_activation_fn=identity,
        activation_fn=torch.relu,
        post_activation_fn=identity,
    ):
        super(CNNLayer, self).__init__()
        self.cnn = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.pre_activation_fn = pre_activation_fn
        self.activation_fn = activation_fn
        self.post_activation_fn = post_activation_fn

    def forward(self, x):
        x = self.cnn(x)
        x = self.pre_activation_fn(x)
        x = self.activation_fn(x)
        x = self.post_activation_fn(x)

        return x


class CNN(nn.Module):
    """ Baseline of Convolution neural network. """
    def __init__(self, cnn_layers, fc_layers):
        """
        cnn_layers: List[CNNLayer]
        fc_layers: MLP
        """
        super(CNN, self).__init__()

        self.cnn_layers = cnn_layers
        self.fc_layers = fc_layers

        self.cnn = nn.Sequential()
        for i, cnn_layer in enumerate(self.cnn_layers):
            self.cnn.add_module("cnn_{}".format(i), cnn_layer)

    def get_cnn_features(self, x):
        """
        Get the output of CNN.
        """
        if len(x.size()) == 3:
            x = x.unsqueeze(0)
        x = self.cnn(x)
        # flatten x
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x, **fc_kwargs):
        """
        Forward method implementation.
        x: torch.Tensor
        :return: torch.Tensor
        """
        x = self.get_cnn_features(x)
        x = self.fc_layers(x, **fc_kwargs)
        return x


class Conv2d_MLP_Model(nn.Module):
    """ Default convolution neural network composed of conv2d layer followed by fully-connected MLP models """

    # noinspection PyDefaultArgument
    def __init__(self,
                 # conv2d layer arguments
                 input_channels,
                 # fc layer arguments
                 fc_input_size,
                 fc_output_size,
                 # conv2d optional arguments
                 channels=[32, 32, 32],
                 kernel_sizes=[8, 4, 3],
                 strides=[4, 2, 1],
                 paddings=[0, 1, 1],
                 nonlinearity=torch.relu,
                 use_maxpool=False,
                 # fc layer optional arguments
                 fc_hidden_sizes=[100, 100],
                 fc_hidden_activation=torch.relu,
                 fc_output_activation=identity
                 ):
        super(Conv2d_MLP_Model, self).__init__()
        if paddings is None:
            paddings = [0 for _ in range(len(channels))]
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        in_channels = [input_channels] + channels[:-1]

        post_activation_fns = [identity for _ in range(len(strides))]
        ones = [1 for _ in range(len(strides))]
        if use_maxpool:
            post_activation_fns = [torch.nn.MaxPool2d(max_pool_stride) for max_pool_stride in strides]
            strides = ones
        activation_fns = [nonlinearity for _ in range(len(strides))]

        conv_layers = [CNNLayer(input_channels=ic, output_channels=oc,
                                kernel_size=k, stride=s, padding=p, activation_fn=a_fn, post_activation_fn=p_fn)
                       for (ic, oc, k, s, p, a_fn, p_fn) in zip(in_channels, channels, kernel_sizes, strides, paddings,
                                                                activation_fns, post_activation_fns)]
        fc_layers = MLP(
            input_size=fc_input_size,
            output_size=fc_output_size,
            hidden_sizes=fc_hidden_sizes,
            hidden_activation=fc_hidden_activation,
            output_activation=fc_output_activation
        )

        self.conv_mlp = CNN(cnn_layers=conv_layers, fc_layers=fc_layers)

    def forward(self, x):
        return self.conv_mlp.forward(x)


class Conv2d_MLP_Gaussian(nn.Module):
    """ Default convolution neural network composed of conv2d layer followed by fully-connected MLP models """

    # noinspection PyDefaultArgument
    def __init__(self,
                 # conv2d layer arguments
                 input_channels,
                 # fc layer arguments
                 fc_input_size,
                 fc_output_size,
                 # conv2d optional arguments
                 channels=[32, 32, 32],
                 kernel_sizes=[8, 4, 3],
                 strides=[4, 2, 1],
                 paddings=[0, 1, 1],
                 nonlinearity=torch.relu,
                 use_maxpool=False,
                 # fc layer optional arguments
                 fc_hidden_sizes=[100, 100],
                 fc_hidden_activation=torch.relu
                 ):
        super(Conv2d_MLP_Gaussian, self).__init__()
        if paddings is None:
            paddings = [0 for _ in range(len(channels))]
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        in_channels = [input_channels] + channels[:-1]

        post_activation_fns = [identity for _ in range(len(strides))]
        ones = [1 for _ in range(len(strides))]
        if use_maxpool:
            post_activation_fns = [torch.nn.MaxPool2d(max_pool_stride) for max_pool_stride in strides]
            strides = ones
        activation_fns = [nonlinearity for _ in range(len(strides))]

        conv_layers = [CNNLayer(input_channels=ic, output_channels=oc,
                                kernel_size=k, stride=s, padding=p, activation_fn=a_fn, post_activation_fn=p_fn)
                       for (ic, oc, k, s, p, a_fn, p_fn) in zip(in_channels, channels, kernel_sizes, strides, paddings,
                                                                activation_fns, post_activation_fns)]
        fc_layers = GaussianDist(
            input_size=fc_input_size,
            output_size=fc_output_size,
            hidden_sizes=fc_hidden_sizes,
            hidden_activation=fc_hidden_activation
        )

        self.conv_mlp = CNN(cnn_layers=conv_layers, fc_layers=fc_layers)

    def forward(self, x):
        return self.conv_mlp.forward(x)


class Conv2d_MLP_Categorical(nn.Module):
    """ Default convolution neural network composed of conv2d layer followed by fully-connected MLP models """

    # noinspection PyDefaultArgument
    def __init__(self,
                 # conv2d layer arguments
                 input_channels,
                 # fc layer arguments
                 fc_input_size,
                 fc_output_size,
                 # conv2d optional arguments
                 channels=[32, 32, 32],
                 kernel_sizes=[8, 4, 3],
                 strides=[4, 2, 1],
                 paddings=[0, 1, 1],
                 nonlinearity=torch.relu,
                 use_maxpool=False,
                 # fc layer optional arguments
                 fc_hidden_sizes=[100, 100],
                 fc_hidden_activation=torch.relu
                 ):
        super(Conv2d_MLP_Categorical, self).__init__()
        if paddings is None:
            paddings = [0 for _ in range(len(channels))]
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        in_channels = [input_channels] + channels[:-1]

        post_activation_fns = [identity for _ in range(len(strides))]
        ones = [1 for _ in range(len(strides))]
        if use_maxpool:
            post_activation_fns = [torch.nn.MaxPool2d(max_pool_stride) for max_pool_stride in strides]
            strides = ones
        activation_fns = [nonlinearity for _ in range(len(strides))]

        conv_layers = [CNNLayer(input_channels=ic, output_channels=oc,
                                kernel_size=k, stride=s, padding=p, activation_fn=a_fn, post_activation_fn=p_fn)
                       for (ic, oc, k, s, p, a_fn, p_fn) in zip(in_channels, channels, kernel_sizes, strides, paddings,
                                                                activation_fns, post_activation_fns)]
        fc_layers = CategoricalDistParams(
            input_size=fc_input_size,
            output_size=fc_output_size,
            hidden_sizes=fc_hidden_sizes,
            hidden_activation=fc_hidden_activation
        )

        self.conv_categorical_mlp = CNN(cnn_layers=conv_layers, fc_layers=fc_layers)

    def forward(self, x, deterministic=False):
        return self.conv_categorical_mlp.forward(x, deterministic=deterministic)


class Conv2d_MLP_TanhGaussian(nn.Module):
    """ Default convolution neural network composed of conv2d layer followed by fully-connected MLP models """

    # noinspection PyDefaultArgument
    def __init__(self,
                 # conv2d layer arguments
                 input_channels,
                 # fc layer arguments
                 fc_input_size,
                 fc_output_size,
                 # conv2d optional arguments
                 channels=[32, 32, 32],
                 kernel_sizes=[8, 4, 3],
                 strides=[4, 2, 1],
                 paddings=[0, 1, 1],
                 nonlinearity=torch.relu,
                 use_maxpool=False,
                 # fc layer optional arguments
                 fc_hidden_sizes=[100, 100],
                 fc_hidden_activation=torch.relu
                 ):
        super(Conv2d_MLP_TanhGaussian, self).__init__()
        if paddings is None:
            paddings = [0 for _ in range(len(channels))]
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        in_channels = [input_channels] + channels[:-1]

        post_activation_fns = [identity for _ in range(len(strides))]
        ones = [1 for _ in range(len(strides))]
        if use_maxpool:
            post_activation_fns = [torch.nn.MaxPool2d(max_pool_stride) for max_pool_stride in strides]
            strides = ones
        activation_fns = [nonlinearity for _ in range(len(strides))]

        conv_layers = [CNNLayer(input_channels=ic, output_channels=oc,
                                kernel_size=k, stride=s, padding=p, activation_fn=a_fn, post_activation_fn=p_fn)
                       for (ic, oc, k, s, p, a_fn, p_fn) in zip(in_channels, channels, kernel_sizes, strides, paddings,
                                                                activation_fns, post_activation_fns)]
        fc_layers = TanhGaussianDistParams(
            input_size=fc_input_size,
            output_size=fc_output_size,
            hidden_sizes=fc_hidden_sizes,
            hidden_activation=fc_hidden_activation
        )

        self.conv_tanh_gaussian_mlp = CNN(cnn_layers=conv_layers, fc_layers=fc_layers)

    def forward(self, x, epsilon=1e-6, deterministic=False, reparameterize=True):
        return self.conv_tanh_gaussian_mlp.forward(x, epsilon=1e-6, deterministic=False, reparameterize=True)


class Conv2d_Flatten_MLP(Conv2d_MLP_Model):
    """
    Augmented convolution neural network, in which a feature vector will be appended to
        the features extracted by CNN before entering mlp
    """
    # noinspection PyDefaultArgument
    def __init__(self,
                 # conv2d layer arguments
                 input_channels,
                 # fc layer arguments
                 fc_input_size,
                 fc_output_size,
                 # conv2d optional arguments
                 channels=[32, 32, 32],
                 kernel_sizes=[8, 4, 3],
                 strides=[4, 2, 1],
                 paddings=[0, 1, 1],
                 nonlinearity=torch.relu,
                 use_maxpool=False,
                 # fc layer optional arguments
                 fc_hidden_sizes=[100, 100],
                 fc_hidden_activation=torch.relu,
                 fc_output_activation=identity
                 ):
        super(Conv2d_Flatten_MLP, self).__init__(input_channels=input_channels,
                                                 fc_input_size=fc_input_size,
                                                 fc_output_size=fc_output_size,
                                                 channels=channels, kernel_sizes=kernel_sizes, strides=strides,
                                                 paddings=paddings, nonlinearity=nonlinearity,
                                                 use_maxpool=use_maxpool, fc_hidden_sizes=fc_hidden_sizes,
                                                 fc_hidden_activation=fc_hidden_activation,
                                                 fc_output_activation=fc_output_activation)

    def forward(self, *args):
        obs_x, augment_features = args
        cnn_features = self.conv_mlp.get_cnn_features(obs_x)
        features = torch.cat((cnn_features, augment_features), dim=1)
        return self.conv_mlp.fc_layers(features)


class Conv2d_MLP_Model_v1(nn.Module):
    """ Default convolution neural network composed of conv2d layer followed by fully-connected MLP models """

    # noinspection PyDefaultArgument
    def __init__(self,
                 # conv2d layer arguments
                 observation_space,
                 # fc layer arguments
                 fc_output_size,
                 # conv2d optional arguments
                 channels=[32, 32, 32],
                 kernel_sizes=[8, 4, 3],
                 strides=[4, 2, 1],
                 paddings=[0, 1, 1],
                 nonlinearity=torch.relu,
                 use_maxpool=False,
                 # fc layer optional arguments
                 fc_hidden_sizes=[100, 100],
                 fc_hidden_activation=torch.relu,
                 fc_output_activation=identity
                 ):
        super(Conv2d_MLP_Model_v1, self).__init__()

        if isinstance(observation_space, tuple):
            self.toAugFeatSz = observation_space[1]
            self.observation_space = observation_space[0]
        else:
            self.observation_space = observation_space
            self.toAugFeatSz = 0

        input_channels = self.observation_space.shape[0] - 1

        if paddings is None:
            paddings = [0 for _ in range(len(channels))]
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings), print(len(channels), len(kernel_sizes), strides, len(paddings))
        in_channels = [input_channels] + channels[:-1]

        post_activation_fns = [identity for _ in range(len(strides))]
        ones = [1 for _ in range(len(strides))]
        if use_maxpool:
            post_activation_fns = [torch.nn.MaxPool2d(max_pool_stride) for max_pool_stride in strides]
            strides = ones
        activation_fns = [nonlinearity for _ in range(len(strides))]

        conv_layers = [CNNLayer(input_channels=ic, output_channels=oc,
                                kernel_size=k, stride=s, padding=p, activation_fn=a_fn, post_activation_fn=p_fn)
                       for (ic, oc, k, s, p, a_fn, p_fn) in zip(in_channels, channels, kernel_sizes, strides, paddings,
                                                                activation_fns, post_activation_fns)]

        self.cnn = nn.Sequential()
        for i, cnn_layer in enumerate(conv_layers):
            self.cnn.add_module("cnn_{}".format(i), cnn_layer)

        # Compute shape by doing one forward pass
        dummy = torch.as_tensor(self.observation_space.sample()[None]).float()
        with torch.no_grad():
            self.n_flatten = self.cnn(dummy[:, :3, :, :]).flatten().shape[0]

        print("QQQ", self.n_flatten)
        self.fc_layers = MLP(
            input_size=self.n_flatten + self.toAugFeatSz,
            output_size=fc_output_size,
            hidden_sizes=fc_hidden_sizes,
            hidden_activation=fc_hidden_activation,
            output_activation=fc_output_activation
        )

        # self.conv_mlp = CNN(cnn_layers=conv_layers, fc_layers=fc_layers)

    def get_cnn_features(self, x):
        """
        Get the output of CNN.
        """
        if len(x.size()) == 3:
            x = x.unsqueeze(0)
        x = self.cnn(x)
        # flatten x
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        """
        Forward method implementation.
        x: torch.Tensor
        :return: torch.Tensor
        """
        x = self.get_cnn_features(x)
        x = self.fc_layers(x)
        return x



class Conv2d_Flatten_MLP_v1(Conv2d_MLP_Model_v1):
    """
    Augmented convolution neural network, in which a feature vector will be appended to
        the features extracted by CNN before entering mlp
    """
    # noinspection PyDefaultArgument
    def __init__(self,
                 # conv2d layer arguments
                 observation_space,
                 # fc layer arguments
                 fc_output_size,
                 # conv2d optional arguments
                 channels=[32, 32, 32],
                 kernel_sizes=[8, 4, 3],
                 strides=[4, 2, 1],
                 paddings=[0, 1, 1],
                 nonlinearity=torch.relu,
                 use_maxpool=False,
                 # fc layer optional arguments
                 fc_hidden_sizes=[100, 100],
                 fc_hidden_activation=torch.relu,
                 fc_output_activation=identity
                 ):
        super(Conv2d_Flatten_MLP_v1, self).__init__(observation_space=observation_space,
                                                 fc_output_size=fc_output_size,
                                                 channels=channels, kernel_sizes=kernel_sizes, strides=strides,
                                                 paddings=paddings, nonlinearity=nonlinearity,
                                                 use_maxpool=use_maxpool, fc_hidden_sizes=fc_hidden_sizes,
                                                 fc_hidden_activation=fc_hidden_activation,
                                                 fc_output_activation=fc_output_activation)

    def forward(self, obs, actions=None):
        num_predicates = self.toAugFeatSz if not isinstance(actions, torch.Tensor) else self.toAugFeatSz - actions.size()[1]
        # print("MMM", num_predicates, self.toAugFeatSz)
        if len(obs.shape) == 3:
            # https://github.com/hill-a/stable-baselines/issues/133#issuecomment-561805417
            # Take last channel as extra augmenting features
            augment_features = torch.flatten(obs[-1, ...])
            # Take known amount of direct features, rest are padding zeros
            augment_features = augment_features[:num_predicates]
            # print("CCC", augment_features)
            obs = obs[:-1, ...]
            cnn_features = self.get_cnn_features(obs)
            features = torch.cat((cnn_features.squeeze(0), augment_features))
            print("WWW", features.size())
            if isinstance(actions, torch.Tensor):
                features = torch.cat((features, actions))

        if len(obs.shape) == 4:
            obs, augment_features = obs[:, :-1, :, :], obs[:, -1, :, :]
            augment_features = torch.flatten(augment_features, start_dim=1)
            augment_features = augment_features[:, :num_predicates]
            # print("CCC", augment_features[0])
            cnn_features = self.get_cnn_features(obs)
            features = torch.cat((cnn_features, augment_features), dim=1)
            if isinstance(actions, torch.Tensor):
                features = torch.cat((features, actions), dim=1)

        return self.fc_layers(features)

class Conv2d_Flatten_MLP_v2(Conv2d_MLP_Model_v1):
    """
    Augmented convolution neural network, in which a feature vector will be appended to
        the features extracted by CNN before entering mlp
    """
    # noinspection PyDefaultArgument
    def __init__(self,
                 # conv2d layer arguments
                 observation_space,
                 # fc layer arguments
                 fc_output_size,
                 # conv2d optional arguments
                 channels=[32, 32, 32],
                 kernel_sizes=[8, 4, 3],
                 strides=[4, 2, 1],
                 paddings=[0, 1, 1],
                 nonlinearity=torch.relu,
                 use_maxpool=False,
                 # fc layer optional arguments
                 fc_hidden_sizes=[100, 100],
                 fc_hidden_activation=torch.relu,
                 fc_output_activation=identity
                 ):
        super(Conv2d_Flatten_MLP_v2, self).__init__(observation_space=observation_space,
                                                 fc_output_size=fc_output_size,
                                                 channels=channels, kernel_sizes=kernel_sizes, strides=strides,
                                                 paddings=paddings, nonlinearity=nonlinearity,
                                                 use_maxpool=use_maxpool, fc_hidden_sizes=fc_hidden_sizes,
                                                 fc_hidden_activation=fc_hidden_activation,
                                                 fc_output_activation=fc_output_activation)
    def forward(self, obs, actions=None):
        num_predicates = self.toAugFeatSz if not isinstance(actions, torch.Tensor) else self.toAugFeatSz - actions.size()[1]
        # print("MMM", num_predicates, self.toAugFeatSz)
        if len(obs.shape) == 3:
            # https://github.com/hill-a/stable-baselines/issues/133#issuecomment-561805417
            # Take last channel as extra augmenting features
            augment_features = torch.flatten(obs[-1, ...])
            # Take known amount of direct features, rest are padding zeros
            augment_features = augment_features[:num_predicates]
            # print("CCC", augment_features)
            obs = obs[:-1, ...]
            cnn_features = self.get_cnn_features(obs)
            features = torch.cat((cnn_features.squeeze(0), augment_features))

            if isinstance(actions, torch.Tensor):
                features = torch.cat((features, actions))
            out1 = self.fc_layers(features)
            return out1

        if len(obs.shape) == 4:
            obs, augment_features = obs[:, :-1, :, :], obs[:, -1, :, :]
            augment_features = torch.flatten(augment_features, start_dim=1)
            augment_features = augment_features[:, :num_predicates]
            # print("CCC", augment_features[3])

            cnn_features = self.get_cnn_features(obs)
            features = torch.cat((cnn_features, augment_features), dim=1)
            if isinstance(actions, torch.Tensor):
                features = torch.cat((features, actions), dim=1)
            # print("XXX", features.shape, obs.shape, isinstance(actions, torch.Tensor))
            out1 = self.fc_layers(features)
            return out1

class Conv2d_Flatten_MLP_v3(Conv2d_MLP_Model_v1):
    """
    Augmented convolution neural network, in which a feature vector will be appended to
        the features extracted by CNN before entering mlp
    """
    # noinspection PyDefaultArgument
    def __init__(self,
                 # conv2d layer arguments
                 observation_space,
                 # fc layer arguments
                 fc_output_size,
                 # conv2d optional arguments
                 channels=[32, 32, 32],
                 kernel_sizes=[8, 4, 3],
                 strides=[4, 2, 1],
                 paddings=[0, 1, 1],
                 nonlinearity=torch.relu,
                 use_maxpool=False,
                 # fc layer optional arguments
                 fc_hidden_sizes=[100, 100],
                 fc_hidden_activation=torch.relu,
                 fc_output_activation=identity
                 ):
        super(Conv2d_Flatten_MLP_v3, self).__init__(observation_space=observation_space,
                                                 fc_output_size=fc_output_size,
                                                 channels=channels, kernel_sizes=kernel_sizes, strides=strides,
                                                 paddings=paddings, nonlinearity=nonlinearity,
                                                 use_maxpool=use_maxpool, fc_hidden_sizes=fc_hidden_sizes,
                                                 fc_hidden_activation=fc_hidden_activation,
                                                 fc_output_activation=fc_output_activation)
        self.fc_layers_2 = MLP(
            input_size=self.n_flatten + self.toAugFeatSz,
            output_size=2,
            hidden_sizes=[fc_hidden_sizes[0]],
            hidden_activation=fc_hidden_activation,
            output_activation=torch.sigmoid
        )
        self.Softmax = torch.nn.Softmax()

    def forward(self, obs, actions=None):
        num_predicates = self.toAugFeatSz if not isinstance(actions, torch.Tensor) else self.toAugFeatSz - actions.size()[1]
        # print("MMM", num_predicates, self.toAugFeatSz)
        if len(obs.shape) == 3:
            # https://github.com/hill-a/stable-baselines/issues/133#issuecomment-561805417
            # Take last channel as extra augmenting features
            augment_features = torch.flatten(obs[-1, ...])
            # Take known amount of direct features, rest are padding zeros
            augment_features = augment_features[:num_predicates]
            # print("CCC", augment_features)
            obs = obs[:-1, ...]
            cnn_features = self.get_cnn_features(obs)
            features = torch.cat((cnn_features.squeeze(0), augment_features))

            if isinstance(actions, torch.Tensor):
                features = torch.cat((features, actions))
            out1 = self.fc_layers(features)
            if isinstance(actions, torch.Tensor):
                return out1
            out2 = self.Softmax(self.fc_layers_2(features))
            return torch.cat((out2, out1))

        if len(obs.shape) == 4:
            obs, augment_features = obs[:, :-1, :, :], obs[:, -1, :, :]
            augment_features = torch.flatten(augment_features, start_dim=1)
            augment_features = augment_features[:, :num_predicates]
            # print("CCC", augment_features[3])

            cnn_features = self.get_cnn_features(obs)
            features = torch.cat((cnn_features, augment_features), dim=1)
            if isinstance(actions, torch.Tensor):
                features = torch.cat((features, actions), dim=1)
            # print("XXX", features.shape, obs.shape, isinstance(actions, torch.Tensor))
            out1 = self.fc_layers(features)
            if isinstance(actions, torch.Tensor):
                return out1
            out2 = self.Softmax(self.fc_layers_2(features))
            return torch.cat((out2, out1), dim=1)