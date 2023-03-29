import torch
import torch.nn as nn
import torch.nn.functional as F

from learning_agents.common.common_utils import identity
from learning_agents.architectures.mlp import MLP
from learning_agents.architectures.cnn import CNNLayer


class DefaultActorCNN(nn.Module):
    def __init__(self, input_channel, output_size):
        super(DefaultActorCNN, self).__init__()

        # define CNN layers
        cnn_layers = [
            CNNLayer(input_channels=input_channel, output_channels=6, kernel_size=8, stride=4, padding=1),
            CNNLayer(input_channels=6, output_channels=16, kernel_size=4, stride=2)
        ]
        self.cnn_layers = cnn_layers
        # define FC layers
        self.fc_layers = MLP(
            input_size=3168,
            output_size=output_size,
            hidden_sizes=[128, 128],
            output_activation=torch.tanh,
        )

        self.cnn = nn.Sequential()
        for i, cnn_layer in enumerate(self.cnn_layers):
            self.cnn.add_module("cnn_{}".format(i), cnn_layer)

    def get_cnn_features(self, x):
        """
        Get the output of CNN.
        """
        # if it's three channel rgb observation, flatten it
        if len(x.size()) == 3:
            x = x.unsqueeze(0)
        x = self.cnn(x)
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


class DefaultCriticCNN(nn.Module):
    def __init__(self, input_channel, output_size):
        super(DefaultCriticCNN, self).__init__()

        # define CNN layers
        cnn_layers = [
            CNNLayer(input_channels=input_channel, output_channels=6, kernel_size=8, stride=4, padding=1),
            CNNLayer(input_channels=6, output_channels=16, kernel_size=4, stride=2)
        ]
        self.cnn_layers = cnn_layers
        # define FC layers
        self.fc_layers = MLP(
            input_size=3173,
            output_size=output_size,
            hidden_sizes=[128, 128],
            output_activation=torch.tanh,
        )

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
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        """
        Forward method implementation.
        x: tuple of (state, action)
        :return: torch.Tensor
        """
        state = x[0]
        action = x[1]
        cnn_features = self.get_cnn_features(state)
        cat_feature = torch.cat((cnn_features, action), dim=-1)
        output = self.fc_layers(cat_feature)
        return output