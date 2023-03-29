import torch
import torch.nn as nn

from learning_agents.architectures.cnn import Conv2d_MLP_Model


class TrpoActorCNN(nn.Module):
    def __init__(self, input_channels, num_outputs, fc_input_size=1):
        super(TrpoActorCNN, self).__init__()
        self.cnn = Conv2d_MLP_Model(input_channels=input_channels,
                                    fc_input_size=320,
                                    fc_output_size=num_outputs,
                                    channels=[32, 64, 64],
                                    kernel_sizes=[8, 4, 3],
                                    strides=[4, 2, 2],
                                    paddings=[1, 0, 0],
                                    fc_hidden_sizes=[128, 64],
                                    fc_hidden_activation=torch.tanh)

    def forward(self, x):
        mu = self.cnn(x)
        logstd = torch.zeros_like(mu)
        std = torch.exp(logstd)
        return mu, std


class TrpoCriticCNN(nn.Module):
    def __init__(self, input_channels,
                 num_outputs=1,
                 fc_input_size=1,

                 ):
        super(TrpoCriticCNN, self).__init__()
        self.cnn = Conv2d_MLP_Model(input_channels=input_channels,
                                    fc_input_size=320,
                                    fc_output_size=num_outputs,
                                    channels=[32, 64, 64],
                                    kernel_sizes=[8, 4, 3],
                                    strides=[4, 2, 2],
                                    paddings=[1, 0, 0],
                                    fc_hidden_sizes=[128, 64],
                                    fc_hidden_activation=torch.tanh)

    def forward(self, x):
        v = self.cnn(x)
        return v


class TrpoActor(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(TrpoActor, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, num_outputs)

        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mu = self.fc3(x)
        logstd = torch.zeros_like(mu)
        std = torch.exp(logstd)
        return mu, std


class TrpoCritic(nn.Module):
    def __init__(self, num_inputs):
        super(TrpoCritic, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        v = self.fc3(x)
        return v
