from encoder.encoder import Encoder
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import gym

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Unflatten(nn.Module):
    def forward(self, x, size=5120):
        return x.view(x.size(0), 128, 4, 10)


class VAE_Model(nn.Module):
    def __init__(self):
        super(VAE_Model, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4, 3), stride=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(128 * 4 * 10, 32)
        self.fc2 = nn.Linear(128 * 4 * 10, 32)
        self.fc3 = nn.Linear(32, 128 * 4 * 10)

        self.decoder = nn.Sequential(
            Unflatten(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 3), stride=3),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=3),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar


class VAE(Encoder):
    """
    Abstract Encoder.
    Attributes:
        env (gym.Env): openAI Gym environment
        state_dim (int): dimension of states
        action_dim (int): dimension of actions
    """

    def __init__(self, env, args):
        Encoder.__init__(self, env, args)
        self.encoder = VAE_Model()
        self.encoder_optim = optim.Adam(self.encoder.parameters(), lr=self.args.encoder_lr)

    def encode(self, x):
        z, _, _ = self.encoder.encode(x)
        return z.detach().cpu().numpy()

    def load_params(self, path):
        params = torch.load(path)
        self.encoder.load_state_dict(params["encoder_state_dict"])
        self.encoder_optim.load_state_dict(params["encoder_optim_state_dict"])
        print("[INFO] loaded the model and optimizer from", path)

    def save_params(self, params, n_episode):
        """
        n_episode: int
        params: dict
        """
        params = {
            "encoder_state_dict": self.encoder.state_dict(),
            "encoder_optim_state_dict": self.encoder_optim.state_dict(),
            "epoch": n_episode
        }
        torch.save(params, self.args.model_save_dir)

    def train(self):
        pass
