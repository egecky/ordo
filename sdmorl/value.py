import numpy as np
import torch
import torch.nn as nn


def mlp(sizes, activation=nn.Tanh, out_activation=None):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else out_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1])]
        if act is not None:
            layers += [act()]
    return nn.Sequential(*layers)


class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden=(64, 64)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, 1], activation=nn.Tanh, out_activation=None)
        layers = [m for m in self.net.modules() if isinstance(m, nn.Linear)]
        for (i, layer) in enumerate(layers):
            gain = np.sqrt(2.0) if i < len(layers) - 1 else 1.0
            nn.init.orthogonal_(layer.weight, gain=gain)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, obs):
        v = self.net(obs)
        return v.squeeze(-1)


def build_value(env, hidden=(64, 64)):
    obs_dim = int(np.prod(env.observation_space.shape))
    return ValueNet(obs_dim, hidden=hidden)
