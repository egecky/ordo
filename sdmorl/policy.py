from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


def _orthogonal_init_(m, gain):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def mlp(sizes, activation=nn.Tanh, out_activation=None):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else out_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1])]
        if act is not None:
            layers += [act()]
    return nn.Sequential(*layers)


class DiscretePolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(64, 64)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, act_dim], activation=nn.Tanh, out_activation=None)
        layers = [m for m in self.net.modules() if isinstance(m, nn.Linear)]
        for (i, layer) in enumerate(layers):
            gain = np.sqrt(2.0) if i < len(layers) - 1 else 0.01
            _orthogonal_init_(layer, gain)

    def forward(self, obs):
        logits = self.net(obs)
        return Categorical(logits=logits)


class ContinuousPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, act_low, act_high, hidden=(64, 64), log_std_init=0.0):
        super().__init__()
        self.mu_net = mlp([obs_dim, *hidden, act_dim], activation=nn.Tanh, out_activation=None)
        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std_init)
        self.register_buffer("act_low", torch.tensor(act_low, dtype=torch.float32))
        self.register_buffer("act_high", torch.tensor(act_high, dtype=torch.float32))
        layers = [m for m in self.mu_net.modules() if isinstance(m, nn.Linear)]
        for (i, layer) in enumerate(layers):
            gain = np.sqrt(2.0) if i < len(layers) - 1 else 0.01
            _orthogonal_init_(layer, gain)

    def forward(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)


def build_policy(env, hidden=(64, 64)):
    obs_dim = int(np.prod(env.observation_space.shape))
    act_space = env.action_space
    if hasattr(act_space, "n"):
        return (DiscretePolicy(obs_dim, act_space.n, hidden=hidden), "discrete")
    if hasattr(act_space, "shape"):
        act_dim = int(np.prod(act_space.shape))
        act_low = np.asarray(act_space.low, dtype=np.float32).reshape(-1)
        act_high = np.asarray(act_space.high, dtype=np.float32).reshape(-1)
        return (
            ContinuousPolicy(obs_dim, act_dim, act_low=act_low, act_high=act_high, hidden=hidden),
            "continuous",
        )
    raise ValueError(f"Unsupported action space: {act_space}")


def flatten_obs(obs):
    return np.asarray(obs, dtype=np.float32).reshape(-1)
