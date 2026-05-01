from typing import Literal
import math
import torch
import torch.nn.functional as F

SmoothKind = Literal["relu", "softplus", "gaussian"]


def relu(x):
    return torch.clamp(x, min=0.0)


def softplus_smooth(x, beta=1.0):
    b = torch.tensor(beta, device=x.device, dtype=x.dtype)
    return b * F.softplus(x / b)


def gaussian_soft_relu(x, mu=1.0):
    m = torch.tensor(mu, device=x.device, dtype=x.dtype)
    z = x / (m * math.sqrt(2.0))
    Phi = 0.5 * (1.0 + torch.erf(z))
    phi = 1.0 / (m * math.sqrt(2.0 * math.pi)) * torch.exp(-0.5 * (x / m) ** 2)
    return x * Phi + m**2 * phi


def smooth_pos(x, kind, param):
    if kind == "relu":
        return relu(x)
    if kind == "softplus":
        return softplus_smooth(x, beta=param)
    if kind == "gaussian":
        return gaussian_soft_relu(x, mu=param)
    raise ValueError(f"Unknown smoothing kind: {kind}")
