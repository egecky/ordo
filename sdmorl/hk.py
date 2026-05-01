import math
import torch
from .smoothing import smooth_pos, SmoothKind


def ck_from_kd(k, d):
    return float(1.0 / math.factorial(k - 1) ** d)


def hk(R, m, k, smooth="relu", smooth_param=1.0):
    diff = m - R
    pos = smooth_pos(diff, kind=smooth, param=smooth_param)
    if k == 1:
        return torch.prod((pos > 0).to(pos.dtype), dim=-1)
    exp = k - 1
    return torch.prod(pos**exp, dim=-1)


def L_hat(R_a, R_b, m, k, smooth, smooth_param):
    d = R_a.shape[-1]
    ck = ck_from_kd(k, d)
    return ck * (
        hk(R_a, m, k, smooth, smooth_param).mean() - hk(R_b, m, k, smooth, smooth_param).mean()
    )
