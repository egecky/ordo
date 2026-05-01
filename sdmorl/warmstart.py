from typing import Dict, Any, Optional
import numpy as np
import torch
from .rollout import collect_trajectories, returns_matrix, logprob_sums, entropy_means


class WarmstartConfig:
    def __init__(
        self,
        gamma=0.99,
        max_steps=200,
        episodes_per_iter=256,
        eta=0.001,
        entropy_coef=0.01,
        reward_indices=(0, 1),
        weights=None,
    ):
        self.gamma = gamma
        self.max_steps = max_steps
        self.episodes_per_iter = episodes_per_iter
        self.eta = eta
        self.entropy_coef = entropy_coef
        self.reward_indices = reward_indices
        self.weights = weights


def warmstart_step(env, policy, policy_kind, device, cfg):
    traj = collect_trajectories(
        env, policy, policy_kind, device, cfg.gamma, cfg.max_steps, cfg.episodes_per_iter
    )
    env_steps = int(sum((len(tr.rewards) for tr in traj)))
    R_np = returns_matrix(traj, cfg.gamma)
    if cfg.reward_indices is not None:
        R_np = R_np[:, list(cfg.reward_indices)]
    K = R_np.shape[-1]
    w = cfg.weights if cfg.weights is not None else np.ones(K, dtype=np.float32) / K
    s = torch.tensor(R_np @ w, dtype=torch.float32, device=device)
    logp = logprob_sums(traj)
    ent = entropy_means(traj)
    adv = (s - s.mean()).detach()
    adv = adv / (adv.std(unbiased=False) + 1e-08)
    adv = torch.clamp(adv, -5.0, 5.0)
    loss = -(adv * logp).mean() - cfg.entropy_coef * ent.mean()
    return {
        "loss": float(loss.detach().cpu().item()),
        "entropy_mean": float(ent.detach().cpu().mean().item()),
        "R_mean": R_np.mean(axis=0).tolist(),
        "env_steps": int(env_steps),
        "loss_tensor": loss,
    }


def alpha_to_weights(alpha):
    a = float(alpha)
    a = max(0.0, min(1.0, a))
    return np.asarray([a, 1.0 - a], dtype=np.float32)


def seed_to_alpha_grid(seed, n, a_min, a_max):
    if n <= 1:
        return float(a_max)
    idx = int(seed) % int(n)
    frac = idx / (n - 1)
    return float(a_min + (a_max - a_min) * frac)
