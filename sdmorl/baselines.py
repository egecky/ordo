from typing import Dict, Any, Optional
import numpy as np
import torch
from .rollout import collect_trajectories, returns_matrix, logprob_sums, entropy_means


class WeightedSumConfig:
    def __init__(
        self,
        gamma=0.99,
        reward_indices=(0, 1),
        max_steps=200,
        episodes_per_iter=8,
        eta=0.0003,
        weights=None,
        entropy_coef=0.0,
    ):
        self.gamma = gamma
        self.reward_indices = reward_indices
        self.max_steps = max_steps
        self.episodes_per_iter = episodes_per_iter
        self.eta = eta
        self.weights = weights
        self.entropy_coef = entropy_coef


def weighted_sum_step(env, policy, policy_kind, device, cfg):
    traj = collect_trajectories(
        env, policy, policy_kind, device, cfg.gamma, cfg.max_steps, cfg.episodes_per_iter
    )
    R_np = returns_matrix(traj, cfg.gamma)
    if cfg.reward_indices is not None:
        R_np = R_np[:, list(cfg.reward_indices)]
    d = R_np.shape[-1]
    w = cfg.weights if cfg.weights is not None else np.ones(d, dtype=np.float32) / d
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
        "loss_tensor": loss,
    }
