from typing import Dict, Any
import numpy as np
from .rollout import collect_trajectories, returns_matrix
from .utils import pareto_nondominated, hypervolume_2d, hypervolume_mc


class EvalConfig:
    def __init__(self, gamma=0.99, max_steps=200, num_episodes=200):
        self.gamma = gamma
        self.max_steps = max_steps
        self.num_episodes = num_episodes


def eval_policy(env, policy, policy_kind, device, cfg):
    traj = collect_trajectories(
        env, policy, policy_kind, device, cfg.gamma, cfg.max_steps, cfg.num_episodes
    )
    R = returns_matrix(traj, cfg.gamma)
    return {
        "R_mean": R.mean(axis=0).tolist(),
        "R_std": R.std(axis=0).tolist(),
        "R_samples": R.tolist(),
    }


def estimate_hypervolume(points, ref):
    d = points.shape[1]
    if d == 2:
        return hypervolume_2d(points, ref)
    return hypervolume_mc(points, ref, samples=200000)
