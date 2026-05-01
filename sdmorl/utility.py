"""Utility families used by the training script."""
from typing import Dict, Tuple
import numpy as np


def calibrate_return_stats(env_id, seed, episodes, max_steps, gamma, reward_indices):
    try:
        import mo_gymnasium as mo_gym

        env = mo_gym.make(env_id)
    except Exception:
        import mo_gym

        env = mo_gym.make(env_id)
    if hasattr(env, "_max_episode_steps"):
        env._max_episode_steps = int(max_steps)
    if hasattr(env, "spec") and env.spec is not None:
        try:
            env.spec.max_episode_steps = int(max_steps)
        except Exception:
            pass
    (obs, _) = env.reset(seed=int(seed))
    _ = obs
    Rs = []
    for ep in range(int(episodes)):
        (obs, _) = env.reset()
        done = False
        t = 0
        disc = 1.0
        ret = np.zeros(len(reward_indices), dtype=np.float32)
        while not done and t < int(max_steps):
            a = env.action_space.sample()
            (obs, r, terminated, truncated, _) = env.step(a)
            r = np.asarray(r, dtype=np.float32)
            r = r[list(reward_indices)]
            ret += disc * r
            disc *= float(gamma)
            done = bool(terminated) or bool(truncated)
            t += 1
        Rs.append(ret)
    Rs = np.asarray(Rs, dtype=np.float32)
    mean = Rs.mean(axis=0)
    std = Rs.std(axis=0)
    std = np.maximum(std, 0.001)
    try:
        env.close()
    except Exception:
        pass
    return (mean, std)


def _log_uniform(rng, lo, hi, size):
    lo = float(lo)
    hi = float(hi)
    lo = max(lo, 1e-08)
    hi = max(hi, lo * 1.0001)
    u = rng.uniform(np.log(lo), np.log(hi), size=size)
    return np.exp(u)


def sample_monotone_rf_utility(
    seed, base_w, J=32, kappa=10.0, beta_min=0.5, beta_max=5.0, c_std=1.0
):
    rng = np.random.RandomState(int(seed))
    base_w = np.asarray(base_w, dtype=np.float32).copy()
    if base_w.ndim != 1:
        raise ValueError("base_w must be 1D")
    K = int(base_w.shape[0])
    if K < 1:
        raise ValueError("base_w must have at least 1 dim")
    w = np.maximum(base_w, 0.001)
    w = w / w.sum()
    alpha = float(max(kappa, 0.001)) * w
    b = rng.dirichlet(alpha, size=int(J)).astype(np.float32)
    c = rng.normal(loc=0.0, scale=float(c_std), size=(int(J),)).astype(np.float32)
    beta = _log_uniform(rng, float(beta_min), float(beta_max), size=(int(J),)).astype(np.float32)
    a = rng.exponential(scale=1.0, size=(int(J),)).astype(np.float32)
    a = np.maximum(a, 1e-08)
    a = (a / a.sum()).astype(np.float32)
    return {"a": a, "b": b, "beta": beta, "c": c, "family": np.array([0], dtype=np.int32)}
