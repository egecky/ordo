from typing import Dict, Any, Optional
import numpy as np
import torch
from .hk import hk, ck_from_kd
from .smoothing import SmoothKind
from .rollout import collect_trajectories, returns_matrix, logprob_sums, entropy_means
from .utils import dpmorl_normalize_returns
from .ordo import estimate_M_bounds


class SoftOrdoConfig:
    def __init__(
        self,
        k=2,
        gamma=0.99,
        max_steps=200,
        episodes_per_iter=8,
        eta=0.0003,
        kappa=0.5,
        grid_points_per_dim=15,
        smooth="softplus",
        smooth_param=0.5,
        independent_batches=False,
        m_grid_margin=0.1,
        reward_indices=(0, 1),
        normalize_returns=False,
        entropy_coef=0.0,
    ):
        self.k = k
        self.gamma = gamma
        self.max_steps = max_steps
        self.episodes_per_iter = episodes_per_iter
        self.eta = eta
        self.kappa = kappa
        self.grid_points_per_dim = grid_points_per_dim
        self.smooth = smooth
        self.smooth_param = smooth_param
        self.independent_batches = independent_batches
        self.m_grid_margin = m_grid_margin
        self.reward_indices = reward_indices
        self.normalize_returns = normalize_returns
        self.entropy_coef = entropy_coef


def build_m_grid(m_lo, m_hi, points_per_dim):
    d = m_lo.shape[0]
    axes = [np.linspace(m_lo[i], m_hi[i], points_per_dim) for i in range(d)]
    mesh = np.meshgrid(*axes, indexing="ij")
    grid = np.stack([m.reshape(-1) for m in mesh], axis=-1)
    return grid.astype(np.float32)


def soft_ordo_step(
    env,
    policy,
    policy_kind,
    device,
    cfg,
    ref_policy=None,
    ref_kind=None,
    running_min=None,
    running_max=None,
    m_grid=None,
):
    ref_policy = policy if ref_policy is None else ref_policy
    ref_kind = policy_kind if ref_kind is None else ref_kind
    traj_ref = collect_trajectories(
        env, ref_policy, ref_kind, device, cfg.gamma, cfg.max_steps, cfg.episodes_per_iter
    )
    traj_plus = collect_trajectories(
        env, policy, policy_kind, device, cfg.gamma, cfg.max_steps, cfg.episodes_per_iter
    )
    if cfg.independent_batches:
        traj_q = collect_trajectories(
            env, policy, policy_kind, device, cfg.gamma, cfg.max_steps, cfg.episodes_per_iter
        )
    else:
        traj_q = traj_plus
    R_ref_np = returns_matrix(traj_ref, cfg.gamma)
    R_plus_np = returns_matrix(traj_plus, cfg.gamma)
    if cfg.reward_indices is not None:
        R_ref_np = R_ref_np[:, list(cfg.reward_indices)]
        R_plus_np = R_plus_np[:, list(cfg.reward_indices)]
    R_q_np = returns_matrix(traj_q, cfg.gamma)
    if cfg.reward_indices is not None:
        R_q_np = R_q_np[:, list(cfg.reward_indices)]
    if running_min is None:
        running_min = np.min(np.vstack([R_ref_np, R_plus_np]), axis=0)
        running_max = np.max(np.vstack([R_ref_np, R_plus_np]), axis=0)
    else:
        running_min = np.minimum(running_min, np.min(np.vstack([R_ref_np, R_plus_np]), axis=0))
        running_max = np.maximum(running_max, np.max(np.vstack([R_ref_np, R_plus_np]), axis=0))
    if cfg.normalize_returns:
        R_ref_np_n = dpmorl_normalize_returns(R_ref_np, running_min, running_max)
        R_plus_np_n = dpmorl_normalize_returns(R_plus_np, running_min, running_max)
        R_q_np_n = dpmorl_normalize_returns(R_q_np, running_min, running_max)
        m_lo = -cfg.m_grid_margin * np.ones(R_ref_np_n.shape[1], dtype=np.float32)
        m_hi = (1.0 + cfg.m_grid_margin) * np.ones(R_ref_np_n.shape[1], dtype=np.float32)
        if m_grid is None:
            m_grid = build_m_grid(m_lo, m_hi, cfg.grid_points_per_dim)
        R_ref = torch.tensor(R_ref_np_n, dtype=torch.float32, device=device)
        R_plus = torch.tensor(R_plus_np_n, dtype=torch.float32, device=device)
        R_q = torch.tensor(R_q_np_n, dtype=torch.float32, device=device)
    else:
        (m_lo, m_hi) = estimate_M_bounds(running_min, running_max, cfg.m_grid_margin)
        if m_grid is None:
            m_grid = build_m_grid(m_lo, m_hi, cfg.grid_points_per_dim)
        R_ref = torch.tensor(R_ref_np, dtype=torch.float32, device=device)
        R_plus = torch.tensor(R_plus_np, dtype=torch.float32, device=device)
        R_q = torch.tensor(R_q_np, dtype=torch.float32, device=device)
    M = torch.tensor(m_grid, dtype=torch.float32, device=device)
    d = R_ref.shape[-1]
    ck = ck_from_kd(cfg.k, d)

    def hk_mat(R):
        return hk(R[:, None, :], M[None, :, :], cfg.k, cfg.smooth, cfg.smooth_param)

    H_q = hk_mat(R_q).mean(dim=0)
    H_ref = hk_mat(R_ref).mean(dim=0)
    H_hat = ck * (H_q - H_ref)
    q = torch.softmax(H_hat / cfg.kappa, dim=0).detach()
    H_plus = hk_mat(R_plus)
    w = (H_plus * q[None, :]).sum(dim=1).detach()
    adv = w - w.mean()
    adv = adv / (adv.std(unbiased=False) + 1e-08)
    adv = torch.clamp(adv, -5.0, 5.0)
    logp = logprob_sums(traj_plus)
    ent = entropy_means(traj_plus)
    loss = (adv * logp).mean() - cfg.entropy_coef * ent.mean()
    sd_violation = float(torch.max(H_hat).detach().cpu().item())
    q_entropy = float((-q * torch.log(q + 1e-12)).sum().detach().cpu().item())
    info = {
        "loss": float(loss.detach().cpu().item()),
        "entropy_mean": float(ent.detach().cpu().mean().item()),
        "sd_violation_max_m": sd_violation,
        "q_entropy": q_entropy,
        "R_ref_mean": R_ref_np.mean(axis=0).tolist(),
        "R_plus_mean": R_plus_np.mean(axis=0).tolist(),
        "running_min": running_min.tolist(),
        "running_max": running_max.tolist(),
        "loss_tensor": loss,
        "running_min_arr": running_min,
        "running_max_arr": running_max,
        "m_grid": m_grid,
    }
    return info
