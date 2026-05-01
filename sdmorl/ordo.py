from typing import Dict, Any, Tuple
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from .hk import L_hat, hk
from .smoothing import SmoothKind
from .rollout import collect_trajectories, returns_matrix, logprob_sums, entropy_means
from .utils import dpmorl_normalize_returns
from .utility import sample_monotone_rf_utility


class OrdoConfig:
    def __init__(
        self,
        k=2,
        gamma=0.99,
        max_steps=200,
        episodes_per_iter=8,
        inner_m_steps=50,
        inner_m_lr=0.1,
        eta=0.0003,
        eps_progress=0.01,
        smooth="softplus",
        smooth_param=0.5,
        independent_batches=False,
        m_grid_margin=0.1,
        reward_indices=(0, 1),
        normalize_returns=False,
        entropy_coef=0.0,
        action_temp=1.0,
        use_gae=True,
        gae_lambda=0.95,
        vf_coef=0.5,
        std_algo="ac",
        ppo_clip_range=0.2,
        ppo_epochs=10,
        ppo_minibatch_size=64,
        max_grad_norm=0.5,
        std_w=None,
        std_utility_family="linear",
        std_utility_params=None,
        std_utility_calib=None,
        mix_alpha=1.0,
    ):
        self.k = k
        self.gamma = gamma
        self.max_steps = max_steps
        self.episodes_per_iter = episodes_per_iter
        self.inner_m_steps = inner_m_steps
        self.inner_m_lr = inner_m_lr
        self.eta = eta
        self.eps_progress = eps_progress
        self.smooth = smooth
        self.smooth_param = smooth_param
        self.independent_batches = independent_batches
        self.m_grid_margin = m_grid_margin
        self.reward_indices = reward_indices
        self.normalize_returns = normalize_returns
        self.entropy_coef = entropy_coef
        self.action_temp = action_temp
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        self.vf_coef = vf_coef
        self.std_algo = std_algo
        self.ppo_clip_range = ppo_clip_range
        self.ppo_epochs = ppo_epochs
        self.ppo_minibatch_size = ppo_minibatch_size
        self.max_grad_norm = max_grad_norm
        self.std_w = std_w
        self.std_utility_family = std_utility_family
        self.std_utility_params = std_utility_params
        self.std_utility_calib = std_utility_calib
        self.mix_alpha = mix_alpha


def estimate_M_bounds(running_min, running_max, margin):
    span = np.maximum(running_max - running_min, 1e-06)
    lo = running_min - margin * span
    hi = running_max + margin * span
    return (lo, hi)


def optimize_m_hat(R_cand, R_ref, m_lo, m_hi, cfg, device, return_info=False, m_init=None):
    if m_init is None:
        init = (m_lo + m_hi) / 2.0
    else:
        init = (
            m_init.detach().cpu().numpy()
            if isinstance(m_init, torch.Tensor)
            else np.asarray(m_init, dtype=np.float32)
        )
        init = np.clip(init, m_lo, m_hi)
    m = torch.nn.Parameter(torch.tensor(init, dtype=torch.float32, device=device))
    opt = optim.Adam([m], lr=cfg.inner_m_lr)
    t0 = time.perf_counter()
    with torch.no_grad():
        initial_obj = float(
            L_hat(R_cand, R_ref, m, cfg.k, cfg.smooth, cfg.smooth_param).detach().cpu().item()
        )
    best_obj = initial_obj
    for _ in range(cfg.inner_m_steps):
        opt.zero_grad()
        obj = L_hat(R_cand, R_ref, m, cfg.k, cfg.smooth, cfg.smooth_param)
        loss = -obj
        loss.backward()
        opt.step()
        with torch.no_grad():
            m.clamp_(torch.tensor(m_lo, device=device), torch.tensor(m_hi, device=device))
            best_obj = max(best_obj, float(obj.detach().cpu().item()))
    elapsed = float(time.perf_counter() - t0)
    final_obj_t = L_hat(R_cand, R_ref, m, cfg.k, cfg.smooth, cfg.smooth_param)
    final_obj = float(final_obj_t.detach().cpu().item())
    grad = torch.autograd.grad(
        final_obj_t, m, retain_graph=False, create_graph=False, allow_unused=True
    )[0]
    grad_norm = float("nan") if grad is None else float(grad.detach().norm().cpu().item())
    out = m.detach()
    if not return_info:
        return out
    info = {
        "m_obj_initial": float(initial_obj),
        "m_obj_final": float(final_obj),
        "m_obj_best_seen": float(max(best_obj, final_obj)),
        "m_obj_improvement": float(final_obj - initial_obj),
        "m_grad_norm": grad_norm,
        "m_steps": int(cfg.inner_m_steps),
        "m_lr": float(cfg.inner_m_lr),
        "m_time_sec": elapsed,
    }
    return (out, info)


def ordo_step(
    env,
    policy,
    policy_kind,
    device,
    cfg,
    ref_policy=None,
    ref_kind=None,
    running_min=None,
    running_max=None,
    value_fn=None,
):
    ref_policy = policy if ref_policy is None else ref_policy
    ref_kind = policy_kind if ref_kind is None else ref_kind
    traj_ref = collect_trajectories(
        env,
        ref_policy,
        ref_kind,
        device,
        cfg.gamma,
        cfg.max_steps,
        cfg.episodes_per_iter,
        action_temp=1.0,
    )
    traj_cand = collect_trajectories(
        env,
        policy,
        policy_kind,
        device,
        cfg.gamma,
        cfg.max_steps,
        cfg.episodes_per_iter,
        action_temp=float(cfg.action_temp),
    )
    env_steps_ref = sum((len(tr.rewards) for tr in traj_ref))
    env_steps_cand = sum((len(tr.rewards) for tr in traj_cand))
    env_steps = int(env_steps_ref + env_steps_cand)
    R_ref_np = returns_matrix(traj_ref, cfg.gamma)
    R_cand_np = returns_matrix(traj_cand, cfg.gamma)
    if cfg.reward_indices is not None:
        R_ref_np = R_ref_np[:, list(cfg.reward_indices)]
        R_cand_np = R_cand_np[:, list(cfg.reward_indices)]
    if cfg.independent_batches:
        traj_cand_m = collect_trajectories(
            env,
            policy,
            policy_kind,
            device,
            cfg.gamma,
            cfg.max_steps,
            cfg.episodes_per_iter,
            action_temp=float(cfg.action_temp),
        )
        env_steps_m = sum((len(tr.rewards) for tr in traj_cand_m))
        env_steps = int(env_steps + env_steps_m)
        R_cand_m_np = returns_matrix(traj_cand_m, cfg.gamma)
        if cfg.reward_indices is not None:
            R_cand_m_np = R_cand_m_np[:, list(cfg.reward_indices)]
    else:
        R_cand_m_np = R_cand_np
    if running_min is None:
        running_min = np.min(np.vstack([R_ref_np, R_cand_np]), axis=0)
        running_max = np.max(np.vstack([R_ref_np, R_cand_np]), axis=0)
    else:
        running_min = np.minimum(running_min, np.min(np.vstack([R_ref_np, R_cand_np]), axis=0))
        running_max = np.maximum(running_max, np.max(np.vstack([R_ref_np, R_cand_np]), axis=0))
    if cfg.normalize_returns:
        R_ref_np_n = dpmorl_normalize_returns(R_ref_np, running_min, running_max)
        R_cand_np_n = dpmorl_normalize_returns(R_cand_np, running_min, running_max)
        R_cand_m_np_n = dpmorl_normalize_returns(R_cand_m_np, running_min, running_max)
        m_lo = -cfg.m_grid_margin * np.ones(R_ref_np_n.shape[1], dtype=np.float32)
        m_hi = (1.0 + cfg.m_grid_margin) * np.ones(R_ref_np_n.shape[1], dtype=np.float32)
        R_ref = torch.tensor(R_ref_np_n, dtype=torch.float32, device=device)
        R_cand = torch.tensor(R_cand_np_n, dtype=torch.float32, device=device)
        R_cand_m = torch.tensor(R_cand_m_np_n, dtype=torch.float32, device=device)
    else:
        (m_lo, m_hi) = estimate_M_bounds(running_min, running_max, cfg.m_grid_margin)
        R_ref = torch.tensor(R_ref_np, dtype=torch.float32, device=device)
        R_cand = torch.tensor(R_cand_np, dtype=torch.float32, device=device)
        R_cand_m = torch.tensor(R_cand_m_np, dtype=torch.float32, device=device)
    (m_hat, m_info) = optimize_m_hat(R_cand_m, R_ref, m_lo, m_hi, cfg, device, return_info=True)
    Lm = float(
        L_hat(R_cand, R_ref, m_hat, cfg.k, cfg.smooth, cfg.smooth_param).detach().cpu().item()
    )
    logp = logprob_sums(traj_cand)
    ent_chunks = [torch.stack(tr.entropies) for tr in traj_cand if len(tr.entropies) > 0]
    all_ent_steps = torch.cat(ent_chunks, dim=0) if ent_chunks else torch.tensor([], device=device)
    ent_mean_all = (
        all_ent_steps.mean() if all_ent_steps.numel() > 0 else torch.tensor(0.0, device=device)
    )
    hk_vals = hk(R_cand, m_hat, cfg.k, cfg.smooth, cfg.smooth_param).detach()
    adv_ordo = hk_vals - hk_vals.mean()
    adv_ordo = adv_ordo / (adv_ordo.std(unbiased=False) + 1e-08)
    adv_ordo = torch.clamp(adv_ordo, -5.0, 5.0)
    pg_loss_ordo = (adv_ordo * logp).mean()
    pg_loss_std = None
    vf_loss = None
    if cfg.std_w is not None:
        w = np.asarray(cfg.std_w, dtype=np.float32)
        if w.ndim != 1 or w.shape[0] != R_cand.shape[1]:
            raise ValueError(f"std_w must have shape ({R_cand.shape[1]},), got {w.shape}")
        w_t = torch.tensor(w, dtype=torch.float32, device=device)
        if cfg.std_utility_family == "linear" and value_fn is not None and cfg.use_gae:
            logp_steps = []
            adv_steps = []
            v_steps = []
            target_v_steps = []
            for tr in traj_cand:
                T = len(tr.rewards)
                if T == 0:
                    continue
                r_vec = np.stack(tr.rewards, axis=0)
                if cfg.reward_indices is not None:
                    r_vec = r_vec[:, list(cfg.reward_indices)]
                r = torch.tensor(r_vec, dtype=torch.float32, device=device) @ w_t
                obs_t = torch.tensor(np.stack(tr.obs, axis=0), dtype=torch.float32, device=device)
                next_obs_t = torch.tensor(
                    np.stack(tr.next_obs, axis=0), dtype=torch.float32, device=device
                )
                v = value_fn(obs_t)
                v_next = value_fn(next_obs_t)
                done = torch.zeros(T, dtype=torch.float32, device=device)
                if tr.terminated:
                    done[-1] = 1.0
                v_next = v_next * (1.0 - done)
                delta = r + cfg.gamma * v_next - v
                adv = torch.zeros(T, dtype=torch.float32, device=device)
                gae = 0.0
                for t in reversed(range(T)):
                    gae = delta[t] + cfg.gamma * cfg.gae_lambda * (1.0 - done[t]) * gae
                    adv[t] = gae
                target_v = adv + v
                logp_steps.append(torch.stack(tr.logprobs))
                adv_steps.append(adv)
                v_steps.append(v)
                target_v_steps.append(target_v.detach())
            if logp_steps:
                logp_all = torch.cat(logp_steps, dim=0)
                adv_all = torch.cat(adv_steps, dim=0)
                v_all = torch.cat(v_steps, dim=0)
                target_v_all = torch.cat(target_v_steps, dim=0)
                adv_all_n = adv_all - adv_all.mean()
                adv_all_n = adv_all_n / (adv_all_n.std(unbiased=False) + 1e-08)
                adv_all_n = torch.clamp(adv_all_n, -5.0, 5.0)
                pg_loss_std = -(adv_all_n.detach() * logp_all).mean()
                vf_loss = 0.5 * ((v_all - target_v_all) ** 2).mean()
            else:
                pg_loss_std = None
                vf_loss = None
        else:
            if cfg.std_utility_family == "linear":
                u = (R_cand * w_t[None, :]).sum(dim=1)
            elif cfg.std_utility_family == "monotone_rf":
                params = cfg.std_utility_params or {}
                a = torch.tensor(np.asarray(params.get("a"), dtype=np.float32), device=device)
                b = torch.tensor(np.asarray(params.get("b"), dtype=np.float32), device=device)
                beta = torch.tensor(np.asarray(params.get("beta"), dtype=np.float32), device=device)
                c = torch.tensor(np.asarray(params.get("c"), dtype=np.float32), device=device)
                if cfg.std_utility_calib is not None:
                    mean = torch.tensor(
                        np.asarray(cfg.std_utility_calib.get("mean"), dtype=np.float32),
                        device=device,
                    )
                    std = torch.tensor(
                        np.asarray(cfg.std_utility_calib.get("std"), dtype=np.float32),
                        device=device,
                    )
                else:
                    mean = torch.zeros(R_cand.shape[1], dtype=torch.float32, device=device)
                    std = torch.ones(R_cand.shape[1], dtype=torch.float32, device=device)
                std = torch.clamp(std, min=1e-06)
                Rhat = (R_cand - mean[None, :]) / std[None, :]
                z = (Rhat.unsqueeze(1) * b.unsqueeze(0)).sum(dim=-1) - c.unsqueeze(0)
                u = torch.nn.functional.softplus(z * beta.unsqueeze(0))
                u = (u * a.unsqueeze(0)).sum(dim=1)
            else:
                raise ValueError(f"Unknown std_utility_family: {cfg.std_utility_family}")
            if value_fn is not None:
                logp_steps = []
                adv_steps = []
                v_steps = []
                target_v_steps = []
                u_np = u.detach()
                for (tr, u_tr) in zip(traj_cand, u_np):
                    T = len(tr.rewards)
                    if T == 0:
                        continue
                    obs_t = torch.tensor(
                        np.stack(tr.obs, axis=0), dtype=torch.float32, device=device
                    )
                    v = value_fn(obs_t)
                    u_rep = torch.full((T,), float(u_tr.item()), dtype=torch.float32, device=device)
                    adv = u_rep - v
                    logp_steps.append(torch.stack(tr.logprobs))
                    adv_steps.append(adv)
                    v_steps.append(v)
                    target_v_steps.append(u_rep.detach())
                if logp_steps:
                    logp_all = torch.cat(logp_steps, dim=0)
                    adv_all = torch.cat(adv_steps, dim=0)
                    v_all = torch.cat(v_steps, dim=0)
                    target_v_all = torch.cat(target_v_steps, dim=0)
                    adv_all_n = adv_all - adv_all.mean()
                    adv_all_n = adv_all_n / (adv_all_n.std(unbiased=False) + 1e-08)
                    adv_all_n = torch.clamp(adv_all_n, -5.0, 5.0)
                    pg_loss_std = -(adv_all_n.detach() * logp_all).mean()
                    vf_loss = 0.5 * ((v_all - target_v_all) ** 2).mean()
                else:
                    pg_loss_std = None
                    vf_loss = None
            else:
                adv_u = (u - u.mean()).detach()
                adv_u = adv_u / (adv_u.std(unbiased=False) + 1e-08)
                adv_u = torch.clamp(adv_u, -5.0, 5.0)
                pg_loss_std = -(adv_u * logp).mean()
    alpha = float(np.clip(cfg.mix_alpha, 0.0, 1.0))
    if pg_loss_std is None:
        pg_loss = pg_loss_ordo
    else:
        pg_loss = alpha * pg_loss_ordo + (1.0 - alpha) * pg_loss_std
    loss = pg_loss - cfg.entropy_coef * ent_mean_all
    if vf_loss is not None and cfg.vf_coef > 0.0:
        loss = loss + float(cfg.vf_coef) * vf_loss
    if cfg.normalize_returns:
        zmid = 0.5 * (running_min + running_max)
        dscale = float(np.max(running_max - running_min))
        if dscale < 1e-08:
            dscale = 1.0
    else:
        zmid = None
        dscale = None
    info = {
        "L_mhat": Lm,
        "loss": float(loss.detach().cpu().item()),
        "pg_loss": float(pg_loss.detach().cpu().item()),
        "pg_loss_ordo": float(pg_loss_ordo.detach().cpu().item()),
        "pg_loss_std": None if pg_loss_std is None else float(pg_loss_std.detach().cpu().item()),
        "vf_loss": None if vf_loss is None else float(vf_loss.detach().cpu().item()),
        "mix_alpha": float(alpha),
        "std_w": None if cfg.std_w is None else np.asarray(cfg.std_w, dtype=np.float32).tolist(),
        "std_utility_family": str(cfg.std_utility_family) if cfg.std_w is not None else None,
        "action_temp": float(cfg.action_temp),
        "env_steps": int(env_steps),
        "env_steps_ref": int(env_steps_ref),
        "env_steps_cand": int(env_steps_cand),
        "entropy_mean": float(ent_mean_all.detach().cpu().item()),
        "normalize_d": dscale,
        "normalize_zmid": None if zmid is None else zmid.tolist(),
        "m_hat": m_hat.detach().cpu().numpy().tolist(),
        **m_info,
        "running_min": running_min.tolist(),
        "running_max": running_max.tolist(),
        "R_ref_mean": R_ref_np.mean(axis=0).tolist(),
        "R_cand_mean": R_cand_np.mean(axis=0).tolist(),
        "accept": Lm <= -cfg.eps_progress / 2.0,
        "loss_tensor": loss,
        "running_min_arr": running_min,
        "running_max_arr": running_max,
    }
    return info


def _scalarize_step_rewards(traj, w_t, reward_indices, device):
    with torch.no_grad():
        r_vec = np.stack(traj.rewards, axis=0).astype(np.float32)
        if reward_indices is not None:
            r_vec = r_vec[:, list(reward_indices)]
        r = torch.tensor(r_vec, dtype=torch.float32, device=device) @ w_t
        return r


def _build_ppo_batch(trajs, value_fn, cfg, device):
    assert cfg.std_w is not None
    w = np.asarray(cfg.std_w, dtype=np.float32)
    w_t = torch.tensor(w, dtype=torch.float32, device=device)
    obs_buf = []
    act_buf = []
    old_logp_buf = []
    adv_buf = []
    ret_buf = []
    traj_id_buf = []
    for (i, tr) in enumerate(trajs):
        T = len(tr.rewards)
        if T == 0:
            continue
        obs_t = torch.tensor(np.stack(tr.obs, axis=0), dtype=torch.float32, device=device)
        next_obs_t = torch.tensor(np.stack(tr.next_obs, axis=0), dtype=torch.float32, device=device)
        actions_t = torch.tensor(np.stack(tr.actions, axis=0), dtype=torch.float32, device=device)
        old_logp = torch.stack(tr.logprobs).detach()
        r = _scalarize_step_rewards(tr, w_t, cfg.reward_indices, device=device)
        with torch.no_grad():
            v = value_fn(obs_t)
            v_next = value_fn(next_obs_t)
        done = torch.zeros(T, dtype=torch.float32, device=device)
        if tr.terminated:
            done[-1] = 1.0
        v_next = v_next * (1.0 - done)
        delta = r + cfg.gamma * v_next - v
        adv = torch.zeros(T, dtype=torch.float32, device=device)
        gae = 0.0
        for t in reversed(range(T)):
            gae = delta[t] + cfg.gamma * cfg.gae_lambda * (1.0 - done[t]) * gae
            adv[t] = gae
        ret = adv + v
        obs_buf.append(obs_t)
        act_buf.append(actions_t)
        old_logp_buf.append(old_logp)
        adv_buf.append(adv.detach())
        ret_buf.append(ret.detach())
        traj_id_buf.append(torch.full((T,), i, dtype=torch.long, device=device))
    return {
        "obs": torch.cat(obs_buf, dim=0),
        "actions": torch.cat(act_buf, dim=0),
        "old_logp": torch.cat(old_logp_buf, dim=0),
        "adv_std": torch.cat(adv_buf, dim=0),
        "ret_std": torch.cat(ret_buf, dim=0),
        "traj_ids": torch.cat(traj_id_buf, dim=0),
    }


def _policy_logprob_and_entropy(policy, policy_kind, obs, actions):
    dist = policy(obs)
    if policy_kind == "discrete":
        a = actions.long().squeeze(-1)
        logp = dist.log_prob(a)
        ent = dist.entropy()
        return (logp, ent)
    logp = dist.log_prob(actions).sum(dim=-1)
    ent = dist.entropy().sum(dim=-1)
    return (logp, ent)


def ordo_step_ppo_update(
    env,
    policy,
    policy_kind,
    device,
    cfg,
    optimizer,
    ref_policy=None,
    ref_kind=None,
    running_min=None,
    running_max=None,
    value_fn=None,
):
    if value_fn is None:
        raise ValueError("PPO mode requires a value function (set --use-critic true)")
    if cfg.std_w is None:
        raise ValueError("PPO mode requires std_w (enable mixing with std weights)")
    if cfg.std_algo != "ppo":
        raise ValueError("Called ordo_step_ppo_update but cfg.std_algo != 'ppo'")
    ref_policy = policy if ref_policy is None else ref_policy
    ref_kind = policy_kind if ref_kind is None else ref_kind
    traj_cand = collect_trajectories(
        env,
        policy,
        policy_kind,
        device,
        cfg.gamma,
        cfg.max_steps,
        cfg.episodes_per_iter,
        action_temp=float(cfg.action_temp),
        track_grad=False,
    )
    env_steps_cand = sum((len(tr.rewards) for tr in traj_cand))
    env_steps_ref = 0
    traj_ref = None
    alpha = float(np.clip(cfg.mix_alpha, 0.0, 1.0))
    hk_vals = None
    Lm = None
    m_hat = None
    R_ref_np = None
    R_cand_np = returns_matrix(traj_cand, cfg.gamma)
    if cfg.reward_indices is not None:
        R_cand_np = R_cand_np[:, list(cfg.reward_indices)]
    if alpha > 1e-09:
        traj_ref = collect_trajectories(
            env,
            ref_policy,
            ref_kind,
            device,
            cfg.gamma,
            cfg.max_steps,
            cfg.episodes_per_iter,
            action_temp=1.0,
            track_grad=False,
        )
        env_steps_ref = sum((len(tr.rewards) for tr in traj_ref))
        R_ref_np = returns_matrix(traj_ref, cfg.gamma)
        if cfg.reward_indices is not None:
            R_ref_np = R_ref_np[:, list(cfg.reward_indices)]
        if running_min is None:
            running_min = np.min(np.vstack([R_ref_np, R_cand_np]), axis=0)
            running_max = np.max(np.vstack([R_ref_np, R_cand_np]), axis=0)
        else:
            running_min = np.minimum(running_min, np.min(np.vstack([R_ref_np, R_cand_np]), axis=0))
            running_max = np.maximum(running_max, np.max(np.vstack([R_ref_np, R_cand_np]), axis=0))
        if cfg.normalize_returns:
            R_ref_np_n = dpmorl_normalize_returns(R_ref_np, running_min, running_max)
            R_cand_np_n = dpmorl_normalize_returns(R_cand_np, running_min, running_max)
            m_lo = -cfg.m_grid_margin * np.ones(R_ref_np_n.shape[1], dtype=np.float32)
            m_hi = (1.0 + cfg.m_grid_margin) * np.ones(R_ref_np_n.shape[1], dtype=np.float32)
            R_ref = torch.tensor(R_ref_np_n, dtype=torch.float32, device=device)
            R_cand = torch.tensor(R_cand_np_n, dtype=torch.float32, device=device)
        else:
            (m_lo, m_hi) = estimate_M_bounds(running_min, running_max, cfg.m_grid_margin)
            R_ref = torch.tensor(R_ref_np, dtype=torch.float32, device=device)
            R_cand = torch.tensor(R_cand_np, dtype=torch.float32, device=device)
        (m_hat, m_info) = optimize_m_hat(R_cand, R_ref, m_lo, m_hi, cfg, device, return_info=True)
        Lm = float(
            L_hat(R_cand, R_ref, m_hat, cfg.k, cfg.smooth, cfg.smooth_param).detach().cpu().item()
        )
        hk_vals = hk(R_cand, m_hat, cfg.k, cfg.smooth, cfg.smooth_param).detach()
    else:
        m_info = {}
    batch = _build_ppo_batch(traj_cand, value_fn=value_fn, cfg=cfg, device=device)
    adv_std = batch["adv_std"]
    if hk_vals is not None:
        adv_ordo_tr = -(hk_vals - hk_vals.mean())
        adv_ordo_tr = adv_ordo_tr / (adv_ordo_tr.std(unbiased=False) + 1e-08)
        adv_ordo_tr = torch.clamp(adv_ordo_tr, -5.0, 5.0)
        adv_ordo = adv_ordo_tr[batch["traj_ids"]]
        adv_mix = (1.0 - alpha) * adv_std + alpha * adv_ordo
    else:
        adv_mix = adv_std
    adv_mix = (adv_mix - adv_mix.mean()) / (adv_mix.std(unbiased=False) + 1e-08)
    adv_mix = torch.clamp(adv_mix, -5.0, 5.0)
    obs = batch["obs"]
    actions = batch["actions"]
    old_logp = batch["old_logp"]
    ret = batch["ret_std"]
    n = obs.shape[0]
    mb = int(max(1, cfg.ppo_minibatch_size))
    clip = float(cfg.ppo_clip_range)
    policy_losses = []
    value_losses = []
    entropies = []
    approx_kls = []
    clipfracs = []
    for _ep in range(int(cfg.ppo_epochs)):
        idx = torch.randperm(n, device=device)
        for start in range(0, n, mb):
            j = idx[start : start + mb]
            (logp, ent) = _policy_logprob_and_entropy(policy, policy_kind, obs[j], actions[j])
            ratio = torch.exp(logp - old_logp[j])
            surr1 = ratio * adv_mix[j]
            surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv_mix[j]
            loss_pi = -torch.min(surr1, surr2).mean()
            v_pred = value_fn(obs[j])
            loss_v = 0.5 * F.mse_loss(v_pred, ret[j])
            loss_ent = -cfg.entropy_coef * ent.mean()
            loss = loss_pi + float(cfg.vf_coef) * loss_v + loss_ent
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), float(cfg.max_grad_norm))
            if value_fn is not None:
                torch.nn.utils.clip_grad_norm_(value_fn.parameters(), float(cfg.max_grad_norm))
            optimizer.step()
            with torch.no_grad():
                approx_kl = (old_logp[j] - logp).mean()
                clipfrac = (torch.abs(ratio - 1.0) > clip).float().mean()
                policy_losses.append(loss_pi.detach())
                value_losses.append(loss_v.detach())
                entropies.append(ent.mean().detach())
                approx_kls.append(approx_kl.detach())
                clipfracs.append(clipfrac.detach())
    info = {
        "L_mhat": float("nan") if Lm is None else float(Lm),
        "loss": float(torch.stack(policy_losses).mean().cpu().item())
        if policy_losses
        else float("nan"),
        "pg_loss": float(torch.stack(policy_losses).mean().cpu().item())
        if policy_losses
        else float("nan"),
        "pg_loss_ordo": float("nan"),
        "pg_loss_std": float(torch.stack(policy_losses).mean().cpu().item())
        if policy_losses
        else float("nan"),
        "vf_loss": float(torch.stack(value_losses).mean().cpu().item())
        if value_losses
        else float("nan"),
        "entropy_mean": float(torch.stack(entropies).mean().cpu().item()) if entropies else 0.0,
        "approx_kl": float(torch.stack(approx_kls).mean().cpu().item()) if approx_kls else None,
        "clipfrac": float(torch.stack(clipfracs).mean().cpu().item()) if clipfracs else None,
        "mix_alpha": float(alpha),
        "std_w": np.asarray(cfg.std_w, dtype=np.float32).tolist(),
        "action_temp": float(cfg.action_temp),
        "env_steps": int(env_steps_ref + env_steps_cand),
        "env_steps_ref": int(env_steps_ref),
        "env_steps_cand": int(env_steps_cand),
        "m_hat": None if m_hat is None else m_hat.detach().cpu().numpy().tolist(),
        **m_info,
        "running_min_arr": running_min,
        "running_max_arr": running_max,
        "R_ref_mean": None if R_ref_np is None else R_ref_np.mean(axis=0).tolist(),
        "R_cand_mean": R_cand_np.mean(axis=0).tolist(),
        "accept": None,
    }
    return info
