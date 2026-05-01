import argparse
import os
import sys
import time
from typing import Tuple
import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from sdmorl.utils import set_seed, now_ts, ensure_dir, json_dump
from sdmorl.progress_logger import ProgressLogger
from sdmorl.policy import build_policy
from sdmorl.ordo import OrdoConfig, ordo_step, ordo_step_ppo_update
from sdmorl.warmstart import WarmstartConfig, warmstart_step, seed_to_alpha_grid, alpha_to_weights
from sdmorl.utility import sample_monotone_rf_utility, calibrate_return_stats


def make_env(env_id):
    try:
        import mo_gymnasium as mo_gym

        return mo_gym.make(env_id)
    except Exception:
        import mo_gym

        return mo_gym.make(env_id)


def set_env_max_steps(env, max_steps):
    if int(max_steps) <= 0:
        return env
    if hasattr(env, "_max_episode_steps"):
        env._max_episode_steps = int(max_steps)
    if hasattr(env, "spec") and env.spec is not None:
        try:
            env.spec.max_episode_steps = int(max_steps)
        except Exception:
            pass
    return env


def resolve_max_steps(env, max_steps, fallback=1000):
    if int(max_steps) > 0:
        return int(max_steps)
    if hasattr(env, "_max_episode_steps"):
        try:
            v = int(getattr(env, "_max_episode_steps"))
            if v > 0:
                return v
        except Exception:
            pass
    if hasattr(env, "spec") and env.spec is not None:
        v = getattr(env.spec, "max_episode_steps", None)
        try:
            v = int(v) if v is not None else 0
            if v > 0:
                return v
        except Exception:
            pass
    return int(fallback)


def lin_schedule(step, start, end, T):
    T = max(1, int(T))
    frac = min(1.0, max(0.0, float(step) / float(T)))
    return float(start + (end - start) * frac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=str, default="mo-mountaincarcontinuous-v0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--total-iters",
        type=int,
        default=2000,
        help="number of optimization iterations",
    )
    ap.add_argument(
        "--total-timesteps",
        type=int,
        default=0,
        help="optional environment-step budget",
    )
    ap.add_argument(
        "--inner-iters",
        type=int,
        default=10,
        help="inner steps or reference refresh period",
    )
    ap.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="episode length cap",
    )
    ap.add_argument("--episodes-per-iter", type=int, default=8)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--mu", type=float, default=0.5, help="smoothing parameter")
    ap.add_argument(
        "--smooth", type=str, default="softplus", choices=["relu", "softplus", "gaussian"]
    )
    ap.add_argument("--eta", type=float, default=0.0003)
    ap.add_argument("--eps-progress", type=float, default=0.01)
    ap.add_argument("--independent-batches", type=str, default="false")
    ap.add_argument(
        "--inner-m-steps",
        type=int,
        default=50,
        help="inner maximization steps",
    )
    ap.add_argument(
        "--inner-m-lr",
        type=float,
        default=0.1,
        help="inner maximization learning rate",
    )
    ap.add_argument(
        "--use-critic",
        type=str,
        default="true",
        help="use a value-function baseline",
    )
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--vf-coef", type=float, default=0.5, help="value-loss coefficient")
    ap.add_argument(
        "--vf-eta", type=float, default=None, help="critic learning rate (default: --eta)"
    )
    ap.add_argument(
        "--std-algo",
        type=str,
        default="ac",
        choices=["ac", "ppo"],
        help="standard objective optimizer",
    )
    ap.add_argument("--ppo-clip-range", type=float, default=0.2)
    ap.add_argument("--ppo-epochs", type=int, default=10)
    ap.add_argument("--ppo-minibatch-size", type=int, default=64)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument(
        "--reward-indices",
        type=str,
        default="0,1",
        help="comma-separated objective indices",
    )
    ap.add_argument(
        "--normalize-returns",
        action="store_true",
        help="normalize returns to [0,1]^K",
    )
    ap.add_argument(
        "--entropy-coef",
        type=float,
        default=0.0,
        help="constant entropy coefficient",
    )
    ap.add_argument(
        "--entropy-start",
        type=float,
        default=None,
        help="starting entropy coefficient",
    )
    ap.add_argument(
        "--entropy-end",
        type=float,
        default=0.0,
        help="final entropy coefficient",
    )
    ap.add_argument(
        "--entropy-anneal-iters",
        type=int,
        default=0,
        help="entropy annealing horizon",
    )
    ap.add_argument(
        "--action-temp-start",
        type=float,
        default=None,
        help="starting action-sampling temperature",
    )
    ap.add_argument("--action-temp-end", type=float, default=1.0, help="final action temperature")
    ap.add_argument(
        "--action-temp-anneal-iters",
        type=int,
        default=0,
        help="action-temperature annealing horizon",
    )
    ap.add_argument(
        "--action-temp-by-alpha",
        action="store_true",
        help="scale action temperature by alpha",
    )
    ap.add_argument(
        "--mix-alpha-start",
        type=float,
        default=None,
        help="starting ORDO mixing weight",
    )
    ap.add_argument(
        "--mix-alpha-end",
        type=float,
        default=None,
        help="final ORDO mixing weight",
    )
    ap.add_argument(
        "--mix-alpha-anneal-iters",
        type=int,
        default=0,
        help="mixing-weight annealing horizon",
    )
    ap.add_argument(
        "--mix-alpha-delay",
        type=int,
        default=0,
        help="delay before the ORDO mixing schedule starts",
    )
    ap.add_argument(
        "--mix-alpha-shape",
        type=str,
        default="none",
        choices=["none", "midpeak", "treasure", "time"],
        help="optional alpha-dependent mixing shape",
    )
    ap.add_argument(
        "--mix-alpha-shape-exp",
        type=float,
        default=1.0,
        help="exponent for the mixing shape",
    )
    ap.add_argument(
        "--mix-lambda-start",
        type=float,
        default=0.0,
        help="legacy starting lambda",
    )
    ap.add_argument(
        "--mix-lambda-end",
        type=float,
        default=0.0,
        help="legacy final lambda",
    )
    ap.add_argument(
        "--mix-lambda-anneal-iters",
        type=int,
        default=0,
        help="legacy lambda annealing horizon",
    )
    ap.add_argument(
        "--std-weight-mode",
        type=str,
        default="grid",
        choices=["grid", "fixed", "random"],
        help="How to choose the scalarization weights w for the standard PG term.",
    )
    ap.add_argument(
        "--std-alpha",
        type=float,
        default=0.5,
        help="alpha for fixed 2D weights",
    )
    ap.add_argument("--std-alpha-min", type=float, default=0.0)
    ap.add_argument("--std-alpha-max", type=float, default=1.0)
    ap.add_argument("--std-grid-n", type=int, default=20)
    ap.add_argument(
        "--std-grid-include-zero",
        action="store_true",
        help="include alpha=0 in the grid",
    )
    ap.add_argument(
        "--std-grid-warp",
        type=float,
        default=1.0,
        help="power warp for the alpha grid",
    )
    ap.add_argument(
        "--std-utility-family",
        type=str,
        default="linear",
        choices=["linear", "monotone_rf"],
        help="utility family for the standard objective",
    )
    ap.add_argument(
        "--std-utility-features",
        type=int,
        default=32,
        help="Number of monotone random features J (monotone_rf only).",
    )
    ap.add_argument(
        "--std-utility-kappa",
        type=float,
        default=10.0,
        help="Dirichlet concentration for monotone_rf",
    )
    ap.add_argument(
        "--std-utility-calib-episodes",
        type=int,
        default=512,
        help="calibration episodes for monotone_rf",
    )
    ap.add_argument(
        "--std-utility-beta-min",
        type=float,
        default=0.5,
        help="Min softplus sharpness beta (monotone_rf only).",
    )
    ap.add_argument(
        "--std-utility-beta-max",
        type=float,
        default=5.0,
        help="Max softplus sharpness beta (monotone_rf only).",
    )
    ap.add_argument(
        "--std-utility-c-std",
        type=float,
        default=1.0,
        help="Std dev for feature offsets c (monotone_rf only).",
    )
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--log-every", type=int, default=10, help="log/print every N iterations")
    ap.add_argument(
        "--print-every",
        type=int,
        default=None,
        help="override console print frequency (default: log-every)",
    )
    ap.add_argument("--save-every", type=int, default=50, help="checkpoint every N iterations")
    ap.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="checkpoint to resume from",
    )
    ap.add_argument(
        "--stop-at",
        type=int,
        default=0,
        help="optional early stop",
    )
    ap.add_argument(
        "--accept-gating",
        action="store_true",
        help="enable accept/revert gating",
    )
    ap.add_argument(
        "--ref-refresh",
        type=int,
        default=0,
        help="reference refresh period",
    )
    ap.add_argument(
        "--warmstart-iters",
        type=int,
        default=0,
        help="warm-start iterations on scalarized objective (0 disables)",
    )
    ap.add_argument("--warmstart-episodes-per-iter", type=int, default=256)
    ap.add_argument("--warmstart-eta", type=float, default=0.001)
    ap.add_argument("--warmstart-entropy-coef", type=float, default=0.01)
    ap.add_argument(
        "--warmstart-weight-mode", type=str, default="grid", choices=["grid", "fixed", "random"]
    )
    ap.add_argument("--warmstart-alpha", type=float, default=0.9, help="alpha for fixed mode")
    ap.add_argument("--warmstart-alpha-min", type=float, default=0.7)
    ap.add_argument("--warmstart-alpha-max", type=float, default=0.99)
    ap.add_argument("--warmstart-grid-n", type=int, default=20)
    args = ap.parse_args()
    reward_indices = tuple((int(x) for x in args.reward_indices.split(",") if x.strip() != ""))
    use_timesteps = int(getattr(args, "total_timesteps", 0)) > 0
    total_timesteps = int(getattr(args, "total_timesteps", 0))

    def _default_horizon(anneal_arg):
        if int(anneal_arg) > 0:
            return int(anneal_arg)
        return int(total_timesteps) if use_timesteps else int(args.total_iters)

    def entropy_coef_at(progress):
        if args.entropy_start is None:
            return float(args.entropy_coef)
        T = _default_horizon(int(args.entropy_anneal_iters))
        return lin_schedule(progress, float(args.entropy_start), float(args.entropy_end), T)

    def action_temp_at(progress, alpha0):
        if args.action_temp_start is None:
            base = 1.0
        else:
            T = _default_horizon(int(args.action_temp_anneal_iters))
            base = lin_schedule(
                progress, float(args.action_temp_start), float(args.action_temp_end), T
            )
        base = float(max(1e-06, base))
        if args.action_temp_by_alpha and alpha0 is not None:
            base = 1.0 + (base - 1.0) * float(np.clip(alpha0, 0.0, 1.0))
        return float(max(1e-06, base))

    def _alpha_to_lambda(alpha):
        if alpha <= 0.0:
            return 0.0
        if alpha >= 1.0:
            return 1000000000.0
        lam = float(alpha / (1.0 - alpha))
        return float(min(lam, 1000000000.0))

    def mix_alpha_lambda_at(progress):
        if args.mix_alpha_start is not None or args.mix_alpha_end is not None:
            a0 = float(args.mix_alpha_start) if args.mix_alpha_start is not None else 1.0
            a1 = float(args.mix_alpha_end) if args.mix_alpha_end is not None else a0
            T = _default_horizon(int(args.mix_alpha_anneal_iters))
        else:
            lam0 = float(args.mix_lambda_start)
            lam1 = float(args.mix_lambda_end)
            if lam0 == 0.0 and lam1 == 0.0:
                return (1.0, 0.0)
            a0 = 0.0 if lam0 <= 0.0 else lam0 / (lam0 + 1.0)
            a1 = 0.0 if lam1 <= 0.0 else lam1 / (lam1 + 1.0)
            T = _default_horizon(int(args.mix_lambda_anneal_iters))
        delay = int(getattr(args, "mix_alpha_delay", 0))
        if delay > 0 and progress < delay:
            alpha_base = float(a0)
        else:
            prog2 = int(progress - delay) if delay > 0 else int(progress)
            alpha_base = lin_schedule(prog2, a0, a1, T)
        alpha_base = float(np.clip(alpha_base, 0.0, 1.0))
        lam_base = _alpha_to_lambda(alpha_base)
        return (float(alpha_base), float(lam_base))

    def apply_mix_alpha_shape(alpha_base, alpha0):
        if args.mix_alpha_shape == "none" or alpha0 is None:
            return float(np.clip(alpha_base, 0.0, 1.0))
        a = float(np.clip(alpha0, 0.0, 1.0))
        if args.mix_alpha_shape == "midpeak":
            f = 4.0 * a * (1.0 - a)
        elif args.mix_alpha_shape == "treasure":
            f = a
        elif args.mix_alpha_shape == "time":
            f = 1.0 - a
        else:
            f = 1.0
        f = float(np.clip(f, 0.0, 1.0))
        expv = float(args.mix_alpha_shape_exp)
        if expv != 1.0:
            f = float(f**expv)
        return float(np.clip(alpha_base * f, 0.0, 1.0))

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_env(args.env)
    args.max_steps = resolve_max_steps(env, args.max_steps)
    env = set_env_max_steps(env, args.max_steps)
    (policy, policy_kind) = build_policy(env)
    policy.to(device)
    use_critic = str(args.use_critic).lower() == "true"
    value_fn = None
    if use_critic:
        from sdmorl.value import build_value

        value_fn = build_value(env)
        value_fn.to(device)
    (ref_policy, ref_kind) = build_policy(env)
    ref_policy.to(device)
    run_dir = args.run_dir or os.path.join("runs", f"ordo_{args.env}_{now_ts()}_seed{args.seed}")
    ensure_dir(run_dir)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    ensure_dir(ckpt_dir)
    print_every = args.print_every if args.print_every is not None else max(1, args.log_every)
    logger = ProgressLogger(run_dir, print_every=print_every, header_every=50)
    std_alpha_val = None
    std_w_val = None
    if args.mix_alpha_start is not None or args.mix_alpha_end is not None:
        a0_preview = float(args.mix_alpha_start) if args.mix_alpha_start is not None else 1.0
        a1_preview = float(args.mix_alpha_end) if args.mix_alpha_end is not None else a0_preview
        mixing_enabled = not (a0_preview == 1.0 and a1_preview == 1.0)
    else:
        mixing_enabled = not (
            float(args.mix_lambda_start) == 0.0 and float(args.mix_lambda_end) == 0.0
        )
    if mixing_enabled:
        if len(reward_indices) != 2:
            raise ValueError("mixed update supports only two objectives")
        if args.std_weight_mode == "grid":
            n = int(args.std_grid_n)
            idx = int(args.seed) % max(1, n)
            a_min = float(args.std_alpha_min)
            a_max = float(args.std_alpha_max)
            warp = float(getattr(args, "std_grid_warp", 1.0))
            include_zero = bool(getattr(args, "std_grid_include_zero", False))

            def _warp(frac):
                frac = float(np.clip(frac, 0.0, 1.0))
                if warp <= 1.0:
                    return frac
                return 1.0 - (1.0 - frac) ** warp

            if n <= 1:
                std_alpha = a_max
            elif include_zero and n >= 2:
                if idx == 0:
                    std_alpha = 0.0
                else:
                    denom = max(1, n - 2)
                    frac = (idx - 1) / denom
                    frac = _warp(frac)
                    std_alpha = a_min + (a_max - a_min) * frac
            else:
                denom = max(1, n - 1)
                frac = idx / denom
                frac = _warp(frac)
                std_alpha = a_min + (a_max - a_min) * frac
        elif args.std_weight_mode == "random":
            rng = np.random.RandomState(args.seed)
            std_alpha = float(rng.uniform(args.std_alpha_min, args.std_alpha_max))
        else:
            std_alpha = float(args.std_alpha)
        std_w = alpha_to_weights(std_alpha)
        std_alpha_val = float(std_alpha)
        std_w_val = std_w.tolist()
    std_utility_family = str(args.std_utility_family)
    std_utility_params = None
    std_utility_calib = None
    if mixing_enabled and std_utility_family == "monotone_rf":
        calib_episodes = int(getattr(args, "std_utility_calib_episodes", 0))
        if calib_episodes > 0:
            (mean, std) = calibrate_return_stats(
                env_id=args.env,
                seed=int(args.seed) + 12345,
                episodes=calib_episodes,
                max_steps=int(args.max_steps),
                gamma=0.9999,
                reward_indices=reward_indices,
            )
            std_utility_calib = {
                "mean": mean.tolist(),
                "std": std.tolist(),
                "episodes": int(calib_episodes),
            }
        else:
            std_utility_calib = {
                "mean": [0.0] * len(reward_indices),
                "std": [1.0] * len(reward_indices),
                "episodes": 0,
            }
        std_utility_params = sample_monotone_rf_utility(
            seed=int(args.seed),
            base_w=np.asarray(std_w_val, dtype=np.float32),
            J=int(args.std_utility_features),
            kappa=float(args.std_utility_kappa),
            beta_min=float(args.std_utility_beta_min),
            beta_max=float(args.std_utility_beta_max),
            c_std=float(args.std_utility_c_std),
        )
    if not mixing_enabled:
        mix_desc = "off"
    elif args.mix_alpha_start is not None or args.mix_alpha_end is not None:
        a0 = float(args.mix_alpha_start) if args.mix_alpha_start is not None else 1.0
        a1 = float(args.mix_alpha_end) if args.mix_alpha_end is not None else a0
        Tmix = _default_horizon(int(args.mix_alpha_anneal_iters))
        mix_desc = f"alpha({a0}->{a1} over {Tmix}); w={std_w_val}"
    else:
        lam0 = float(args.mix_lambda_start)
        lam1 = float(args.mix_lambda_end)
        a0 = 0.0 if lam0 <= 0.0 else lam0 / (lam0 + 1.0)
        a1 = 0.0 if lam1 <= 0.0 else lam1 / (lam1 + 1.0)
        Tmix = _default_horizon(int(args.mix_lambda_anneal_iters))
        mix_desc = f"alpha({a0}->{a1} over {Tmix}) [from lambda({lam0}->{lam1})]; w={std_w_val}"
    sched_unit = "timesteps" if use_timesteps else "iters"
    ent_T = _default_horizon(int(args.entropy_anneal_iters))
    temp_T = _default_horizon(int(args.action_temp_anneal_iters))
    msg = (
        f"[ORDO] env={args.env} device={device} seed={args.seed} "
        f"iters={args.total_iters} timesteps={args.total_timesteps} "
        f"inner={args.inner_iters} epi/iter={args.episodes_per_iter} "
        f"k={args.k} smooth={args.smooth} mu={args.mu} eps={args.eps_progress} "
        f"eta={args.eta} sched_unit={sched_unit} "
        f"ent_sched=({args.entropy_start}->{args.entropy_end} over {ent_T}) "
        f"act_temp=({args.action_temp_start}->{args.action_temp_end} over {temp_T}) "
        f"mix={mix_desc} std_utility={std_utility_family} "
        f"norm={bool(args.normalize_returns)} "
        f"accept_gating={bool(args.accept_gating)}"
    )
    print(msg, flush=True)
    vf_eta = float(args.eta) if args.vf_eta is None else float(args.vf_eta)
    if value_fn is None:
        opt = torch.optim.Adam(policy.parameters(), lr=args.eta)
    else:
        opt = torch.optim.Adam(
            [
                {"params": policy.parameters(), "lr": float(args.eta)},
                {"params": value_fn.parameters(), "lr": vf_eta},
            ]
        )
    cfg = OrdoConfig(
        max_steps=args.max_steps,
        k=args.k,
        reward_indices=reward_indices,
        normalize_returns=bool(args.normalize_returns),
        episodes_per_iter=args.episodes_per_iter,
        eta=args.eta,
        eps_progress=args.eps_progress,
        smooth=args.smooth,
        smooth_param=args.mu,
        independent_batches=args.independent_batches.lower() == "true",
        entropy_coef=float(args.entropy_coef),
        action_temp=1.0,
        std_w=None if not mixing_enabled else np.asarray(std_w_val, dtype=np.float32),
        std_utility_family=std_utility_family,
        std_utility_params=std_utility_params,
        std_utility_calib=std_utility_calib,
        mix_alpha=1.0,
        inner_m_steps=args.inner_m_steps,
        inner_m_lr=args.inner_m_lr,
        use_gae=bool(value_fn is not None) and use_critic,
        gae_lambda=float(args.gae_lambda),
        vf_coef=float(args.vf_coef),
        std_algo=str(args.std_algo).lower(),
        ppo_clip_range=float(args.ppo_clip_range),
        ppo_epochs=int(args.ppo_epochs),
        ppo_minibatch_size=int(args.ppo_minibatch_size),
        max_grad_norm=float(args.max_grad_norm),
    )
    start_step = 0
    running_min = None
    running_max = None
    env_steps = 0

    def _save_ckpt(path, step, running_min=None, running_max=None):
        state = {
            "policy": policy.state_dict(),
            "ref_policy": ref_policy.state_dict(),
            "value": value_fn.state_dict() if value_fn is not None else None,
            "opt": opt.state_dict(),
            "step": int(step),
            "env_steps": int(env_steps),
            "running_min": running_min.tolist() if running_min is not None else None,
            "running_max": running_max.tolist() if running_max is not None else None,
        }
        torch.save(state, path)

    if args.resume_from is not None:
        resume_path = args.resume_from
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"--resume-from not found: {resume_path}")
        state = torch.load(resume_path, map_location=device)
        if isinstance(state, dict) and "policy" in state:
            policy.load_state_dict(state["policy"])
            if state.get("ref_policy") is not None:
                ref_policy.load_state_dict(state["ref_policy"])
            else:
                ref_policy.load_state_dict(state["policy"])
            if value_fn is not None and state.get("value") is not None:
                try:
                    value_fn.load_state_dict(state["value"])
                except Exception:
                    pass
            if state.get("opt") is not None:
                try:
                    opt.load_state_dict(state["opt"])
                except Exception:
                    pass
            start_step = int(state.get("step", 0))
            env_steps = int(state.get("env_steps", 0))
            if state.get("running_min") is not None:
                running_min = np.array(state["running_min"], dtype=np.float32)
            if state.get("running_max") is not None:
                running_max = np.array(state["running_max"], dtype=np.float32)
        else:
            policy.load_state_dict(state)
            ref_policy.load_state_dict(state)
            start_step = 0
        print(
            f"[RESUME] from={resume_path} start_step={start_step} env_steps={env_steps}", flush=True
        )
        if start_step > 0 and args.warmstart_iters > 0:
            print("[RESUME] warm-start disabled on resume (continuing ORDO training)", flush=True)
    alpha_val = None
    w_val = None
    if args.warmstart_iters > 0 and start_step == 0:
        if args.warmstart_weight_mode == "grid":
            alpha = seed_to_alpha_grid(
                args.seed, args.warmstart_grid_n, args.warmstart_alpha_min, args.warmstart_alpha_max
            )
        elif args.warmstart_weight_mode == "random":
            rng = np.random.RandomState(args.seed)
            alpha = float(rng.uniform(args.warmstart_alpha_min, args.warmstart_alpha_max))
        else:
            alpha = float(args.warmstart_alpha)
        w = alpha_to_weights(alpha)
        alpha_val = float(alpha)
        w_val = w.tolist()
        msg = (
            f"[WARM] iters={args.warmstart_iters} alpha={alpha_val:.4f} "
            f"w={w_val} epi/iter={args.warmstart_episodes_per_iter} "
            f"eta={args.warmstart_eta} ent={args.warmstart_entropy_coef}"
        )
        print(msg, flush=True)
        warm_cfg = WarmstartConfig(
            gamma=cfg.gamma,
            max_steps=args.max_steps,
            episodes_per_iter=args.warmstart_episodes_per_iter,
            eta=args.warmstart_eta,
            entropy_coef=args.warmstart_entropy_coef,
            reward_indices=reward_indices,
            weights=w,
        )
        warm_opt = torch.optim.Adam(policy.parameters(), lr=args.warmstart_eta)
        for wit in range(args.warmstart_iters):
            info_w = warmstart_step(env, policy, policy_kind, device, warm_cfg)
            env_steps += int(info_w.get("env_steps", 0))
            loss_tensor = info_w.pop("loss_tensor")
            warm_opt.zero_grad()
            loss_tensor.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            warm_opt.step()
            if wit == 0 or (wit + 1) % max(1, args.log_every) == 0:
                rec = {
                    "time": time.time(),
                    "phase": "warm",
                    "env": args.env,
                    "seed": args.seed,
                    "outer_iter": int(wit),
                    "inner_iter": -1,
                    "alpha": alpha_val,
                    "accept": None,
                    "L_mhat": None,
                    "loss": float(info_w.get("loss", float("nan"))),
                    "pg_loss": None,
                    "pg_loss_ordo": None,
                    "pg_loss_std": None,
                    "vf_loss": None,
                    "entropy_mean": float(info_w.get("entropy_mean", float("nan"))),
                    "entropy_coef": None,
                    "mix_alpha": None,
                    "mix_lambda": None,
                    "std_w": std_w_val,
                    "warm_scalar_w": w_val,
                    "R_mean": info_w.get("R_mean", None),
                    "R_ref_mean": None,
                    "R_cand_mean": None,
                    "env_steps": int(env_steps),
                    "env_steps_delta": int(info_w.get("env_steps", 0)),
                    "m_hat": None,
                }
                logger.log(rec)
        torch.save(
            {"policy": policy.state_dict(), "alpha": alpha_val, "weights": w_val},
            os.path.join(ckpt_dir, "warmstart.pt"),
        )
        vf_eta = float(args.eta) if args.vf_eta is None else float(args.vf_eta)
        if value_fn is None:
            opt = torch.optim.Adam(policy.parameters(), lr=args.eta)
        else:
            opt = torch.optim.Adam(
                [
                    {"params": policy.parameters(), "lr": float(args.eta)},
                    {"params": value_fn.parameters(), "lr": vf_eta},
                ]
            )
    if running_min is None or running_max is None:
        (running_min, running_max) = (None, None)
    if running_min is None or running_max is None:
        (running_min, running_max) = (None, None)
    final_step = int(start_step)
    if args.accept_gating:
        if str(args.std_algo).lower() == "ppo":
            raise ValueError(
                "--std-algo ppo is only supported in no-gating mode (do not pass --accept-gating)."
            )
        if not use_timesteps:
            outer_stop = int(args.stop_at) if int(args.stop_at) > 0 else int(args.total_iters)
            outer_stop = min(outer_stop, int(args.total_iters))
            it = int(start_step)
            it_end = int(outer_stop)
            it_iter = range(it, it_end)
        else:
            outer_cap = int(args.stop_at) if int(args.stop_at) > 0 else None
            it_iter = None
        if not use_timesteps:
            for it in it_iter:
                progress = int(env_steps) if use_timesteps else int(it)
                cfg.entropy_coef = entropy_coef_at(progress)
                cfg.action_temp = action_temp_at(
                    progress, std_alpha_val if mixing_enabled else None
                )
                (alpha_mix, lam_mix) = mix_alpha_lambda_at(progress)
                cfg.mix_alpha = alpha_mix
                ref_policy.load_state_dict(policy.state_dict())
                accepted = False
                for inner in range(int(args.inner_iters)):
                    info = ordo_step(
                        env,
                        policy,
                        policy_kind,
                        device,
                        cfg,
                        ref_policy=ref_policy,
                        ref_kind=ref_kind,
                        running_min=running_min,
                        running_max=running_max,
                        value_fn=value_fn,
                    )
                    env_steps += int(info.get("env_steps", 0))
                    loss_tensor = info.pop("loss_tensor")
                    running_min = np.array(info.pop("running_min_arr"))
                    running_max = np.array(info.pop("running_max_arr"))
                    opt.zero_grad()
                    loss_tensor.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
                    opt.step()
                    do_log = inner == 0 or (inner + 1) % max(1, args.log_every) == 0
                    if do_log:
                        rec = {
                            "time": time.time(),
                            "phase": "ordo",
                            "env": args.env,
                            "seed": args.seed,
                            "outer_iter": int(it),
                            "inner_iter": int(inner),
                            "alpha": alpha_val,
                            "accept": bool(info.get("accept", False)),
                            "L_mhat": float(info.get("L_mhat", float("nan"))),
                            "loss": float(info.get("loss", float("nan"))),
                            "pg_loss": float(info.get("pg_loss", float("nan"))),
                            "pg_loss_ordo": float(info.get("pg_loss_ordo", float("nan"))),
                            "pg_loss_std": info.get("pg_loss_std", None),
                            "vf_loss": info.get("vf_loss", None),
                            "entropy_mean": float(info.get("entropy_mean", float("nan"))),
                            "entropy_coef": float(cfg.entropy_coef),
                            "action_temp": float(cfg.action_temp),
                            "mix_alpha": float(alpha_mix) if mixing_enabled else None,
                            "mix_lambda": float(lam_mix) if mixing_enabled else None,
                            "std_w": std_w_val,
                            "std_utility_family": std_utility_family if mixing_enabled else None,
                            "warm_scalar_w": w_val,
                            "R_mean": info.get("R_cand_mean", None),
                            "R_ref_mean": info.get("R_ref_mean", None),
                            "R_cand_mean": info.get("R_cand_mean", None),
                            "m_hat": info.get("m_hat", None),
                            "m_obj_initial": info.get("m_obj_initial", None),
                            "m_obj_final": info.get("m_obj_final", None),
                            "m_obj_best_seen": info.get("m_obj_best_seen", None),
                            "m_obj_improvement": info.get("m_obj_improvement", None),
                            "m_grad_norm": info.get("m_grad_norm", None),
                            "m_steps": info.get("m_steps", None),
                            "m_time_sec": info.get("m_time_sec", None),
                            "env_steps": int(env_steps),
                            "env_steps_delta": int(info.get("env_steps", 0)),
                        }
                        logger.log(rec)
                    if info.get("accept"):
                        accepted = True
                        break
                if not accepted:
                    print(f"[ORDO] stopping early at outer={it + 1}", flush=True)
                    final_step = int(it + 1)
                    break
                final_step = int(it + 1)
                if (it + 1) % max(1, args.save_every) == 0:
                    _save_ckpt(
                        os.path.join(ckpt_dir, f"iter_{it + 1}.pt"),
                        step=int(it + 1),
                        running_min=running_min,
                        running_max=running_max,
                    )
                    _save_ckpt(
                        os.path.join(ckpt_dir, "last.pt"),
                        step=int(it + 1),
                        running_min=running_min,
                        running_max=running_max,
                    )
        else:
            it = int(start_step)
            outer_cap = int(args.stop_at) if int(args.stop_at) > 0 else None
            stop_flag = False
            while True:
                if total_timesteps > 0 and int(env_steps) >= int(total_timesteps):
                    break
                if outer_cap is not None and it >= int(outer_cap):
                    break
                ref_policy.load_state_dict(policy.state_dict())
                accepted = False
                for inner in range(int(args.inner_iters)):
                    progress = int(env_steps) if use_timesteps else int(it)
                    cfg.entropy_coef = entropy_coef_at(progress)
                    cfg.action_temp = action_temp_at(
                        progress, std_alpha_val if mixing_enabled else None
                    )
                    (alpha_mix, lam_mix) = mix_alpha_lambda_at(progress)
                    cfg.mix_alpha = alpha_mix
                    info = ordo_step(
                        env,
                        policy,
                        policy_kind,
                        device,
                        cfg,
                        ref_policy=ref_policy,
                        ref_kind=ref_kind,
                        running_min=running_min,
                        running_max=running_max,
                        value_fn=value_fn,
                    )
                    env_steps += int(info.get("env_steps", 0))
                    loss_tensor = info.pop("loss_tensor")
                    running_min = np.array(info.pop("running_min_arr"))
                    running_max = np.array(info.pop("running_max_arr"))
                    opt.zero_grad()
                    loss_tensor.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
                    opt.step()
                    do_log = inner == 0 or (inner + 1) % max(1, args.log_every) == 0
                    if do_log:
                        rec = {
                            "time": time.time(),
                            "phase": "ordo",
                            "env": args.env,
                            "seed": args.seed,
                            "outer_iter": int(it),
                            "inner_iter": int(inner),
                            "alpha": alpha_val,
                            "accept": bool(info.get("accept", False)),
                            "L_mhat": float(info.get("L_mhat", float("nan"))),
                            "loss": float(info.get("loss", float("nan"))),
                            "pg_loss": float(info.get("pg_loss", float("nan"))),
                            "pg_loss_ordo": float(info.get("pg_loss_ordo", float("nan"))),
                            "pg_loss_std": info.get("pg_loss_std", None),
                            "vf_loss": info.get("vf_loss", None),
                            "entropy_mean": float(info.get("entropy_mean", float("nan"))),
                            "entropy_coef": float(cfg.entropy_coef),
                            "action_temp": float(cfg.action_temp),
                            "mix_alpha": float(alpha_mix) if mixing_enabled else None,
                            "mix_lambda": float(lam_mix) if mixing_enabled else None,
                            "std_w": std_w_val,
                            "std_utility_family": std_utility_family if mixing_enabled else None,
                            "warm_scalar_w": w_val,
                            "R_mean": info.get("R_cand_mean", None),
                            "R_ref_mean": info.get("R_ref_mean", None),
                            "R_cand_mean": info.get("R_cand_mean", None),
                            "m_hat": info.get("m_hat", None),
                            "m_obj_initial": info.get("m_obj_initial", None),
                            "m_obj_final": info.get("m_obj_final", None),
                            "m_obj_best_seen": info.get("m_obj_best_seen", None),
                            "m_obj_improvement": info.get("m_obj_improvement", None),
                            "m_grad_norm": info.get("m_grad_norm", None),
                            "m_steps": info.get("m_steps", None),
                            "m_time_sec": info.get("m_time_sec", None),
                            "env_steps": int(env_steps),
                            "env_steps_delta": int(info.get("env_steps", 0)),
                        }
                        logger.log(rec)
                    if info.get("accept"):
                        accepted = True
                        break
                    if total_timesteps > 0 and int(env_steps) >= int(total_timesteps):
                        stop_flag = True
                        break
                final_step = int(it + 1)
                if (it + 1) % max(1, args.save_every) == 0:
                    _save_ckpt(
                        os.path.join(ckpt_dir, f"iter_{it + 1}.pt"),
                        step=int(it + 1),
                        running_min=running_min,
                        running_max=running_max,
                    )
                    _save_ckpt(
                        os.path.join(ckpt_dir, "last.pt"),
                        step=int(it + 1),
                        running_min=running_min,
                        running_max=running_max,
                    )
                if stop_flag:
                    print(
                        f"[ORDO] stopping at env_steps={env_steps} (budget={total_timesteps})",
                        flush=True,
                    )
                    break
                if not accepted:
                    print(f"[ORDO] stopping early at outer={it + 1}", flush=True)
                    break
                it += 1
    else:
        ref_refresh = int(args.ref_refresh) if int(args.ref_refresh) != 0 else int(args.inner_iters)
        ref_refresh = max(1, ref_refresh)
        if int(start_step) == 0:
            ref_policy.load_state_dict(policy.state_dict())
        if not use_timesteps:
            stop_at = int(args.stop_at) if int(args.stop_at) > 0 else int(args.total_iters)
            stop_at = min(stop_at, int(args.total_iters))
            step_iter = range(int(start_step), int(stop_at))
        else:
            step_cap = int(args.stop_at) if int(args.stop_at) > 0 else None
            step_iter = None
        if not use_timesteps:
            for step in step_iter:
                if step % ref_refresh == 0 and step != 0:
                    ref_policy.load_state_dict(policy.state_dict())
                progress = int(env_steps) if use_timesteps else int(step)
                cfg.entropy_coef = entropy_coef_at(progress)
                cfg.action_temp = action_temp_at(
                    progress, std_alpha_val if mixing_enabled else None
                )
                (alpha_mix, lam_mix) = mix_alpha_lambda_at(progress)
                cfg.mix_alpha = alpha_mix
                if cfg.std_algo == "ppo":
                    info = ordo_step_ppo_update(
                        env,
                        policy,
                        policy_kind,
                        device,
                        cfg,
                        optimizer=opt,
                        ref_policy=ref_policy,
                        ref_kind=ref_kind,
                        running_min=running_min,
                        running_max=running_max,
                        value_fn=value_fn,
                    )
                else:
                    info = ordo_step(
                        env,
                        policy,
                        policy_kind,
                        device,
                        cfg,
                        ref_policy=ref_policy,
                        ref_kind=ref_kind,
                        running_min=running_min,
                        running_max=running_max,
                        value_fn=value_fn,
                    )
                env_steps += int(info.get("env_steps", 0))
                rm = info.pop("running_min_arr", None)
                rM = info.pop("running_max_arr", None)
                if rm is not None:
                    running_min = np.array(rm, dtype=np.float32)
                if rM is not None:
                    running_max = np.array(rM, dtype=np.float32)
                if cfg.std_algo != "ppo":
                    loss_tensor = info.pop("loss_tensor")
                    opt.zero_grad()
                    loss_tensor.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
                    opt.step()
                do_log = step == 0 or (step + 1) % max(1, args.log_every) == 0
                if do_log:
                    outer_iter = step // ref_refresh
                    inner_iter = step % ref_refresh
                    rec = {
                        "time": time.time(),
                        "phase": "ordo",
                        "env": args.env,
                        "seed": args.seed,
                        "outer_iter": int(outer_iter),
                        "inner_iter": int(inner_iter),
                        "alpha": alpha_val,
                        "accept": bool(info.get("accept", False)),
                        "L_mhat": float(info.get("L_mhat", float("nan"))),
                        "loss": float(info.get("loss", float("nan"))),
                        "pg_loss": float(info.get("pg_loss", float("nan"))),
                        "pg_loss_ordo": float(info.get("pg_loss_ordo", float("nan"))),
                        "pg_loss_std": info.get("pg_loss_std", None),
                        "vf_loss": info.get("vf_loss", None),
                        "entropy_mean": float(info.get("entropy_mean", float("nan"))),
                        "entropy_coef": float(cfg.entropy_coef),
                        "action_temp": float(cfg.action_temp),
                        "mix_alpha": float(alpha_mix) if mixing_enabled else None,
                        "mix_lambda": float(lam_mix) if mixing_enabled else None,
                        "std_w": std_w_val,
                        "std_utility_family": std_utility_family if mixing_enabled else None,
                        "warm_scalar_w": w_val,
                        "R_mean": info.get("R_cand_mean", None),
                        "R_ref_mean": info.get("R_ref_mean", None),
                        "R_cand_mean": info.get("R_cand_mean", None),
                        "m_hat": info.get("m_hat", None),
                        "m_obj_initial": info.get("m_obj_initial", None),
                        "m_obj_final": info.get("m_obj_final", None),
                        "m_obj_best_seen": info.get("m_obj_best_seen", None),
                        "m_obj_improvement": info.get("m_obj_improvement", None),
                        "m_grad_norm": info.get("m_grad_norm", None),
                        "m_steps": info.get("m_steps", None),
                        "m_time_sec": info.get("m_time_sec", None),
                        "env_steps": int(env_steps),
                        "env_steps_delta": int(info.get("env_steps", 0)),
                    }
                    logger.log(rec)
                final_step = int(step + 1)
                if (step + 1) % max(1, args.save_every) == 0:
                    _save_ckpt(
                        os.path.join(ckpt_dir, f"iter_{step + 1}.pt"),
                        step=int(step + 1),
                        running_min=running_min,
                        running_max=running_max,
                    )
                    _save_ckpt(
                        os.path.join(ckpt_dir, "last.pt"),
                        step=int(step + 1),
                        running_min=running_min,
                        running_max=running_max,
                    )
        else:
            step = int(start_step)
            step_cap = int(args.stop_at) if int(args.stop_at) > 0 else None
            while True:
                if total_timesteps > 0 and int(env_steps) >= int(total_timesteps):
                    break
                if step_cap is not None and step >= int(step_cap):
                    break
                if step % ref_refresh == 0 and step != 0:
                    ref_policy.load_state_dict(policy.state_dict())
                progress = int(env_steps) if use_timesteps else int(step)
                cfg.entropy_coef = entropy_coef_at(progress)
                cfg.action_temp = action_temp_at(
                    progress, std_alpha_val if mixing_enabled else None
                )
                (alpha_mix, lam_mix) = mix_alpha_lambda_at(progress)
                cfg.mix_alpha = alpha_mix
                if cfg.std_algo == "ppo":
                    info = ordo_step_ppo_update(
                        env,
                        policy,
                        policy_kind,
                        device,
                        cfg,
                        optimizer=opt,
                        ref_policy=ref_policy,
                        ref_kind=ref_kind,
                        running_min=running_min,
                        running_max=running_max,
                        value_fn=value_fn,
                    )
                else:
                    info = ordo_step(
                        env,
                        policy,
                        policy_kind,
                        device,
                        cfg,
                        ref_policy=ref_policy,
                        ref_kind=ref_kind,
                        running_min=running_min,
                        running_max=running_max,
                        value_fn=value_fn,
                    )
                env_steps += int(info.get("env_steps", 0))
                rm = info.pop("running_min_arr", None)
                rM = info.pop("running_max_arr", None)
                if rm is not None:
                    running_min = np.array(rm, dtype=np.float32)
                if rM is not None:
                    running_max = np.array(rM, dtype=np.float32)
                if cfg.std_algo != "ppo":
                    loss_tensor = info.pop("loss_tensor")
                    opt.zero_grad()
                    loss_tensor.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
                    opt.step()
                do_log = step == 0 or (step + 1) % max(1, args.log_every) == 0
                if do_log:
                    outer_iter = step // ref_refresh
                    inner_iter = step % ref_refresh
                    rec = {
                        "time": time.time(),
                        "phase": "ordo",
                        "env": args.env,
                        "seed": args.seed,
                        "outer_iter": int(outer_iter),
                        "inner_iter": int(inner_iter),
                        "alpha": alpha_val,
                        "accept": bool(info.get("accept", False)),
                        "L_mhat": float(info.get("L_mhat", float("nan"))),
                        "loss": float(info.get("loss", float("nan"))),
                        "pg_loss": float(info.get("pg_loss", float("nan"))),
                        "pg_loss_ordo": float(info.get("pg_loss_ordo", float("nan"))),
                        "pg_loss_std": info.get("pg_loss_std", None),
                        "vf_loss": info.get("vf_loss", None),
                        "entropy_mean": float(info.get("entropy_mean", float("nan"))),
                        "entropy_coef": float(cfg.entropy_coef),
                        "action_temp": float(cfg.action_temp),
                        "mix_alpha": float(alpha_mix) if mixing_enabled else None,
                        "mix_lambda": float(lam_mix) if mixing_enabled else None,
                        "std_w": std_w_val,
                        "std_utility_family": std_utility_family if mixing_enabled else None,
                        "warm_scalar_w": w_val,
                        "R_mean": info.get("R_cand_mean", None),
                        "R_ref_mean": info.get("R_ref_mean", None),
                        "R_cand_mean": info.get("R_cand_mean", None),
                        "m_hat": info.get("m_hat", None),
                        "m_obj_initial": info.get("m_obj_initial", None),
                        "m_obj_final": info.get("m_obj_final", None),
                        "m_obj_best_seen": info.get("m_obj_best_seen", None),
                        "m_obj_improvement": info.get("m_obj_improvement", None),
                        "m_grad_norm": info.get("m_grad_norm", None),
                        "m_steps": info.get("m_steps", None),
                        "m_time_sec": info.get("m_time_sec", None),
                        "env_steps": int(env_steps),
                        "env_steps_delta": int(info.get("env_steps", 0)),
                    }
                    logger.log(rec)
                final_step = int(step + 1)
                if (step + 1) % max(1, args.save_every) == 0:
                    _save_ckpt(
                        os.path.join(ckpt_dir, f"iter_{step + 1}.pt"),
                        step=int(step + 1),
                        running_min=running_min,
                        running_max=running_max,
                    )
                    _save_ckpt(
                        os.path.join(ckpt_dir, "last.pt"),
                        step=int(step + 1),
                        running_min=running_min,
                        running_max=running_max,
                    )
                if total_timesteps > 0 and int(env_steps) >= int(total_timesteps):
                    print(
                        f"[ORDO] stopping at env_steps={env_steps} (budget={total_timesteps})",
                        flush=True,
                    )
                    break
                step += 1
    _save_ckpt(
        os.path.join(ckpt_dir, f"iter_{final_step}.pt"),
        step=final_step,
        running_min=running_min,
        running_max=running_max,
    )
    _save_ckpt(
        os.path.join(ckpt_dir, "final.pt"),
        step=final_step,
        running_min=running_min,
        running_max=running_max,
    )
    _save_ckpt(
        os.path.join(ckpt_dir, "last.pt"),
        step=final_step,
        running_min=running_min,
        running_max=running_max,
    )
    json_dump(os.path.join(run_dir, "config.json"), vars(args))
    print(f"[ORDO] done. run_dir={run_dir}", flush=True)


if __name__ == "__main__":
    main()
