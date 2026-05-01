import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from sdmorl.policy import build_policy
from sdmorl.rollout import collect_trajectories, returns_matrix


def enable_paper_style():
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.dpi": 400,
        }
    )


def paper_axes(ax):
    ax.grid(True, alpha=0.2, linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    ax.tick_params(width=1.0, length=4)


def make_env(env_id):
    try:
        import mo_gymnasium as mo_gym

        return mo_gym.make(env_id)
    except Exception:
        import mo_gym

        return mo_gym.make(env_id)


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


def load_policy_from_seed_dir(seed_dir, env, device):
    ckpt = seed_dir / "checkpoints" / "final.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
    (policy, policy_kind) = build_policy(env)
    policy.to(device)
    state = torch.load(ckpt, map_location=device)
    if isinstance(state, dict) and "policy" in state:
        state = state["policy"]
    policy.load_state_dict(state)
    policy.eval()
    return (policy, policy_kind)


ENV_DISPLAY = {
    "mo-mountaincarcontinuous-v0": "MountainCar",
    "deep-sea-treasure-v0": "DeepSeaTreasure",
    "fruit-tree-v0": "FruitTree",
    "mo-hopper-v4": "Hopper",
    "mo-halfcheetah-v4": "HalfCheetah",
}
DEFAULT_ENVS = [
    "mo-mountaincarcontinuous-v0",
    "deep-sea-treasure-v0",
    "fruit-tree-v0",
    "mo-hopper-v4",
    "mo-halfcheetah-v4",
]
TAB20 = list(plt.get_cmap("tab20").colors)
MARKERS = [
    "o",
    "s",
    "^",
    "v",
    "D",
    "P",
    "X",
    "<",
    ">",
    "h",
    "H",
    "p",
    "8",
    "d",
    "*",
    "+",
    "x",
    "1",
    "2",
    "3",
]


def seed_dirs_for_env(policy_root, env_name, expected_n=20):
    base = policy_root / env_name
    if not base.exists():
        raise FileNotFoundError(f"Missing env directory: {base}")
    dirs = []
    for i in range(expected_n):
        d = base / f"seed_{i}"
        if d.exists():
            dirs.append(d)
    return dirs


def parse_reward_indices(s):
    xs = [int(x.strip()) for x in s.split(",") if x.strip() != ""]
    if len(xs) != 2:
        raise ValueError(f"--reward-indices must have exactly 2 ints, got: {s}")
    return (xs[0], xs[1])


def cache_path(
    cache_dir, env_name, seed, gamma, max_steps, reward_indices, action_temp, episodes_cached
):
    ri = f"{reward_indices[0]}_{reward_indices[1]}"
    g = f"{gamma:.6g}".replace(".", "p")
    at = f"{action_temp:.6g}".replace(".", "p")
    ms = str(int(max_steps))
    return cache_dir / env_name / f"seed_{seed}_ep{episodes_cached}_g{g}_ms{ms}_ri{ri}_at{at}.npz"


def collect_or_load_returns(
    env_name,
    seed_dirs,
    episodes_plot,
    episodes_cache,
    gamma,
    max_steps,
    device,
    reward_indices,
    action_temp,
    cache_dir,
    recompute,
):
    env = make_env(env_name)
    eff_max_steps = resolve_max_steps(env, max_steps)
    env = set_env_max_steps(env, eff_max_steps)
    out = {}
    for seed_dir in seed_dirs:
        seed_idx = int(seed_dir.name.split("_")[-1])
        if cache_dir is not None:
            p = cache_path(
                cache_dir,
                env_name,
                seed_idx,
                gamma,
                eff_max_steps,
                reward_indices,
                action_temp,
                episodes_cache,
            )
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.exists() and (not recompute):
                data = np.load(p)
                R = data["R"]
                out[seed_idx] = R[:episodes_plot]
                continue
        (policy, policy_kind) = load_policy_from_seed_dir(seed_dir, env, device)
        trajs = collect_trajectories(
            env=env,
            policy=policy,
            policy_kind=policy_kind,
            device=device,
            gamma=gamma,
            max_steps=eff_max_steps,
            n_episodes=episodes_cache,
            action_temp=action_temp,
        )
        R_full = returns_matrix(trajs, gamma)[:, list(reward_indices)].astype(np.float32)
        if cache_dir is not None:
            np.savez_compressed(
                p,
                R=R_full,
                env=env_name,
                seed=seed_idx,
                episodes_cached=episodes_cache,
                episodes_plot=episodes_plot,
                gamma=gamma,
                max_steps=eff_max_steps,
                reward_indices=np.array(reward_indices, dtype=np.int64),
                action_temp=action_temp,
            )
        out[seed_idx] = R_full[:episodes_plot]
    try:
        env.close()
    except Exception:
        pass
    return out


def make_legend_handles(n=20):
    handles = []
    for i in range(n):
        handles.append(
            Line2D(
                [0],
                [0],
                linestyle="None",
                marker=MARKERS[i % len(MARKERS)],
                markersize=9,
                markerfacecolor=TAB20[i % 20],
                markeredgecolor="black",
                markeredgewidth=0.4,
                alpha=0.95,
                label=f"$\\pi_{{{i}}}$",
            )
        )
    return handles


def plot_grid(
    data_by_env, outpath, env_order, point_alpha, point_size, dpi, means_only, max_policies
):
    enable_paper_style()
    fig = plt.figure(figsize=(14.5, 8.4))
    gs = GridSpec(3, 6, figure=fig, height_ratios=[1.0, 1.0, 0.22], hspace=0.45, wspace=0.65)
    axes = {}
    axes[env_order[0]] = fig.add_subplot(gs[0, 0:2])
    axes[env_order[1]] = fig.add_subplot(gs[0, 2:4])
    axes[env_order[2]] = fig.add_subplot(gs[0, 4:6])
    axes[env_order[3]] = fig.add_subplot(gs[1, 1:3])
    axes[env_order[4]] = fig.add_subplot(gs[1, 3:5])
    leg_ax = fig.add_subplot(gs[2, :])
    leg_ax.axis("off")
    for env_name in env_order:
        ax = axes[env_name]
        title = ENV_DISPLAY.get(env_name, env_name)
        env_data = data_by_env.get(env_name, {})
        for seed_idx in sorted(env_data.keys()):
            R = env_data[seed_idx]
            if R.size == 0:
                continue
            if means_only:
                R_plot = R.mean(axis=0, keepdims=True)
                s = point_size * 2.5
                alpha = 1.0
                edgecolors = "black"
                linewidths = 0.5
            else:
                R_plot = R
                s = point_size
                alpha = point_alpha
                edgecolors = "none"
                linewidths = 0.0
            ax.scatter(
                R_plot[:, 0],
                R_plot[:, 1],
                s=s,
                alpha=alpha,
                c=[TAB20[seed_idx % 20]],
                marker=MARKERS[seed_idx % len(MARKERS)],
                edgecolors=edgecolors,
                linewidths=linewidths,
            )
        ax.set_title(title, pad=6)
        ax.set_xlabel("Return 1")
        ax.set_ylabel("Return 2")
        paper_axes(ax)
    handles = make_legend_handles(max_policies)
    leg_ax.legend(
        handles=handles,
        loc="center",
        ncol=10,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        borderpad=0.5,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(outpath), dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-root", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="Episodes to PLOT per policy (more -> denser clouds)",
    )
    ap.add_argument(
        "--cache-episodes",
        type=int,
        default=200,
        help="episodes to cache per policy",
    )
    ap.add_argument(
        "--cache-dir",
        type=str,
        default="eval_cache/return_clouds",
        help="Where to save/reuse rollout return clouds (.npz).",
    )
    ap.add_argument(
        "--recompute-cache", action="store_true", help="Force re-rollout even if cache exists."
    )
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--reward-indices", type=str, default="0,1")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--action-temp", type=float, default=1.0)
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--point-size", type=float, default=48.0)
    ap.add_argument("--point-alpha", type=float, default=0.85)
    ap.add_argument(
        "--max-policies",
        type=int,
        default=20,
        help="Plot only the first N policy seeds to reduce clutter.",
    )
    ap.add_argument(
        "--means-only",
        action="store_true",
        help="Plot one mean-return marker per policy instead of all episodes.",
    )
    ap.add_argument("--also-pdf", action="store_true")
    args = ap.parse_args()
    policy_root = Path(args.policy_root)
    outpath = Path(args.out)
    reward_indices = parse_reward_indices(args.reward_indices)
    env_order = DEFAULT_ENVS
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    episodes_plot = int(args.episodes)
    episodes_cache = int(max(args.cache_episodes, episodes_plot))
    data_by_env = {}
    for env_name in env_order:
        seed_dirs = seed_dirs_for_env(policy_root, env_name, expected_n=20)[
            : int(args.max_policies)
        ]
        env_data = collect_or_load_returns(
            env_name=env_name,
            seed_dirs=seed_dirs,
            episodes_plot=episodes_plot,
            episodes_cache=episodes_cache,
            gamma=float(args.gamma),
            max_steps=int(args.max_steps),
            device=str(args.device),
            reward_indices=reward_indices,
            action_temp=float(args.action_temp),
            cache_dir=cache_dir,
            recompute=bool(args.recompute_cache),
        )
        data_by_env[env_name] = env_data
    plot_grid(
        data_by_env=data_by_env,
        outpath=outpath,
        env_order=env_order,
        point_alpha=float(args.point_alpha),
        point_size=float(args.point_size),
        dpi=int(args.dpi),
        means_only=bool(args.means_only),
        max_policies=int(args.max_policies),
    )
    if args.also_pdf:
        pdf_path = outpath.with_suffix(".pdf")
        plot_grid(
            data_by_env=data_by_env,
            outpath=pdf_path,
            env_order=env_order,
            point_alpha=float(args.point_alpha),
            point_size=float(args.point_size),
            dpi=int(args.dpi),
            means_only=bool(args.means_only),
            max_policies=int(args.max_policies),
        )


if __name__ == "__main__":
    main()
