import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from scripts.evaluate_dpmorl_metrics import (
    compute_constraint_satisfaction,
    compute_expected_utility,
    compute_variance_objective,
    hypervolume_2d,
    hypervolume_mc,
    load_policy_from_run,
    make_env,
    resolve_max_steps,
    set_env_max_steps,
)
from sdmorl.rollout import collect_trajectories, returns_matrix


def _tcrit_975(n):
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.16,
        14: 2.145,
        15: 2.131,
        16: 2.12,
        17: 2.11,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.08,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.06,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
    }
    df = max(1, int(n) - 1)
    return table.get(df, 1.96)


def _compute_portfolio_metrics(args, means, samples_by_pi, rng):
    ref = means.min(axis=0)
    hv = (
        hypervolume_2d(means, ref)
        if means.shape[1] == 2
        else hypervolume_mc(means, ref, args.hv_mc_samples, rng)
    )
    return {
        "EU": compute_expected_utility(means, args.num_weights, rng),
        "HV": hv,
        "ConstraintSatisfaction": compute_constraint_satisfaction(
            samples_by_pi, args.pref_M, args.constraint_n, rng
        ),
        "VarianceObjective": compute_variance_objective(samples_by_pi, args.pref_M, rng),
    }


def _aggregate_metrics(rows, metric_names):
    aggregate = {}
    for name in metric_names:
        vals = np.asarray([r[name] for r in rows], dtype=np.float64)
        n = int(vals.size)
        std = float(vals.std(ddof=1)) if n > 1 else 0.0
        mean = float(vals.mean())
        sem = float(std / np.sqrt(n)) if n > 1 else 0.0
        radius = float(_tcrit_975(n) * sem) if n > 1 else 0.0
        aggregate[name] = {
            "mean": mean,
            "std": std,
            "sem": sem,
            "ci95_low": mean - radius,
            "ci95_high": mean + radius,
            "n": n,
        }
    return aggregate


def _policy_return_summary(policy_means):
    return {
        "mean_over_policies": policy_means.mean(axis=0).astype(float).tolist(),
        "std_over_policies": policy_means.std(axis=0, ddof=1).astype(float).tolist()
        if policy_means.shape[0] > 1
        else np.zeros(policy_means.shape[1], dtype=float).tolist(),
        "n_policies": int(policy_means.shape[0]),
    }


def _portfolio_bootstrap(args, samples_by_pi, rng):
    reps = int(args.portfolio_bootstrap_reps)
    if reps <= 0:
        return None
    n_policies = len(samples_by_pi)
    policy_count = max(1, int(round(n_policies * float(args.bootstrap_policy_frac))))
    boot_rows = []
    for _ in range(reps):
        boot_samples = []
        chosen = rng.integers(0, n_policies, size=policy_count)
        for idx in chosen:
            z = samples_by_pi[int(idx)]
            episode_count = max(1, int(round(len(z) * float(args.bootstrap_episode_frac))))
            ep_idx = rng.integers(0, len(z), size=episode_count)
            boot_samples.append(z[ep_idx])
        boot_means = np.stack([z.mean(axis=0) for z in boot_samples], axis=0)
        boot_rows.append(_compute_portfolio_metrics(args, boot_means, boot_samples, rng))
    out = {}
    for name in ["EU", "HV", "ConstraintSatisfaction", "VarianceObjective"]:
        vals = np.asarray([r[name] for r in boot_rows], dtype=np.float64)
        out[name] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
            "ci95_low": float(np.quantile(vals, 0.025)),
            "ci95_high": float(np.quantile(vals, 0.975)),
            "n": int(vals.size),
        }
    return out


def _aggregate_policy_return_summaries(reps):
    means = np.asarray(
        [r["policy_return_summary"]["mean_over_policies"] for r in reps], dtype=np.float64
    )
    return {
        "mean_over_replicate_portfolios": means.mean(axis=0).astype(float).tolist(),
        "std_over_replicate_portfolios": means.std(axis=0, ddof=1).astype(float).tolist()
        if means.shape[0] > 1
        else np.zeros(means.shape[1], dtype=float).tolist(),
        "n_replicates": int(means.shape[0]),
    }


def _discover_runs(root, pattern):
    runs = sorted((p for p in root.glob(pattern) if (p / "checkpoints").exists()))
    if not runs:
        raise FileNotFoundError(
            f"No run dirs with checkpoints found under {root} using pattern {pattern!r}"
        )
    return runs


def _eval_one_portfolio(args, root, rep_seed):
    rng = np.random.default_rng(rep_seed)
    r_idx = tuple((int(x) for x in args.reward_indices.split(",") if x.strip() != ""))
    env = make_env(args.env)
    max_steps = resolve_max_steps(env, args.max_steps)
    env = set_env_max_steps(env, max_steps)
    runs = _discover_runs(root, args.run_pattern)
    if args.max_policies > 0:
        runs = runs[: int(args.max_policies)]
    means = []
    samples_by_pi = []
    for run_dir in runs:
        (policy, policy_kind) = load_policy_from_run(run_dir, env, args.device)
        traj = collect_trajectories(
            env, policy, policy_kind, args.device, args.gamma, max_steps, args.episodes
        )
        R = returns_matrix(traj, args.gamma)[:, list(r_idx)].astype(np.float32)
        samples_by_pi.append(R)
        means.append(R.mean(axis=0))
    means = np.stack(means, axis=0)
    metrics = _compute_portfolio_metrics(args, means, samples_by_pi, rng)
    per_policy_mean = means.astype(float).tolist()
    per_policy_std = np.stack(
        [z.std(axis=0, ddof=1) if len(z) > 1 else np.zeros(z.shape[1]) for z in samples_by_pi],
        axis=0,
    )
    out = {
        "root": str(root),
        "N_policies": int(len(runs)),
        "episodes_per_policy": int(args.episodes),
        "per_policy_return_mean": per_policy_mean,
        "per_policy_return_std": per_policy_std.astype(float).tolist(),
        "policy_return_summary": _policy_return_summary(means),
        **metrics,
    }
    portfolio_bootstrap = _portfolio_bootstrap(args, samples_by_pi, rng)
    if portfolio_bootstrap is not None:
        out["portfolio_bootstrap"] = portfolio_bootstrap
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument(
        "--portfolio-roots",
        nargs="+",
        required=True,
        help="Each root contains one complete policy portfolio.",
    )
    ap.add_argument(
        "--run-pattern",
        default="seed_*",
        help="Glob under each portfolio root for individual policy run dirs.",
    )
    ap.add_argument("--max-policies", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--reward-indices", type=str, default="0,1")
    ap.add_argument("--num-weights", type=int, default=100)
    ap.add_argument("--hv-mc-samples", type=int, default=200000)
    ap.add_argument("--pref-M", type=int, default=100)
    ap.add_argument("--constraint-n", type=int, default=3)
    ap.add_argument("--portfolio-bootstrap-reps", type=int, default=0)
    ap.add_argument("--bootstrap-policy-frac", type=float, default=1.0)
    ap.add_argument("--bootstrap-episode-frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="eval/portfolio_replicates.json")
    args = ap.parse_args()
    reps = []
    for (i, root) in enumerate(args.portfolio_roots):
        reps.append(_eval_one_portfolio(args, Path(root), int(args.seed) + i))
    metric_names = ["EU", "HV", "ConstraintSatisfaction", "VarianceObjective"]
    aggregate = _aggregate_metrics(reps, metric_names)
    aggregate_source = "portfolio_roots"
    if len(reps) == 1:
        policy_return_summary = reps[0]["policy_return_summary"]
        policy_return_summary_source = "policies"
        portfolio_bootstrap = reps[0].get("portfolio_bootstrap")
    else:
        policy_return_summary = _aggregate_policy_return_summaries(reps)
        policy_return_summary_source = "portfolio_roots"
        portfolio_bootstrap = None
    out = {
        "env": args.env,
        "replicates": reps,
        "aggregate": aggregate,
        "aggregate_source": aggregate_source,
        "policy_return_summary": policy_return_summary,
        "policy_return_summary_source": policy_return_summary_source,
    }
    if portfolio_bootstrap is not None:
        out["portfolio_bootstrap"] = portfolio_bootstrap
        out["portfolio_bootstrap_source"] = "policy_episode_bootstrap"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
