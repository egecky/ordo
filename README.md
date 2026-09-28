# ORDO: Stochastic-Dominance-Driven Policy Optimization

Code for the paper:

> Ege C. Kaya, Kadierdan Kaheman, Jason M. Cloud, and Abolfazl Hashemi.
> “Stochastic Dominance Driven First-Order Policy Optimization for
> Multi-Objective Reinforcement Learning.” UAI 2026.

## Overview

Orthant Dominance Policy Optimization (ORDO) compares policies using
multivariate, (k)-th-order stochastic dominance in the lower-orthant sense.
It works with integrated multivariate CDFs and minimizes the candidate policy's
worst-case dominance violation relative to an incumbent. Under the paper's
regularity assumptions, the theory gives an ε-almost-non-dominatedness
certificate within the policy class. The experiments focus on the practically
relevant second-order case and include Soft-ORDO smoothing for stable
rollout-based optimization.

The benchmark suite uses two-objective variants of DeepSeaTreasure,
FruitTree, MountainCar, Hopper, and HalfCheetah from MO-Gymnasium. The
benchmark implementation combines ORDO with PPO in some settings. As noted in
the paper, this mixed PPO-ORDO update is an empirical recipe and is **not**
covered by the theorem for the pure ORDO update.

## Setup

Python 3.10 or newer is recommended. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

The continuous-control Hopper and HalfCheetah tasks may require Gymnasium's
MuJoCo dependencies in your environment:

```bash
python -m pip install 'gymnasium[mujoco]'
```

## Single run

Example run on continuous MountainCar:

```bash
python scripts/train_ordo.py \
  --env mo-mountaincarcontinuous-v0 \
  --seed 0 \
  --reward-indices 0,1 \
  --total-timesteps 1000000 \
  --episodes-per-iter 20 \
  --std-algo ppo \
  --std-weight-mode grid \
  --std-grid-n 20 \
  --mix-alpha-start 0.0 \
  --mix-alpha-end 0.2 \
  --mix-alpha-delay 700000 \
  --mix-alpha-anneal-iters 300000 \
  --run-dir runs/example
```

The training script exposes additional options for the environment, policy,
stochastic-dominance order, smoothing, and optimization schedule. Use
`python scripts/train_ordo.py --help` for the full argument list.

## Portfolio and ablation runs

The launch scripts print their commands without running them. Set `RUN=1` to
execute the configured sweep. These are multi-seed, multi-environment
experiments and can require substantial compute.

```bash
bash scripts/run_main_commands.sh
RUN=1 JOBS=4 GPU_IDS=0,1,2,3 bash scripts/run_main_commands.sh

bash scripts/run_ablation_commands.sh
RUN=1 ENV_ID=mo-mountaincarcontinuous-v0 JOBS=4 GPU_IDS=0,1,2,3 \
  bash scripts/run_ablation_commands.sh
```

## Evaluation and plots

Evaluate a saved portfolio across replicate runs:

```bash
python scripts/evaluate_portfolio_replicates.py \
  --env mo-mountaincarcontinuous-v0 \
  --portfolio-roots runs/main/rep_0/ppo_ordo/mo-mountaincarcontinuous-v0 \
  --episodes 256 \
  --reward-indices 0,1 \
  --out eval/mountaincar.json
```

Plot policy return distributions with:

```bash
python scripts/plot_policy_return_distributions.py \
  --policy-root policies \
  --out figures/return_clouds.png
```

## Citation

```bibtex
@inproceedings{kaya2026ordo,
  title     = {Stochastic Dominance Driven First-Order Policy Optimization for Multi-Objective Reinforcement Learning},
  author    = {Kaya, Ege C. and Kaheman, Kadierdan and Cloud, Jason M. and Hashemi, Abolfazl},
  booktitle = {Proceedings of the Conference on Uncertainty in Artificial Intelligence},
  year      = {2026}
}
```
