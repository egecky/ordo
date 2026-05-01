# ORDO Experiment Code

This directory contains the ORDO implementation and the scripts used for the
MO-Gym experiments.

## Setup

```bash
pip install -r requirements.txt
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

## Single Run

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

## Portfolio Runs

The main launcher prints commands by default. Set `RUN=1` to execute them.

```bash
bash scripts/run_main_commands.sh
RUN=1 JOBS=4 GPU_IDS=0,1,2,3 bash scripts/run_main_commands.sh
```

The ablation launcher has the same interface.

```bash
bash scripts/run_ablation_commands.sh
RUN=1 ENV_ID=mo-mountaincarcontinuous-v0 JOBS=4 GPU_IDS=0,1,2,3 bash scripts/run_ablation_commands.sh
```

## Evaluation

```bash
python scripts/evaluate_portfolio_replicates.py \
  --env mo-mountaincarcontinuous-v0 \
  --portfolio-roots runs/main/rep_0/ppo_ordo/mo-mountaincarcontinuous-v0 \
  --episodes 256 \
  --reward-indices 0,1 \
  --out eval/mountaincar.json
```

Figure-style return clouds can be regenerated with:

```bash
python scripts/plot_policy_return_distributions.py --policy-root policies --out figures/return_clouds.png
```
