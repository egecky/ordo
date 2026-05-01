#!/usr/bin/env bash
set -euo pipefail

ENVS="${ENVS:-mo-mountaincarcontinuous-v0 deep-sea-treasure-v0 fruit-tree-v0 mo-hopper-v4 mo-halfcheetah-v4}"
REPS="${REPS:-5}"
SEEDS="${SEEDS:-20}"
SEED_STRIDE="${SEED_STRIDE:-1000}"
BASE_ROOT="${BASE_OUT:-runs/main}"
TIMESTEPS="${TIMESTEPS:-10000000}"
JOBS="${JOBS:-1}"
GPU_IDS="${GPU_IDS:-0}"
STD_GRID_N="${STD_GRID_N:-20}"

for env_id in $ENVS; do
  for rep in $(seq 0 $((REPS - 1))); do
    export ENV_ID="$env_id"
    export SEEDS
    export SEED_START=$((rep * SEED_STRIDE))
    export BASE_OUT="${BASE_ROOT}/rep_${rep}"
    export TIMESTEPS
    export JOBS
    export GPU_IDS
    export STD_GRID_N
    if [[ "${RUN:-0}" == "1" ]]; then
      ONLY=ppo_ordo RUN=1 bash scripts/run_ablation_commands.sh
    else
      ONLY=ppo_ordo bash scripts/run_ablation_commands.sh
    fi
  done
done
