#!/usr/bin/env bash
set -euo pipefail

ENV_ID="${ENV_ID:-mo-mountaincarcontinuous-v0}"
SEEDS="${SEEDS:-5}"
SEED_START="${SEED_START:-0}"
JOBS="${JOBS:-1}"
GPU_IDS="${GPU_IDS:-0}"
TIMESTEPS="${TIMESTEPS:-10000000}"
BASE_OUT="${BASE_OUT:-runs/ablation}"
STD_GRID_N="${STD_GRID_N:-20}"

COMMON="--env ${ENV_ID} --seed {seed} --reward-indices 0,1 --total-timesteps ${TIMESTEPS} --total-iters 100000000 --max-steps 0 --episodes-per-iter 20 --eta 3e-4 --use-critic true --gae-lambda 0.98 --vf-coef 0.5 --entropy-coef 0.001 --std-algo ppo --ppo-clip-range 0.1 --ppo-epochs 5 --ppo-minibatch-size 64 --max-grad-norm 0.5 --std-weight-mode grid --std-grid-n ${STD_GRID_N} --std-utility-family linear --log-every 10 --save-every 999999"
COMMON_AC="${COMMON/--std-algo ppo/--std-algo ac}"
COMMON_REINFORCE="${COMMON_AC/--use-critic true/--use-critic false}"

cmd_for_variant() {
  case "$1" in
    ppo_only)
      printf '%s\n' "python scripts/train_ordo.py ${COMMON} --mix-alpha-start 0.0 --mix-alpha-end 0.0 --run-dir ${BASE_OUT}/ppo_only/${ENV_ID}/seed_{seed}"
      ;;
    ppo_ordo)
      printf '%s\n' "python scripts/train_ordo.py ${COMMON} --mix-alpha-start 0.0 --mix-alpha-end 0.2 --mix-alpha-delay 7000000 --mix-alpha-anneal-iters 3000000 --run-dir ${BASE_OUT}/ppo_ordo/${ENV_ID}/seed_{seed}"
      ;;
    no_delay_ordo)
      printf '%s\n' "python scripts/train_ordo.py ${COMMON} --mix-alpha-start 0.0 --mix-alpha-end 0.2 --mix-alpha-delay 0 --mix-alpha-anneal-iters 3000000 --run-dir ${BASE_OUT}/no_delay_ordo/${ENV_ID}/seed_{seed}"
      ;;
    reduced_ordo)
      printf '%s\n' "python scripts/train_ordo.py ${COMMON} --mix-alpha-start 0.0 --mix-alpha-end 0.05 --mix-alpha-delay 9000000 --mix-alpha-anneal-iters 1000000 --run-dir ${BASE_OUT}/reduced_ordo/${ENV_ID}/seed_{seed}"
      ;;
    no_smoothing)
      printf '%s\n' "python scripts/train_ordo.py ${COMMON} --smooth relu --mix-alpha-start 0.0 --mix-alpha-end 0.2 --mix-alpha-delay 7000000 --mix-alpha-anneal-iters 3000000 --run-dir ${BASE_OUT}/no_smoothing/${ENV_ID}/seed_{seed}"
      ;;
    reinforce_only)
      printf '%s\n' "python scripts/train_ordo.py ${COMMON_REINFORCE} --mix-alpha-start 0.0 --mix-alpha-end 0.0 --run-dir ${BASE_OUT}/reinforce_only/${ENV_ID}/seed_{seed}"
      ;;
    pure_ordo_small)
      printf '%s\n' "python scripts/train_ordo.py --env deep-sea-treasure-v0 --reward-indices 0,1 --seed {seed} --max-steps 400 --total-iters 600 --inner-iters 20 --episodes-per-iter 1024 --k 2 --smooth softplus --mu 2.0 --eta 1e-3 --eps-progress 0.0 --inner-m-steps 50 --inner-m-lr 0.05 --use-critic true --std-algo ac --log-every 1 --print-every 25 --save-every 999999 --run-dir ${BASE_OUT}/pure_ordo_small/deep-sea-treasure-v0/seed_{seed}"
      ;;
    *)
      echo "unknown variant: $1" >&2
      return 2
      ;;
  esac
}

for variant in ppo_only ppo_ordo no_delay_ordo reduced_ordo no_smoothing reinforce_only pure_ordo_small; do
  if [[ -n "${ONLY:-}" && "${ONLY}" != "${variant}" ]]; then
    continue
  fi
  cmd="$(cmd_for_variant "${variant}")"
  echo
  echo "### ${variant}"
  echo "python scripts/launch_many.py --jobs ${JOBS} --seeds ${SEEDS} --seed-start ${SEED_START} --gpu-ids ${GPU_IDS} --cmd \"${cmd}\""
  if [[ "${RUN:-0}" == "1" ]]; then
    python scripts/launch_many.py --jobs "${JOBS}" --seeds "${SEEDS}" --seed-start "${SEED_START}" --gpu-ids "${GPU_IDS}" --cmd "${cmd}"
  fi
done
