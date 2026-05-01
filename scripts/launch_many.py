import argparse
import os
import subprocess
import time
from pathlib import Path


def parse_gpu_ids(s):
    ids = [x.strip() for x in s.split(",") if x.strip() != ""]
    if not ids:
        raise ValueError("Empty --gpu-ids")
    return [int(x) for x in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4, help="max concurrent trainings")
    ap.add_argument("--seeds", type=int, default=20, help="number of seeds to run")
    ap.add_argument("--seed-start", type=int, default=0, help="first seed value")
    ap.add_argument(
        "--gpu-ids",
        type=str,
        default="",
        help='comma-separated GPU ids, e.g. "0,2,3"',
    )
    ap.add_argument(
        "--cmd",
        type=str,
        required=True,
        help='command template with a "{seed}" placeholder',
    )
    args = ap.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    gpu_ids = parse_gpu_ids(args.gpu_ids) if args.gpu_ids.strip() else None
    procs = []
    next_local = 0
    while next_local < args.seeds or procs:
        while next_local < args.seeds and len(procs) < args.jobs:
            seed = int(args.seed_start) + next_local
            gpu = gpu_ids[next_local % len(gpu_ids)] if gpu_ids else None
            cmd = args.cmd.format(seed=seed)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
            if gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            print(f"[LAUNCH] seed={seed} gpu={gpu} cmd={cmd}", flush=True)
            p = subprocess.Popen(cmd, shell=True, cwd=str(repo_root), env=env)
            procs.append((seed, gpu, p))
            next_local += 1
        alive = []
        for (seed, gpu, p) in procs:
            ret = p.poll()
            if ret is None:
                alive.append((seed, gpu, p))
            else:
                print(f"[DONE] seed={seed} gpu={gpu} exit={ret}", flush=True)
        procs = alive
        time.sleep(1)


if __name__ == "__main__":
    main()
