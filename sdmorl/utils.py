from typing import Any, Dict
import os, json, time, random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def now_ts():
    return time.strftime("%Y%m%d-%H%M%S")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def jsonl_append(path, row):
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def json_dump(path, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def pareto_nondominated(points, maximize=True):
    (N, _) = points.shape
    nd = np.ones(N, dtype=bool)
    for i in range(N):
        if not nd[i]:
            continue
        if maximize:
            dom = np.all(points >= points[i], axis=1) & np.any(points > points[i], axis=1)
        else:
            dom = np.all(points <= points[i], axis=1) & np.any(points < points[i], axis=1)
        if np.any(dom):
            nd[i] = False
            continue
        if maximize:
            i_dom = np.all(points[i] >= points, axis=1) & np.any(points[i] > points, axis=1)
        else:
            i_dom = np.all(points[i] <= points, axis=1) & np.any(points[i] < points, axis=1)
        nd[i_dom] = False
        nd[i] = True
    return nd


def hypervolume_2d(points, ref):
    pts = points.copy()
    pts = pts[(pts[:, 0] > ref[0]) & (pts[:, 1] > ref[1])]
    if len(pts) == 0:
        return 0.0
    nd = pareto_nondominated(pts, maximize=True)
    pts = pts[nd]
    pts = pts[np.argsort(-pts[:, 0])]
    hv = 0.0
    y_max = ref[1]
    for (x, y) in pts:
        if y > y_max:
            hv += (x - ref[0]) * (y - y_max)
            y_max = y
    return float(hv)


def hypervolume_mc(points, ref, samples=200000):
    pts = points.copy()
    pts = pts[np.all(pts > ref, axis=1)]
    if len(pts) == 0:
        return 0.0
    upper = np.max(pts, axis=0)
    U = np.random.rand(samples, pts.shape[1])
    X = ref + U * (upper - ref)
    dom = np.zeros(samples, dtype=bool)
    for p in pts:
        dom |= np.all(p >= X, axis=1)
    vol_box = float(np.prod(upper - ref))
    return vol_box * float(np.mean(dom))


def dpmorl_normalize_returns(R, zmin, zmax):
    zmin = np.asarray(zmin, dtype=np.float32)
    zmax = np.asarray(zmax, dtype=np.float32)
    zmid = 0.5 * (zmin + zmax)
    d = float(np.max(zmax - zmin))
    if d < 1e-08:
        d = 1.0
    Rn = (R - zmid[None, :]) / d + 0.5
    return np.clip(Rn, 0.0, 1.0).astype(np.float32)
