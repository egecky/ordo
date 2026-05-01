from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import time
import os
import sys

DEFAULT_KEYS = [
    "time",
    "phase",
    "env",
    "seed",
    "outer_iter",
    "inner_iter",
    "alpha",
    "accept",
    "L_mhat",
    "loss",
    "pg_loss",
    "pg_loss_ordo",
    "pg_loss_std",
    "vf_loss",
    "entropy_mean",
    "entropy_coef",
    "mix_alpha",
    "mix_lambda",
    "std_w",
    "warm_scalar_w",
    "R_mean",
    "R_ref_mean",
    "R_cand_mean",
    "m_hat",
]


def _ensure_schema(rec):
    out = {k: rec.get(k, None) for k in DEFAULT_KEYS}
    for (k, v) in rec.items():
        if k not in out:
            out[k] = v
    return out


def _flatten_2(v):
    if v is None:
        return [None, None]
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        return [float(v[0]), float(v[1])]
    return [None, None]


def _fmt_float(x, width=9):
    if x is None:
        return "-".rjust(width)
    try:
        s = f"{float(x):.3g}"
        return s.rjust(width)
    except Exception:
        return "-".rjust(width)


def _fmt_int(x, width=4):
    try:
        return f"{int(x):d}".rjust(width)
    except Exception:
        return "-".rjust(width)


class ProgressLogger:
    def __init__(self, run_dir, print_every=1, header_every=50):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.run_dir / "metrics.jsonl"
        self.tsv_path = self.run_dir / "progress.tsv"
        self.print_every = max(1, int(print_every))
        self.header_every = max(1, int(header_every))
        self._row_idx = 0
        self._t0 = time.time()
        term = os.environ.get("TERM", "")
        self.use_color = (
            bool(sys.stdout.isatty())
            and os.environ.get("NO_COLOR") is None
            and (term.lower() != "dumb")
        )
        if not self.tsv_path.exists():
            self.tsv_path.write_text(self._tsv_header() + "\n", encoding="utf-8")

    def _tsv_header(self):
        cols = [
            "time",
            "phase",
            "outer",
            "inner",
            "alpha",
            "accept",
            "L_mhat",
            "loss",
            "pg_loss",
            "pg_ordo",
            "pg_std",
            "vf_loss",
            "entropy",
            "entcoef",
            "mix_alpha",
            "mix_lambda",
            "warm_w0",
            "warm_w1",
            "std_w0",
            "std_w1",
            "R0",
            "R1",
            "ref0",
            "ref1",
            "cand0",
            "cand1",
            "m0",
            "m1",
        ]
        return "\t".join(cols)

    def _seed_color(self, seed):
        palette = ["\x1b[96m", "\x1b[92m", "\x1b[93m", "\x1b[95m"]
        try:
            s = int(seed)
        except Exception:
            s = 0
        return palette[s % len(palette)]

    def _colorize(self, line, seed):
        if not self.use_color:
            return line
        return f"{self._seed_color(seed)}{line}\x1b[0m"

    def log(self, rec):
        rec = _ensure_schema(rec)
        t = rec.get("time", None)
        if t is None:
            t_rel = int(time.time() - self._t0)
        else:
            try:
                t_f = float(t)
                if t_f > 10000000.0:
                    t_rel = int(round(t_f - self._t0))
                else:
                    t_rel = int(round(t_f))
            except Exception:
                t_rel = int(time.time() - self._t0)
        rec["time"] = t_rel
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=False, separators=(",", ":")) + "\n")
        (R0, R1) = _flatten_2(rec.get("R_mean"))
        (ref0, ref1) = _flatten_2(rec.get("R_ref_mean"))
        (cand0, cand1) = _flatten_2(rec.get("R_cand_mean"))
        (m0, m1) = _flatten_2(rec.get("m_hat"))
        (w0, w1) = _flatten_2(rec.get("warm_scalar_w"))
        (sw0, sw1) = _flatten_2(rec.get("std_w"))
        row = [
            str(int(rec.get("time", 0))),
            str(rec.get("phase")),
            str(rec.get("outer_iter")),
            str(rec.get("inner_iter")),
            str(rec.get("alpha")),
            str(rec.get("accept")),
            str(rec.get("L_mhat")),
            str(rec.get("loss")),
            str(rec.get("pg_loss")),
            str(rec.get("pg_loss_ordo")),
            str(rec.get("pg_loss_std")),
            str(rec.get("vf_loss")),
            str(rec.get("entropy_mean")),
            str(rec.get("entropy_coef")),
            str(rec.get("mix_alpha")),
            str(rec.get("mix_lambda")),
            str(w0),
            str(w1),
            str(sw0),
            str(sw1),
            str(R0),
            str(R1),
            str(ref0),
            str(ref1),
            str(cand0),
            str(cand1),
            str(m0),
            str(m1),
        ]
        with self.tsv_path.open("a", encoding="utf-8") as f:
            f.write("\t".join(row) + "\n")
        self._maybe_print(rec, R0, R1, ref0, ref1, cand0, cand1, m0, m1)
        self._row_idx += 1

    def _maybe_print(self, rec, R0, R1, ref0, ref1, cand0, cand1, m0, m1):
        if self._row_idx % self.print_every != 0:
            return
        if self._row_idx % self.header_every == 0:
            print(self._pretty_header(), flush=True)
        line = self._pretty_row(rec, R0, R1, ref0, ref1, cand0, cand1, m0, m1)
        print(self._colorize(line, rec.get("seed")), flush=True)

    def _pretty_header(self):
        cols = [
            ("phase", 5, "<"),
            ("out", 4, ">"),
            ("in", 3, ">"),
            ("seed", 4, ">"),
            ("acc", 3, ">"),
            ("L_mhat", 9, ">"),
            ("loss", 9, ">"),
            ("pgO", 9, ">"),
            ("pgS", 9, ">"),
            ("mix", 6, ">"),
            ("ent", 9, ">"),
            ("entc", 9, ">"),
            ("R0", 9, ">"),
            ("R1", 9, ">"),
            ("cand0", 9, ">"),
            ("cand1", 9, ">"),
            ("m0", 9, ">"),
            ("m1", 9, ">"),
        ]
        return " ".join((f"{name:{align}{width}}" for (name, width, align) in cols))

    def _pretty_row(self, rec, R0, R1, ref0, ref1, cand0, cand1, m0, m1):
        acc = rec.get("accept")
        acc_s = "Y" if acc is True else "N" if acc is False else "-"
        mix_a = rec.get("mix_alpha")
        mix_s = "-" if mix_a is None else f"{float(mix_a):.2f}"
        parts = [
            f"{str(rec.get('phase')):<5}",
            _fmt_int(rec.get("outer_iter"), 4),
            _fmt_int(rec.get("inner_iter"), 3),
            _fmt_int(rec.get("seed"), 4),
            f"{acc_s:>3}",
            _fmt_float(rec.get("L_mhat")),
            _fmt_float(rec.get("loss")),
            _fmt_float(rec.get("pg_loss_ordo")),
            _fmt_float(rec.get("pg_loss_std")),
            f"{mix_s:>6}",
            _fmt_float(rec.get("entropy_mean")),
            _fmt_float(rec.get("entropy_coef")),
            _fmt_float(R0),
            _fmt_float(R1),
            _fmt_float(cand0),
            _fmt_float(cand1),
            _fmt_float(m0),
            _fmt_float(m1),
        ]
        return " ".join(parts)
