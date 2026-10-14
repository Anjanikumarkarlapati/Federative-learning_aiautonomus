"""Multi-seed statistics for the patent evidence (mean, 95% CI, paired deltas).

With a handful of seeds a bootstrap interval is unreliable, so intervals use
Student's t on the per-seed values, and every table also lists the per-seed
values themselves. Comparisons between arms are *paired*: each seed runs every
arm on the same fleet, so the per-seed difference removes fleet-to-fleet noise.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

# two-sided 95% Student-t quantiles by degrees of freedom
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086, 30: 2.042}


def t95(df: int) -> float:
    if df <= 0:
        return float("nan")
    keys = sorted(_T95)
    for k in keys:
        if df <= k:
            return _T95[k]
    return 1.96


def summarize(values: Sequence[float]) -> Dict[str, Any]:
    v = np.array([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {"n": 0, "mean": None, "ci95": None, "values": []}
    mean = float(v.mean())
    half = float(t95(v.size - 1) * v.std(ddof=1) / math.sqrt(v.size)) if v.size > 1 else None
    return {"n": int(v.size), "mean": mean, "ci95": half,
            "values": [float(x) for x in v]}


def rounds_to(history: List[Dict[str, Any]], threshold: float) -> Optional[int]:
    for r in history:
        if r.get("accuracy", 0.0) >= threshold:
            return int(r["round"])
    return None


def cost_to(history: List[Dict[str, Any]], threshold: float, key: str) -> Optional[float]:
    """Cumulative ``key`` (e.g. mb_round, uplink_airtime_s) up to the first round
    reaching ``threshold`` accuracy."""
    total = 0.0
    for r in history:
        total += float(r.get(key, 0.0) or 0.0)
        if r.get("accuracy", 0.0) >= threshold:
            return total
    return None


def run_metrics(history: List[Dict[str, Any]], summary: Dict[str, Any],
                thresholds: Iterable[float]) -> Dict[str, Any]:
    rows = [r for r in history if "accuracy" in r]
    acc = [r["accuracy"] for r in rows]
    rob = summary.get("robustness", {}) or {}
    out = {
        "best_accuracy": max(acc) if acc else None,
        "last5_accuracy": float(np.mean(acc[-5:])) if acc else None,
        "mb_per_round": summary.get("mb_per_round"),
        "uplink_airtime_s_per_round": float(np.mean([r.get("uplink_airtime_s", 0) for r in rows])) if rows else None,
        "uplink_energy_j_per_round": float(np.mean([r.get("uplink_energy_j", 0) for r in rows])) if rows else None,
        "max_round_airtime_s": float(max(r.get("uplink_airtime_s", 0) for r in rows)) if rows else None,
        "worst_vehicle_accuracy": rob.get("client_acc_min"),
        "client_acc_std": rob.get("client_acc_std"),
        "clients_reported_per_round": float(np.mean([r.get("clients_reported", 0) for r in rows])) if rows else None,
    }
    for t in thresholds:
        tag = f"{int(round(100 * t))}"
        out[f"rounds_to_{tag}"] = rounds_to(rows, t)
        out[f"mb_to_{tag}"] = cost_to(rows, t, "mb_round")
        out[f"airtime_s_to_{tag}"] = cost_to(rows, t, "uplink_airtime_s")
    return out


def aggregate(per_seed: Dict[str, Dict[int, Dict[str, Any]]], reference: str
              ) -> Dict[str, Any]:
    """per_seed[arm][seed] -> metrics. Returns summaries and paired deltas."""
    arms = list(per_seed)
    metrics = sorted({m for arm in arms for s in per_seed[arm].values() for m in s})
    table: Dict[str, Dict[str, Any]] = {}
    for arm in arms:
        table[arm] = {m: summarize([per_seed[arm][s].get(m) for s in sorted(per_seed[arm])])
                      for m in metrics}
    paired: Dict[str, Dict[str, Any]] = {}
    if reference in per_seed:
        for arm in arms:
            if arm == reference:
                continue
            seeds = sorted(set(per_seed[arm]) & set(per_seed[reference]))
            paired[arm] = {}
            for m in metrics:
                diffs = []
                for s in seeds:
                    a, b = per_seed[arm][s].get(m), per_seed[reference][s].get(m)
                    if a is not None and b is not None:
                        diffs.append(a - b)
                paired[arm][m] = summarize(diffs)
    return {"reference": reference, "arms": table, "paired_vs_reference": paired}


def fmt(stat: Dict[str, Any], scale: float = 1.0, digits: int = 2) -> str:
    if not stat or stat.get("mean") is None:
        return "—"
    mean = stat["mean"] * scale
    if stat.get("ci95") is None:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {stat['ci95'] * abs(scale):.{digits}f}"


def write(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


__all__ = ["summarize", "run_metrics", "aggregate", "rounds_to", "cost_to", "fmt", "t95"]
