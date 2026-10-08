"""Data access for the Streamlit dashboard.

Everything here is plain Python + pandas so it can be tested without Streamlit.
The dashboard only *reads* what the framework already writes:

    runs/<name>/{config.json, rounds.jsonl, rounds.csv, summary.json}
    results/benchmark/benchmark_report.json
    results/{ingredients,attribution,fix}/*.json

and launches new runs through the existing CLI, so nothing in ``fedav`` changes.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Fixed colour per arm (validated categorical order, adjacent pairs). Colour
# follows the entity, never its rank, so filtering arms never repaints the rest.
ARM_COLORS = {
    "proposed_fl": "#2a78d6",
    "std_fedavg": "#eb6834",
    "centralized": "#1baf7a",
    "local_only": "#eda100",
}
ARM_LABELS = {
    "centralized": "Centralized",
    "local_only": "Local-only",
    "std_fedavg": "Std. FedAvg",
    "proposed_fl": "Proposed FL",
}
ARM_ORDER = ["proposed_fl", "std_fedavg", "centralized", "local_only"]

# Ablation studies committed under results/: file -> human title.
ABLATION_STUDIES = {
    "ingredients": ("results/ingredients/ingredients.json",
                    "Each enhancement alone, on top of FedAvg"),
    "attribution": ("results/attribution/attribution.json",
                    "Algorithmic changes vs. the compression stack"),
    "fix": ("results/fix/fix.json",
            "FedNova-normalised adaptive epochs"),
}


def _path(*parts: str) -> str:
    return os.path.join(REPO_ROOT, *parts)


def read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------- runs

@dataclass(frozen=True)
class RunInfo:
    name: str
    path: str
    source: str          # "live" (runs/) or "committed" (results/)
    has_summary: bool


def discover_runs(roots: Optional[List[str]] = None) -> List[RunInfo]:
    """Every directory holding ``rounds.jsonl`` or ``rounds.csv`` under the roots."""
    roots = roots or [_path("runs"), _path("results"), _path("runs_ingredient")]
    found: List[RunInfo] = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        source = "committed" if os.path.basename(root) == "results" else "live"
        for dirpath, _dirs, files in os.walk(root):
            if "rounds.jsonl" in files or "rounds.csv" in files:
                rel = os.path.relpath(dirpath, os.path.dirname(root))
                found.append(RunInfo(rel, dirpath, source, "summary.json" in files))
    return sorted(found, key=lambda r: (r.source != "live", r.name))


def load_rounds(run_dir: str) -> pd.DataFrame:
    """Per-round telemetry. ``rounds.jsonl`` is richer, ``rounds.csv`` is the fallback.

    Tolerates a half-written last line, which is normal while a run is live.
    """
    jsonl = os.path.join(run_dir, "rounds.jsonl")
    records: List[Dict[str, Any]] = []
    if os.path.exists(jsonl):
        with open(jsonl, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except ValueError:
                    continue
        if records:
            return pd.DataFrame.from_records(records)
    csv = os.path.join(run_dir, "rounds.csv")
    if os.path.exists(csv):
        try:
            return pd.read_csv(csv)
        except (OSError, ValueError, pd.errors.EmptyDataError):
            pass
    return pd.DataFrame()


def load_summary(run_dir: str) -> Optional[Dict[str, Any]]:
    return read_json(os.path.join(run_dir, "summary.json"))


def load_config(run_dir: str) -> Optional[Dict[str, Any]]:
    return read_json(os.path.join(run_dir, "config.json"))


# ---------------------------------------------------------------------- benchmark

def load_benchmark(path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    return read_json(path or _path("results", "benchmark", "benchmark_report.json"))


def benchmark_table(report: Dict[str, Any]) -> pd.DataFrame:
    """One row per arm with the headline metrics, in display order."""
    rows = []
    results = report.get("results", {})
    for arm in [a for a in ARM_ORDER if a in results] + [a for a in results if a not in ARM_ORDER]:
        s = results[arm]
        rob = s.get("robustness", {}) or {}
        rows.append({
            "arm": arm,
            "Arm": ARM_LABELS.get(arm, arm),
            "Accuracy (%)": 100 * s.get("accuracy", float("nan")),
            "mAP@0.5": s.get("mAP50"),
            "ADE": s.get("ade"),
            "MB / round": s.get("mb_per_round"),
            "Total MB": s.get("total_mb"),
            "Client acc. std": rob.get("client_acc_std"),
            "Worst client (%)": 100 * rob["client_acc_min"] if "client_acc_min" in rob else None,
            "Rounds": s.get("rounds_run"),
            "Raw data leaves vehicle": bool((s.get("privacy") or {}).get("raw_data_egress")),
        })
    return pd.DataFrame(rows)


def acceptance_table(report: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key, check in (report.get("acceptance") or {}).items():
        if not isinstance(check, dict) or "pass" not in check:
            continue
        observed = {k: v for k, v in check.items() if k not in ("requirement", "pass")}
        rows.append({
            "check": key,
            "Requirement": check.get("requirement", ""),
            "Observed": ", ".join(f"{k}={_fmt(v)}" for k, v in observed.items()),
            "pass": bool(check["pass"]),
        })
    return pd.DataFrame(rows)


def benchmark_rounds(report_path: Optional[str] = None) -> pd.DataFrame:
    """Long-format per-round telemetry for every arm next to the benchmark report."""
    base = os.path.dirname(report_path or _path("results", "benchmark", "benchmark_report.json"))
    frames = []
    for arm in ARM_ORDER:
        df = load_rounds(os.path.join(base, arm))
        if df.empty:
            continue
        df = df.copy()
        df["arm"] = arm
        df["Arm"] = ARM_LABELS[arm]
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


# ------------------------------------------------------------------------ ablation

def ablation_table(study: str) -> pd.DataFrame:
    rel, _title = ABLATION_STUDIES[study]
    data = read_json(_path(rel)) or {}
    rows = []
    for name, s in data.items():
        if not isinstance(s, dict) or "accuracy" not in s:
            continue
        rob = s.get("robustness", {}) or {}
        rows.append({
            "Variant": name.replace("_", " "),
            "Accuracy (%)": 100 * s["accuracy"],
            "Client acc. std": rob.get("client_acc_std"),
            "Worst client (%)": 100 * rob["client_acc_min"] if "client_acc_min" in rob else None,
            "MB / round": s.get("mb_per_round"),
            "Rounds": s.get("rounds_run"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- launching

CONFIG_FILES = {
    "quick": "configs/quick.yaml",
    "default": "configs/default.yaml",
    "dp": "configs/dp.yaml",
}


def build_run_command(config: str, name: str, out_dir: str,
                      overrides: Dict[str, Any]) -> List[str]:
    """The CLI invocation the dashboard launches - identical to running it by hand."""
    cmd = [sys.executable, "-m", "fedav.cli", "run", "--config", config,
           "--name", name, "--out-dir", out_dir, "--quiet"]
    if overrides:
        cmd.append("--set")
        cmd.extend(f"{k}={_cli_value(v)}" for k, v in overrides.items())
    return cmd


def _cli_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def launch_run(cmd: List[str], log_path: str) -> subprocess.Popen:
    """Start a run in the background; its rounds stream into ``rounds.jsonl``."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log = open(log_path, "w", encoding="utf-8")
    return subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)


def validation_figures() -> List[str]:
    fig_dir = _path("docs", "figures")
    if not os.path.isdir(fig_dir):
        return []
    return [os.path.join(fig_dir, f) for f in sorted(os.listdir(fig_dir)) if f.endswith(".png")]
