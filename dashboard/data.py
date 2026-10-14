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


# ----------------------------------------------------------------------- vehicles

VEHICLE_STATUSES = ["reported", "not_selected", "dropped", "straggler", "corrupt", "rejected"]


def has_vehicle_data(run_dir: str) -> bool:
    return os.path.exists(os.path.join(run_dir, "vehicles.json")) or \
        os.path.exists(os.path.join(run_dir, "config.json"))


def load_vehicle_profiles(run_dir: str) -> pd.DataFrame:
    """One row per vehicle as the fleet was built.

    Runs written by this version carry ``vehicles.json``. Older runs only have
    ``config.json``; the fleet is a deterministic function of the config and
    seed, so it is rebuilt exactly from that instead.
    """
    records = read_json(os.path.join(run_dir, "vehicles.json"))
    if records is None:
        cfg = load_config(run_dir)
        if cfg is None:
            return pd.DataFrame()
        records = rebuild_profiles(json.dumps(cfg, sort_keys=True))
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    hist = pd.json_normalize(df.pop("label_histogram")).add_prefix("share_")
    return pd.concat([df, hist], axis=1)


def rebuild_profiles(cfg_json: str) -> List[Dict[str, Any]]:
    from fedav.config import FLConfig
    from fedav.data.federated import build_fleet
    from fedav.metrics.fleet import vehicle_profiles

    return vehicle_profiles(build_fleet(FLConfig.from_dict(json.loads(cfg_json))))


def load_vehicle_rounds(run_dir: str) -> pd.DataFrame:
    """Long format, one row per (round, vehicle). Tolerates a half-written tail."""
    path = os.path.join(run_dir, "vehicle_rounds.jsonl")
    rows: List[Dict[str, Any]] = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    return pd.DataFrame.from_records(rows)


def load_vehicle_summary(run_dir: str) -> pd.DataFrame:
    data = read_json(os.path.join(run_dir, "vehicle_summary.json"))
    if not data:
        return pd.DataFrame()
    return pd.DataFrame.from_records(data.get("vehicles", []))


def participation_from_rounds(rounds: pd.DataFrame) -> pd.DataFrame:
    """Participation counts recomputed from vehicle_rounds - works mid-run."""
    if rounds.empty:
        return pd.DataFrame()
    counts = pd.crosstab(rounds["client_id"], rounds["status"])
    for s in VEHICLE_STATUSES:
        if s not in counts:
            counts[s] = 0
    out = pd.DataFrame({
        "times_selected": counts.drop(columns="not_selected").sum(axis=1),
        "times_reported": counts["reported"],
        "times_dropped": counts["dropped"],
        "times_straggler": counts["straggler"],
        "times_rejected": counts["rejected"],
    })
    out["delivery_rate"] = (out["times_reported"] / out["times_selected"]).where(out["times_selected"] > 0)
    rep = rounds[rounds["status"] == "reported"]
    if not rep.empty:
        means = rep.groupby("client_id")[[c for c in ("q_k", "alpha") if c in rep]].mean()
        out = out.join(means.rename(columns={"q_k": "mean_q", "alpha": "mean_alpha"}))
        out["last_contribution_round"] = rep.groupby("client_id")["round"].max()
    if "accuracy" in rounds:
        latest = rounds.dropna(subset=["accuracy"]).sort_values("round").groupby("client_id").last()
        out["latest_accuracy"] = latest["accuracy"]
    return out.reset_index()


def fleet_table(run_dir: str) -> pd.DataFrame:
    """Every vehicle: profile + participation + how it ends up, in one table."""
    profiles = load_vehicle_profiles(run_dir)
    if profiles.empty:
        return profiles
    summary = load_vehicle_summary(run_dir)
    if summary.empty:
        summary = participation_from_rounds(load_vehicle_rounds(run_dir))
    if not summary.empty:
        profiles = profiles.merge(summary, on="client_id", how="left")
    if "final_accuracy" in profiles:
        profiles["rank"] = profiles["final_accuracy"].rank(ascending=False, method="min")
    return profiles


def group_accuracy(table: pd.DataFrame, by: str, metric: str = "final_accuracy") -> pd.DataFrame:
    if metric not in table or by not in table:
        return pd.DataFrame()
    g = table.dropna(subset=[metric]).groupby(by)[metric]
    return pd.DataFrame({"vehicles": g.size(), "mean": g.mean(), "min": g.min(),
                         "max": g.max()}).reset_index()


def compare_arms_per_vehicle(base_dir: Optional[str] = None) -> pd.DataFrame:
    """Wide table: each vehicle's final accuracy under every benchmark arm."""
    base_dir = base_dir or _path("results", "benchmark")
    cols = {}
    for arm in ARM_ORDER:
        s = load_vehicle_summary(os.path.join(base_dir, arm))
        if not s.empty and "final_accuracy" in s:
            cols[arm] = s.set_index("client_id")["final_accuracy"]
    return pd.DataFrame(cols)


# ------------------------------------------------------------ patent evidence

PATENT_ARMS = {"std_fedavg": "A · Std. FedAvg", "fedavg_transport": "B · FedAvg + same transport",
               "proposed_secure": "C · Quality-hiding secure agg.",
               "proposed_tiered": "D · C + airtime tiers"}
PATENT_COLORS = {"std_fedavg": "#eb6834", "fedavg_transport": "#eda100",
                 "proposed_secure": "#1baf7a", "proposed_tiered": "#2a78d6"}


def load_patent_effects() -> Optional[Dict[str, Any]]:
    return read_json(_path("results", "patent", "effects.json"))


def patent_table(effects: Dict[str, Any], metric: str, scale: float = 1.0) -> pd.DataFrame:
    """One row per arm: mean, CI half-width and per-seed values of ``metric``."""
    arms = effects["vs_std_fedavg"]["arms"]
    rows = []
    for arm, label in PATENT_ARMS.items():
        s = arms.get(arm, {}).get(metric) or {}
        if s.get("mean") is None:
            continue
        rows.append({"arm": arm, "Arm": label, "mean": scale * s["mean"],
                     "ci95": scale * (s.get("ci95") or 0.0),
                     "values": [scale * v for v in s.get("values", [])]})
    return pd.DataFrame(rows)


def patent_paired(effects: Dict[str, Any], arm: str, metric: str,
                  reference: str = "vs_fedavg_transport") -> Optional[Dict[str, Any]]:
    return (effects.get(reference, {}).get("paired_vs_reference", {}).get(arm, {}) or {}).get(metric)


def patent_drawings() -> List[str]:
    d = _path("docs", "patent", "drawings")
    if not os.path.isdir(d):
        return []
    return [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".png")]


def load_options() -> Optional[Dict[str, Any]]:
    return read_json(_path("results", "options", "options.json"))


def options_paired(options: Dict[str, Any], metric: str, scale: float = 1.0) -> pd.DataFrame:
    """Per option: mean and CI of (option − base) for ``metric``."""
    paired = options["result"]["paired_vs_reference"]
    rows = []
    for arm, metrics in paired.items():
        s = metrics.get(metric) or {}
        if s.get("mean") is None:
            continue
        rows.append({"option": arm.replace("_", " "), "mean": scale * s["mean"],
                     "ci95": scale * (s.get("ci95") or 0.0), "n": s.get("n", 0)})
    return pd.DataFrame(rows)
