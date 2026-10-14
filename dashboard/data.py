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


# -------------------------------------------------------------- richer summaries

def run_summary_row(s: Dict[str, Any]) -> Dict[str, Any]:
    """Everything a ``summary.json`` says, flattened into one display row."""
    rob = s.get("robustness", {}) or {}
    sel = s.get("selection", {}) or {}
    priv = s.get("privacy", {}) or {}

    def pct(v):
        return None if v is None else 100 * v

    return {
        "Accuracy (%)": pct(s.get("accuracy")),
        "mAP@0.5": s.get("mAP50"),
        "ADE": s.get("ade"),
        "FDE": s.get("fde"),
        "Mean client (%)": pct(rob.get("client_acc_mean")),
        "P10 client (%)": pct(rob.get("client_acc_p10")),
        "Worst client (%)": pct(rob.get("client_acc_min")),
        "Client acc. std": rob.get("client_acc_std"),
        "MB / round": s.get("mb_per_round"),
        "Total MB": s.get("total_mb"),
        "Rounds": s.get("rounds_run"),
        "Rounds to target": s.get("rounds_to_target"),
        "Converged": s.get("converged"),
        "Selection Gini": sel.get("selection_gini"),
        "Mean participation": sel.get("mean_participation"),
        "Never selected": sel.get("clients_never_selected"),
        "Privacy": priv.get("mode"),
        "Secure agg.": priv.get("secure_aggregation"),
        "Wall clock (min)": None if s.get("wall_clock_s") is None else s["wall_clock_s"] / 60,
    }


def benchmark_full_table(report: Dict[str, Any]) -> pd.DataFrame:
    """Like :func:`benchmark_table` but with every field the summaries carry."""
    results = report.get("results", {})
    rows = []
    for arm in [a for a in ARM_ORDER if a in results] + [a for a in results if a not in ARM_ORDER]:
        rows.append({"arm": arm, "Arm": ARM_LABELS.get(arm, arm), **run_summary_row(results[arm])})
    return pd.DataFrame(rows)


def rounds_to_threshold(rounds: pd.DataFrame, threshold: float,
                        metric: str = "accuracy", group: str = "arm") -> Dict[str, Optional[int]]:
    """First round at which each group reaches ``metric >= threshold`` (None if never)."""
    out: Dict[str, Optional[int]] = {}
    if rounds.empty or metric not in rounds:
        return out
    for key, sub in rounds.groupby(group):
        hit = sub[sub[metric] >= threshold]
        out[key] = int(hit["round"].min()) if not hit.empty else None
    return out


def smooth(series: pd.Series, window: int) -> pd.Series:
    """Trailing moving average; window 1 returns the series unchanged."""
    return series if window <= 1 else series.rolling(window, min_periods=1).mean()


def status_per_round(vehicle_rounds: pd.DataFrame) -> pd.DataFrame:
    """Rows = round, columns = vehicle status, values = number of vehicles."""
    if vehicle_rounds.empty or "status" not in vehicle_rounds:
        return pd.DataFrame()
    counts = pd.crosstab(vehicle_rounds["round"], vehicle_rounds["status"])
    return counts.reindex(columns=[s for s in VEHICLE_STATUSES if s in counts])


# -------------------------------------------------------------------- comparing

def flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def config_diff(configs: Dict[str, Dict[str, Any]], only_different: bool = True) -> pd.DataFrame:
    """Rows = config keys, columns = runs. ``name``/``out_dir`` are always ignored."""
    flat = {name: flatten(cfg) for name, cfg in configs.items() if cfg}
    if not flat:
        return pd.DataFrame()
    df = pd.DataFrame(flat).drop(index=["name", "out_dir"], errors="ignore")
    df = df.map(lambda v: json.dumps(v) if isinstance(v, (list, dict)) else v)
    if only_different and df.shape[1] > 1:
        df = df[df.astype(str).nunique(axis=1) > 1]
    return df.sort_index()


def runs_rounds(run_dirs: Dict[str, str]) -> pd.DataFrame:
    """Long-format per-round telemetry for any set of runs, keyed by label."""
    frames = []
    for label, path in run_dirs.items():
        df = load_rounds(path)
        if not df.empty:
            frames.append(df.assign(run=label))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# Per-round metrics the framework logs, with display name, unit scale and direction.
ROUND_METRICS = {
    "accuracy": ("Global test accuracy", "Accuracy (%)", 100, "higher"),
    "loss": ("Global test loss", "Loss", 1, "lower"),
    "mAP50": ("Detection mAP@0.5", "mAP@0.5", 1, "higher"),
    "ade": ("Trajectory ADE", "ADE", 1, "lower"),
    "fde": ("Trajectory FDE", "FDE", 1, "lower"),
    "client_acc_mean": ("Mean per-vehicle accuracy", "Accuracy (%)", 100, "higher"),
    "client_acc_min": ("Worst per-vehicle accuracy", "Accuracy (%)", 100, "higher"),
    "client_acc_std": ("Cross-vehicle accuracy std", "Std", 1, "lower"),
    "mb_round": ("Uplink per round", "MB", 1, "lower"),
    "mb_cumulative": ("Cumulative uplink", "MB", 1, "lower"),
    "clients_reported": ("Vehicles reporting", "Vehicles", 1, "higher"),
    "delta_norm": ("Global update norm ‖Δw‖", "Norm", 1, "lower"),
    "epsilon_spent": ("Privacy budget spent ε", "ε", 1, "lower"),
    "lr": ("Learning rate", "η", 1, "—"),
    "round_seconds": ("Wall-clock per round", "Seconds", 1, "lower"),
    "mean_q": ("Mean quality weight q̄", "q", 1, "—"),
}


def available_metrics(df: pd.DataFrame) -> List[str]:
    return [m for m in ROUND_METRICS if m in df and df[m].notna().any()]


# ------------------------------------------------------------- ablation, richer

def ablation_full_table(study: str) -> pd.DataFrame:
    """Every summary field per variant, plus deltas against the study's baseline."""
    if study == "cli":
        data = read_json(_path("runs", "ablation.json")) or {}
    else:
        data = read_json(_path(ABLATION_STUDIES[study][0])) or {}
    rows = []
    for name, s in data.items():
        if isinstance(s, dict) and "accuracy" in s:
            rows.append({"Variant": name.replace("_", " "), **run_summary_row(s)})
    if rows and baseline_variant([r["Variant"] for r in rows]) is None and study != "cli":
        # The committed studies share one 30-round FedAvg control (README section 9).
        control = (read_json(_path(ABLATION_STUDIES["attribution"][0])) or {}).get("fedavg_control")
        if control:
            rows.insert(0, {"Variant": "fedavg control (shared)", **run_summary_row(control)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    base = baseline_variant(df["Variant"].tolist())
    if base is not None:
        ref = df.set_index("Variant").loc[base]
        for col in ("Accuracy (%)", "Worst client (%)", "MB / round"):
            df[f"Δ {col}"] = df[col] - ref[col]
    df.attrs["baseline"] = base
    return df


def baseline_variant(names: List[str]) -> Optional[str]:
    for n in names:
        if "fedavg" in n or "control" in n or "baseline" in n:
            return n
    return None


def ablation_studies_available() -> Dict[str, str]:
    """Committed studies plus the latest ``fedav.cli ablation`` run if one exists."""
    out = {k: v[1] for k, v in ABLATION_STUDIES.items()}
    if os.path.exists(_path("runs", "ablation.json")):
        out["cli"] = "Latest `fedav.cli ablation` run (runs/ablation.json)"
    return out


def validation_sections() -> List[Dict[str, str]]:
    """The numbered sections of docs/MATHEMATICAL_VALIDATION.md: title, figure, text.

    Falls back to bare figures when the document is missing.
    """
    import re

    doc = _path("docs", "MATHEMATICAL_VALIDATION.md")
    sections: List[Dict[str, str]] = []
    if os.path.exists(doc):
        with open(doc, encoding="utf-8") as fh:
            text = fh.read()
        for block in re.split(r"(?m)^## ", text)[1:]:
            title, _, body = block.partition("\n")
            m = re.match(r"(\d+)\.\s+(.*)", title.strip())
            img = re.search(r"!\[[^\]]*\]\((figures/[^)]+\.png)\)", body)
            if not m or not img:
                continue
            body = body.replace(img.group(0), "").strip()
            body = re.sub(r"\[([^\]]+)\]\((?!https?://)[^)]+\)", r"\1", body)  # links don't resolve here
            sections.append({"number": m.group(1), "title": m.group(2).strip(),
                             "image": _path("docs", img.group(1)), "text": body})
    if not sections:
        for i, path in enumerate(validation_figures(), 1):
            name = os.path.splitext(os.path.basename(path))[0].split("_", 1)[-1]
            sections.append({"number": str(i), "title": name.replace("_", " ").capitalize(),
                             "image": path, "text": ""})
    return sections
