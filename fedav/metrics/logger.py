"""MetricsLogger - per-round telemetry (PRD 13, FR-08).

Every round writes one JSON line with accuracy, mAP, communication cost,
selected-client count, aggregation weights and privacy spend, plus a CSV view
for spreadsheets and a final ``summary.json``.  Runs are keyed by the config
fingerprint so a fixed-seed re-run is byte-comparable (PRD 15 / 17 regression
test).  MLflow / Weights & Biases (PRD 14) can be attached through
``add_sink``; nothing in the framework depends on them being installed.
"""
from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

CSV_COLUMNS = [
    "round", "accuracy", "mAP50", "ade", "fde", "loss", "clients_selected",
    "clients_reported", "mb_round", "mb_cumulative", "delta_norm",
    "client_acc_mean", "client_acc_std", "client_acc_min", "epsilon_spent",
    "lr", "round_seconds",
]


@dataclass
class MetricsLogger:
    run_name: str
    out_dir: str = "runs"
    config_fingerprint: str = ""
    echo: bool = True
    records: List[Dict[str, Any]] = field(default_factory=list)
    sinks: List[Callable[[Dict[str, Any]], None]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.run_dir = os.path.join(self.out_dir, self.run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.jsonl_path = os.path.join(self.run_dir, "rounds.jsonl")
        self.csv_path = os.path.join(self.run_dir, "rounds.csv")
        self._started = time.time()
        open(self.jsonl_path, "w").close()
        with open(self.csv_path, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(CSV_COLUMNS)

    def add_sink(self, sink: Callable[[Dict[str, Any]], None]) -> None:
        """Attach an external tracker, e.g. ``lambda rec: mlflow.log_metrics(rec)``."""
        self.sinks.append(sink)

    def log(self, record: Dict[str, Any]) -> Dict[str, Any]:
        record = dict(record)
        record.setdefault("run", self.run_name)
        record.setdefault("wall_clock_s", round(time.time() - self._started, 3))
        self.records.append(record)
        with open(self.jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=float) + "\n")
        with open(self.csv_path, "a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([record.get(col, "") for col in CSV_COLUMNS])
        for sink in self.sinks:
            try:
                sink(record)
            except Exception as exc:                      # never fail a run on telemetry
                print(f"[metrics] sink error: {exc}")
        if self.echo:
            print(self.format_line(record))
        return record

    @staticmethod
    def format_line(record: Dict[str, Any]) -> str:
        def fmt(key: str, spec: str = ".4f", default: float = float("nan")) -> str:
            value = record.get(key, default)
            try:
                return format(float(value), spec)
            except (TypeError, ValueError):
                return str(value)
        return (
            f"[{record.get('run','run'):>16}] r{int(record.get('round', 0)):>3} "
            f"acc={fmt('accuracy')} mAP={fmt('mAP50')} ade={fmt('ade','.3f')} "
            f"loss={fmt('loss','.4f')} |S|={record.get('clients_reported','-')}"
            f"/{record.get('clients_selected','-')} "
            f"MB={fmt('mb_round','.3f')} ||dW||={fmt('delta_norm','.4f')}"
        )

    def write_summary(self, summary: Dict[str, Any]) -> str:
        summary = dict(summary)
        summary["config_fingerprint"] = self.config_fingerprint
        summary["run"] = self.run_name
        path = os.path.join(self.run_dir, "summary.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=float)
        return path

    def write_config(self, config: Dict[str, Any]) -> str:
        path = os.path.join(self.run_dir, "config.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2, default=str)
        return path

    def best(self, key: str = "accuracy") -> Optional[Dict[str, Any]]:
        valid = [r for r in self.records if key in r]
        return max(valid, key=lambda r: r[key]) if valid else None

    def rounds_to_target(self, target: float, key: str = "accuracy") -> Optional[int]:
        """Convergence-rounds metric of PRD 12.1: first round reaching ``target``."""
        for record in self.records:
            if float(record.get(key, 0.0)) >= target:
                return int(record["round"])
        return None


__all__ = ["MetricsLogger", "CSV_COLUMNS"]
