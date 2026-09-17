#!/usr/bin/env python3
"""Splice a finished benchmark into the README's Results section.

    python scripts/update_readme_results.py runs/benchmark_report.json

Keeps the README honest: the numbers there are always a rendering of a real
`benchmark_report.json`, never hand-typed.
"""
from __future__ import annotations

import json
import os
import sys

MARKER = "<!-- RESULTS -->"


def _history(arm: str, runs_dir: str) -> list:
    """Per-round records, from rounds.jsonl if present else the committed CSV."""
    jsonl = os.path.join(runs_dir, arm, "rounds.jsonl")
    if os.path.exists(jsonl):
        with open(jsonl, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    csv_path = os.path.join(runs_dir, arm, "rounds.csv")
    if not os.path.exists(csv_path):
        return []
    import csv as _csv
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = []
        for row in _csv.DictReader(fh):
            if not row.get("accuracy"):
                continue
            rows.append({k: (float(v) if k != "round" and v not in ("", None)
                             else (int(v) if k == "round" and v else v))
                         for k, v in row.items()})
        return rows


def render(report: dict, runs_dir: str = "runs") -> str:
    cfg = report["config"]
    results = report["results"]
    acceptance = report["acceptance"]
    arms = [("centralized", "Centralized"), ("local_only", "Local-only"),
            ("std_fedavg", "Std. FedAvg"), ("proposed_fl", "Proposed FL")]

    def value(arm: str, key: str, fmt: str = "{:.4f}", scale: float = 1.0,
              nested: str | None = None) -> str:
        result = results.get(arm)
        if result is None:
            return "—"
        raw = result.get(nested, {}).get(key) if nested else result.get(key)
        if raw is None:
            return "—"
        try:
            return fmt.format(float(raw) * scale)
        except (TypeError, ValueError):
            return str(raw)

    histories = {arm: _history(arm, runs_dir) for arm, _ in arms}

    def tail(arm: str, window: int = 10) -> str:
        values = [float(r["accuracy"]) for r in histories.get(arm, []) if "accuracy" in r]
        if not values:
            return "—"
        window = min(window, len(values))
        return "{:.2f}".format(100 * sum(values[-window:]) / window)

    def rounds_to(arm: str, threshold: float) -> str:
        for record in histories.get(arm, []):
            if float(record.get("accuracy", 0.0)) >= threshold:
                return str(int(record["round"]))
        return "—"

    rows = [
        ("Model accuracy, best round (%)", "accuracy", "{:.2f}", 100.0, None),
        ("mAP@0.5", "mAP50", "{:.4f}", 1.0, None),
        ("Trajectory ADE", "ade", "{:.4f}", 1.0, None),
        ("Rounds to target", "rounds_to_target", "{:.0f}", 1.0, None),
        ("Rounds run", "rounds_run", "{:.0f}", 1.0, None),
        ("Comm. cost (MB/round)", "mb_per_round", "{:.3f}", 1.0, None),
        ("Total comm. (MB)", "total_mb", "{:.1f}", 1.0, None),
        ("Client acc. std (Non-IID)", "client_acc_std", "{:.4f}", 1.0, "robustness"),
        ("Worst-client accuracy", "client_acc_min", "{:.4f}", 1.0, "robustness"),
        ("Wall clock (s)", "wall_clock_s", "{:.0f}", 1.0, None),
    ]

    lines = [
        f"Measured on this repository — fleet `N = {cfg['num_clients']}`, "
        f"`C = {cfg['selection']['fraction']}`, `E = {cfg['train']['local_epochs']}`, "
        f"`R_max = {cfg['train']['rounds']}`, Dirichlet `α = {cfg['data']['dirichlet_alpha']}`, "
        f"seed `{cfg['seed']}`. "
        "All four arms share one fleet and one seed.\n\n",
        "| Metric | " + " | ".join(label for _, label in arms) + " |\n",
        "|---|" + "---|" * len(arms) + "\n",
    ]
    for label, key, fmt, scale, nested in rows:
        cells = " | ".join(value(arm, key, fmt, scale, nested) for arm, _ in arms)
        lines.append(f"| {label} | {cells} |\n")
        if key == "accuracy":
            tail_cells = " | ".join(tail(arm) for arm, _ in arms)
            lines.append(f"| Model accuracy, last-10-round mean (%) | {tail_cells} |\n")

    lines.append("\n**Convergence curve** — first round reaching each accuracy "
                 "(federated arms; lower is faster):\n\n")
    lines.append("| Threshold | " + " | ".join(l for _, l in arms) + " |\n")
    lines.append("|---|" + "---|" * len(arms) + "\n")
    for threshold in (0.60, 0.70, 0.75, 0.80):
        cells = " | ".join(rounds_to(arm, threshold) for arm, _ in arms)
        lines.append(f"| {threshold:.0%} | {cells} |\n")

    lines.append(f"\n### Acceptance (PRD 12.3): **{acceptance.get('status')}**\n\n")
    lines.append("| Check | Requirement | Observed | Result |\n|---|---|---|---|\n")
    for key, check in acceptance.items():
        if not isinstance(check, dict):
            continue
        observed = ", ".join(f"{k}={v}" for k, v in check.items()
                             if k not in ("requirement", "pass"))
        mark = "PASS" if check.get("pass") else "FAIL"
        lines.append(f"| `{key}` | {check.get('requirement')} | {observed} | **{mark}** |\n")
    return "".join(lines)


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "runs/benchmark_report.json"
    readme = sys.argv[2] if len(sys.argv) > 2 else "README.md"
    with open(path, encoding="utf-8") as fh:
        report = json.load(fh)
    text = open(readme, encoding="utf-8").read()
    if MARKER not in text:
        print(f"marker {MARKER} not found in {readme}", file=sys.stderr)
        return 1
    head, _, tail = text.partition(MARKER)
    tail = tail.split("---\n\n## 10.", 1)
    rendered = render(report, os.path.dirname(os.path.abspath(path)))
    with open(readme, "w", encoding="utf-8") as fh:
        fh.write(head + rendered + "\n---\n\n## 10." + tail[1])
    print(f"updated {readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
