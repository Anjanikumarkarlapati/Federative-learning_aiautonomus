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


def render(report: dict) -> str:
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

    rows = [
        ("Model accuracy (%)", "accuracy", "{:.2f}", 100.0, None),
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
        f"seed `{cfg['seed']}`, config fingerprint `{report.get('fingerprint', '')}`. "
        "All four arms share one fleet and one seed.\n\n",
        "| Metric | " + " | ".join(label for _, label in arms) + " |\n",
        "|---|" + "---|" * len(arms) + "\n",
    ]
    for label, key, fmt, scale, nested in rows:
        cells = " | ".join(value(arm, key, fmt, scale, nested) for arm, _ in arms)
        lines.append(f"| {label} | {cells} |\n")

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
    rendered = render(report)
    with open(readme, "w", encoding="utf-8") as fh:
        fh.write(head + rendered + "\n---\n\n## 10." + tail[1])
    print(f"updated {readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
