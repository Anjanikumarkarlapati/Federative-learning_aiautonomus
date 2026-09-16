#!/usr/bin/env python3
"""Reproduce the full experiment suite of the PRD (sections 12 and 17).

    python scripts/run_experiments.py --config configs/default.yaml
    python scripts/run_experiments.py --config configs/quick.yaml --skip-ablation

Writes `runs/benchmark_report.md`, `runs/benchmark_report.json` and
`runs/ablation.json`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedav.config import FLConfig, default_config
from fedav.data.federated import build_fleet
from fedav.sim.ablation import run_ablation
from fedav.sim.benchmark import Benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()

    cfg = FLConfig.from_yaml(args.config) if os.path.exists(args.config) else default_config()
    overrides = {}
    if args.out_dir:
        overrides["out_dir"] = args.out_dir
    if args.seed is not None:
        overrides["seed"] = args.seed
    if overrides:
        cfg = cfg.merged(overrides)

    print(f"config fingerprint: {cfg.fingerprint()}")
    fleet = build_fleet(cfg)              # one fleet shared by every arm
    print(f"fleet: {len(fleet)} vehicles, {fleet.total_samples} training samples, "
          f"{len(fleet.global_test)} global test samples")

    if not args.skip_benchmark:
        benchmark = Benchmark(cfg)
        benchmark.run(fleet=fleet)
        print(benchmark.report())

    if not args.skip_ablation:
        results = run_ablation(cfg, fleet=fleet)
        print(json.dumps({k: {"accuracy": v["accuracy"],
                              "rounds_to_target": v["rounds_to_target"],
                              "mb_per_round": v["mb_per_round"]}
                          for k, v in results.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
