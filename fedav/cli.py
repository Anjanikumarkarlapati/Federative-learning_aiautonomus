"""Command-line entry point.

    python -m fedav.cli run        --config configs/default.yaml
    python -m fedav.cli benchmark  --config configs/default.yaml
    python -m fedav.cli ablation   --config configs/quick.yaml
    python -m fedav.cli info       --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from .config import FLConfig, default_config


def _parse_overrides(pairs: List[str]) -> Dict[str, Any]:
    """``--set train.rounds=10 aggregation.quality_aware=false`` -> nested dict."""
    from .config import _coerce

    out: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"bad --set value {pair!r}, expected key=value")
        key, value = pair.split("=", 1)
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce(value)
    return out


def load_config(args: argparse.Namespace) -> FLConfig:
    cfg = FLConfig.from_yaml(args.config) if args.config else default_config()
    overrides = _parse_overrides(args.set or [])
    if args.name:
        overrides["name"] = args.name
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.out_dir:
        overrides["out_dir"] = args.out_dir
    return cfg.merged(overrides) if overrides else cfg


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser("fedav", description="Federated learning for collaborative AV training")
    parser.add_argument("command", choices=["run", "benchmark", "ablation", "info"])
    parser.add_argument("--config", default=None, help="YAML config path")
    parser.add_argument("--name", default=None, help="run name")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", dest="out_dir", default=None)
    parser.add_argument("--arms", nargs="*", default=None,
                        help="benchmark arms: centralized local_only std_fedavg proposed_fl")
    parser.add_argument("--set", nargs="*", default=None, metavar="KEY=VALUE",
                        help="dotted config overrides, e.g. train.rounds=10")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args)
    echo = not args.quiet

    if args.command == "info":
        from .data.federated import build_fleet
        from .models.perception import build_model
        from .utils.tensors import num_parameters

        model = build_model(cfg.model, cfg.data.num_classes, cfg.data.traj_horizon, seed=cfg.seed)
        fleet = build_fleet(cfg)
        print(json.dumps({
            "config_fingerprint": cfg.fingerprint(),
            "model_parameters": num_parameters(model),
            "clients": len(fleet),
            "total_train_samples": fleet.total_samples,
            "global_test_samples": len(fleet.global_test),
            "clients_per_round": max(1, int(cfg.selection.fraction * cfg.num_clients + 0.999)),
            "regions": sorted({v.profile.region for v in fleet}),
            "weathers": sorted({v.profile.weather for v in fleet}),
            "mean_quality_signal": round(
                sum(v.profile.quality_signal for v in fleet) / len(fleet), 4),
        }, indent=2))
        return 0

    if args.command == "run":
        from .sim.runner import run_experiment
        result = run_experiment(cfg, echo=echo)
        print(json.dumps(result.summary(), indent=2))
        return 0

    if args.command == "benchmark":
        from .sim.benchmark import Benchmark
        benchmark = Benchmark(cfg)
        benchmark.run(arms=args.arms, echo=echo)
        print(benchmark.report())
        return 0 if benchmark.acceptance().get("status") == "PASS" else 1

    if args.command == "ablation":
        from .sim.ablation import run_ablation
        study = run_ablation(cfg, echo=echo)
        print(json.dumps(study, indent=2, default=str))
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
