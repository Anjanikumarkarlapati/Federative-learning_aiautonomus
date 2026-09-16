"""Benchmark harness: all four arms + the PRD 12.3 acceptance check.

Runs centralized / local-only / standard-FedAvg / proposed on the **same fleet
and the same seed** so the comparison is apples to apples, then evaluates the
acceptance criteria of PRD 12.3:

* proposed accuracy >= ``train.target_accuracy`` (default 90%);
* rounds-to-target cut by >= 25% versus standard FedAvg;
* communication cost per round cut by >= 30% versus standard FedAvg, and within
  the ``comm.mb_budget_per_round`` non-functional budget;
* Non-IID robustness no worse than standard FedAvg (lower cross-client accuracy
  dispersion).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..config import FLConfig
from ..data.federated import Fleet, build_fleet
from .baselines import fedavg_config, proposed_config, run_centralized, run_local_only
from .runner import RunResult, run_experiment

ARMS = ("centralized", "local_only", "std_fedavg", "proposed_fl")


@dataclass
class Benchmark:
    cfg: FLConfig
    results: Dict[str, RunResult] = field(default_factory=dict)

    def run(self, arms: Optional[List[str]] = None, fleet: Optional[Fleet] = None,
            echo: bool = True) -> Dict[str, RunResult]:
        arms = list(arms or ARMS)
        fleet = fleet if fleet is not None else build_fleet(self.cfg)
        if "centralized" in arms:
            self.results["centralized"] = run_centralized(self.cfg, fleet, echo=echo)
        if "local_only" in arms:
            self.results["local_only"] = run_local_only(
                self.cfg, fleet, max_clients=min(len(fleet), 12), echo=echo)
        if "std_fedavg" in arms:
            self.results["std_fedavg"] = run_experiment(fedavg_config(self.cfg), fleet, echo)
        if "proposed_fl" in arms:
            self.results["proposed_fl"] = run_experiment(proposed_config(self.cfg), fleet, echo)
        return self.results

    # ------------------------------------------------------------------ report
    def acceptance(self) -> Dict[str, Any]:
        proposed = self.results.get("proposed_fl")
        fedavg = self.results.get("std_fedavg")
        if proposed is None:
            return {"status": "incomplete", "reason": "proposed arm not run"}
        target = self.cfg.train.target_accuracy
        checks: Dict[str, Any] = {
            "accuracy_target": {
                "requirement": f"accuracy >= {target:.2%}",
                "observed": round(proposed.best_eval.accuracy, 4),
                "pass": proposed.best_eval.accuracy >= target,
            }
        }
        if fedavg is not None:
            p_rounds, f_rounds = proposed.rounds_to_target, fedavg.rounds_to_target
            if f_rounds and p_rounds:
                reduction = 1.0 - p_rounds / f_rounds
            elif f_rounds and not p_rounds:
                reduction = -1.0
            elif p_rounds and not f_rounds:
                reduction = 1.0
            else:
                reduction = float("nan")
            checks["convergence_speedup"] = {
                "requirement": "rounds-to-target cut by >= 25% vs standard FedAvg",
                "proposed_rounds": p_rounds, "fedavg_rounds": f_rounds,
                "reduction": None if reduction != reduction else round(reduction, 4),
                "pass": bool(reduction == reduction and reduction >= 0.25),
            }
            saving = 1.0 - proposed.mb_per_round / max(fedavg.mb_per_round, 1e-9)
            checks["communication_saving"] = {
                "requirement": "MB/round cut by >= 30% vs standard FedAvg",
                "proposed_mb": round(proposed.mb_per_round, 4),
                "fedavg_mb": round(fedavg.mb_per_round, 4),
                "reduction": round(saving, 4),
                "pass": bool(saving >= 0.30),
            }
            checks["communication_budget"] = {
                "requirement": f"MB/round <= {self.cfg.comm.mb_budget_per_round}",
                "observed": round(proposed.mb_per_round, 4),
                "pass": bool(proposed.mb_per_round <= self.cfg.comm.mb_budget_per_round),
            }
            p_std = proposed.robustness.get("client_acc_std", 1.0)
            f_std = fedavg.robustness.get("client_acc_std", 1.0)
            checks["non_iid_robustness"] = {
                "requirement": "cross-client accuracy dispersion <= standard FedAvg",
                "proposed_std": p_std, "fedavg_std": f_std,
                "pass": bool(p_std <= f_std + 1e-9),
            }
        checks["privacy"] = {
            "requirement": "no raw sensor data egress in any federated arm",
            "observed": all(not r.privacy.get("raw_data_egress", False)
                            for name, r in self.results.items() if name != "centralized"),
            "pass": all(not r.privacy.get("raw_data_egress", False)
                        for name, r in self.results.items() if name != "centralized"),
        }
        checks["status"] = "PASS" if all(
            v.get("pass", True) for v in checks.values() if isinstance(v, dict)
        ) else "FAIL"
        return checks

    def table(self) -> str:
        """Markdown rendering of the PRD 12.3 comparison table."""
        header = ("| Metric | Centralized | Local-only | Std. FedAvg | Proposed FL |\n"
                  "|---|---|---|---|---|\n")
        def cell(name: str, fn, fmt="{:.4f}", missing="—") -> str:
            result = self.results.get(name)
            if result is None:
                return missing
            try:
                value = fn(result)
            except Exception:
                return missing
            if value is None:
                return missing
            return fmt.format(value) if isinstance(value, (int, float)) else str(value)

        rows = [
            ("Model accuracy (%)", lambda r: 100 * r.best_eval.accuracy, "{:.2f}"),
            ("mAP@0.5", lambda r: r.best_eval.mAP50, "{:.4f}"),
            ("Trajectory ADE", lambda r: r.best_eval.ade, "{:.4f}"),
            ("Rounds to target", lambda r: r.rounds_to_target, "{:d}"),
            ("Rounds run", lambda r: r.rounds_run, "{:d}"),
            ("Comm. cost (MB/round)", lambda r: r.mb_per_round, "{:.3f}"),
            ("Total comm. (MB)", lambda r: r.total_mb, "{:.1f}"),
            ("Client acc. std (Non-IID)", lambda r: r.robustness.get("client_acc_std"), "{:.4f}"),
            ("Worst-client accuracy", lambda r: r.robustness.get("client_acc_min"), "{:.4f}"),
            ("Raw data egress", lambda r: "yes" if r.privacy.get("raw_data_egress") else "no", "{}"),
        ]
        lines = [header]
        for label, fn, fmt in rows:
            cells = " | ".join(cell(arm, fn, fmt) for arm in ARMS)
            lines.append(f"| {label} | {cells} |\n")
        return "".join(lines)

    def report(self, path: Optional[str] = None) -> str:
        acceptance = self.acceptance()
        text = [
            "# FL-AV benchmark report\n",
            f"\nConfig fingerprint: `{self.cfg.fingerprint()}`  ",
            f"\nFleet: N={self.cfg.num_clients}, C={self.cfg.selection.fraction}, "
            f"E={self.cfg.train.local_epochs}, R_max={self.cfg.train.rounds}, "
            f"Dirichlet alpha={self.cfg.data.dirichlet_alpha}\n\n",
            "## Results\n\n", self.table(),
            "\n## Acceptance (PRD 12.3)\n\n",
            f"**{acceptance.get('status')}**\n\n",
        ]
        for key, value in acceptance.items():
            if isinstance(value, dict):
                mark = "PASS" if value.get("pass") else "FAIL"
                text.append(f"- `{key}` — {value.get('requirement')} → **{mark}** "
                            f"({json.dumps({k: v for k, v in value.items() if k not in ('requirement', 'pass')})})\n")
        report = "".join(text)
        out = path or os.path.join(self.cfg.out_dir, "benchmark_report.md")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(report)
        with open(out.replace(".md", ".json"), "w", encoding="utf-8") as fh:
            json.dump({"config": self.cfg.to_dict(),
                       "results": {k: v.summary() for k, v in self.results.items()},
                       "acceptance": acceptance}, fh, indent=2, default=str)
        return report


def run_benchmark(cfg: FLConfig, arms: Optional[List[str]] = None,
                  echo: bool = True) -> Benchmark:
    benchmark = Benchmark(cfg)
    benchmark.run(arms=arms, echo=echo)
    benchmark.report()
    return benchmark


__all__ = ["Benchmark", "run_benchmark", "ARMS"]
