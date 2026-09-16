"""Ablation study (PRD 17): which ingredient is doing the work?

Arms, each differing from the proposed framework by a single switch:

* ``fedavg_baseline``    - plain n_k/n weighting, random selection, no momentum
* ``+quality``           - quality-aware aggregation only
* ``+selection``         - connectivity / trajectory-aware selection only
* ``+momentum``          - server momentum only
* ``proposed``           - all of the above
* ``proposed+dp``        - proposed with differential privacy switched on
* ``proposed+fedprox``   - proposed with the proximal term enabled
* ``proposed+byzantine`` - proposed under attack, robust aggregation on
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from ..config import FLConfig
from ..data.federated import Fleet, build_fleet
from .baselines import fedavg_config, proposed_config
from .runner import run_experiment

ABLATION_ARMS: Dict[str, Dict[str, Any]] = {
    "fedavg_baseline": {},
    "quality_only": {"aggregation": {"quality_aware": True}},
    "selection_only": {"selection": {"trajectory_aware": True, "uptime_threshold": 0.55,
                                     "fairness_k": 12}},
    "momentum_only": {"aggregation": {"server_momentum": 0.9}},
    "adaptive_epochs_only": {"train": {"adaptive_epochs": True}},
}


def run_ablation(cfg: FLConfig, fleet: Optional[Fleet] = None, echo: bool = True,
                 extras: bool = True) -> Dict[str, Any]:
    fleet = fleet if fleet is not None else build_fleet(cfg)
    base = fedavg_config(cfg)
    results: Dict[str, Any] = {}

    for name, overrides in ABLATION_ARMS.items():
        arm = base.merged({**overrides, "name": f"abl_{name}"})
        results[name] = run_experiment(arm, fleet, echo).summary()

    proposed = proposed_config(cfg).merged({"name": "abl_proposed"})
    results["proposed"] = run_experiment(proposed, fleet, echo).summary()

    if extras:
        dp_arm = proposed_config(cfg).merged({"name": "abl_proposed_dp",
                                              "dp": {"enabled": True}})
        results["proposed_dp"] = run_experiment(dp_arm, fleet, echo).summary()

        prox_arm = proposed_config(cfg).merged({"name": "abl_proposed_fedprox",
                                                "train": {"fedprox_mu": 0.01}})
        results["proposed_fedprox"] = run_experiment(prox_arm, fleet, echo).summary()

        byz_ids = list(range(max(1, cfg.num_clients // 10)))
        byz_fleet = build_fleet(cfg, byzantine_ids=byz_ids)
        attacked = proposed_config(cfg).merged({"name": "abl_byzantine_attack"})
        results["byzantine_attack_no_defence"] = run_experiment(
            attacked.merged({"aggregation": {"byzantine_robust": False}}), byz_fleet, echo
        ).summary()
        defended = attacked.merged({"name": "abl_byzantine_defended",
                                    "aggregation": {"byzantine_robust": True}})
        results["byzantine_attack_defended"] = run_experiment(defended, byz_fleet, echo).summary()

    path = os.path.join(cfg.out_dir, "ablation.json")
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    return results


__all__ = ["run_ablation", "ABLATION_ARMS"]
