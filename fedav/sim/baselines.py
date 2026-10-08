"""Baselines required by PRD 12.2: centralized, local-only, standard FedAvg.

* **Centralized** - pools every vehicle's shard on a server and trains normally.
  It violates the privacy goal by construction and is reported only as the
  accuracy upper bound; its communication cost is the *raw data* uplink, which
  the table reports for contrast.
* **Local-only** - each vehicle trains alone on its own Non-IID shard and never
  communicates.  Zero communication, maximum privacy, poor generalisation.
* **Standard FedAvg** - the McMahan reference: uniform-random client selection,
  plain ``n_k / n`` weighting, no server momentum, fixed local epochs, and
  full-precision full-weight transfer over unicast links.
* **Proposed** - the PRD's framework: quality-aware weighting, connectivity /
  trajectory-aware selection, adaptive local epochs, server momentum, fp16 +
  gzip + sparsified multicast transport.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import FLConfig
from ..data.federated import Fleet
from ..metrics.evaluation import EvalResult, evaluate, evaluate_fleet
from ..metrics.fleet import evaluate_vehicles, write_vehicle_profiles, write_vehicle_summary
from ..metrics.logger import MetricsLogger
from ..models.perception import build_model, multitask_loss
from ..utils.seeding import seed_everything
from ..utils.tensors import clone_state, to_float_state
from .runner import RunResult


# --------------------------------------------------------------------- presets
def proposed_config(base: FLConfig) -> FLConfig:
    """The framework this PRD specifies."""
    return base.merged({
        "name": "proposed_fl",
        "aggregation": {"quality_aware": True, "server_momentum": 0.9,
                        "normalize_local_steps": True},
        "selection": {"trajectory_aware": True},
        # adaptive_epochs stays off: measured net-negative even with FedNova
        # normalisation (README section 9).
        "train": {"adaptive_epochs": False},
        "aggregation": {"normalize_local_steps": True},
        "comm": {"fp16": True, "send_deltas": True, "multicast": True,
                 "topk_ratio": 0.25, "adaptive_sparsity": True},
    })


def fedavg_config(base: FLConfig) -> FLConfig:
    """McMahan-style FedAvg: no quality weighting, no momentum, uncompressed."""
    return base.merged({
        "name": "std_fedavg",
        # plain weighted mean of raw deltas: FedNova normalisation is part of
        # the proposed method, not of McMahan FedAvg
        "aggregation": {"quality_aware": False, "server_momentum": 0.0,
                        "byzantine_robust": False, "normalize_local_steps": False},
        "selection": {"trajectory_aware": False, "uptime_threshold": 0.0,
                      "fairness_k": 10**6},
        "train": {"adaptive_epochs": False, "fedprox_mu": 0.0},
        "comm": {"fp16": False, "send_deltas": False, "broadcast_delta": False,
                 "multicast": False, "topk_ratio": 1.0, "adaptive_sparsity": False},
        "dp": {"enabled": False, "secure_aggregation": False},
    })


def _train_epochs(model, loader, cfg: FLConfig, epochs: int, lr: float,
                  device: str = "cpu") -> float:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=cfg.train.momentum,
                                weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1)) \
        if cfg.train.cosine_decay else None
    model.to(device).train()
    last = 0.0
    for _ in range(epochs):
        total, steps = 0.0, 0
        for images, labels, boxes, trajectories in loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images.to(device))
            loss, _ = multitask_loss(outputs, labels.to(device), boxes.to(device),
                                     trajectories.to(device), cfg.model)
            loss.backward()
            if cfg.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            total += float(loss.detach()); steps += 1
        if scheduler is not None:
            scheduler.step()
        last = total / max(steps, 1)
    return last


def run_centralized(cfg: FLConfig, fleet: Fleet, epochs: Optional[int] = None,
                    echo: bool = True) -> RunResult:
    """Upper bound: one model, all data pooled on the server (PRD 12.2a)."""
    cfg = cfg.merged({"name": "centralized"})
    seed_everything(cfg.seed)
    started = time.perf_counter()
    logger = MetricsLogger(cfg.name, cfg.out_dir, cfg.fingerprint(), echo=echo)
    logger.write_config(cfg.to_dict())
    write_vehicle_profiles(logger.run_dir, fleet)
    epochs = epochs or cfg.train.rounds
    model = build_model(cfg.model, cfg.data.num_classes, cfg.data.traj_horizon, seed=cfg.seed)
    dataset = fleet.pooled_train(augment=cfg.data.augment, seed=cfg.seed)
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True)
    test_loader = fleet.test_loader()

    best_eval, best_state = EvalResult(), clone_state(to_float_state(model.state_dict()))
    # Raw-data uplink the centralized design would actually need, for contrast.
    raw_mb = float(sum(v.train.images.numel() for v in fleet) * 4) / (1024 ** 2)
    for epoch in range(1, epochs + 1):
        loss = _train_epochs(model, loader, cfg, 1, cfg.train.lr *
                             (0.5 * (1 + np.cos(np.pi * (epoch - 1) / max(epochs - 1, 1)))
                              if cfg.train.cosine_decay else 1.0), cfg.device)
        result = evaluate(model, test_loader, cfg.model, cfg.data.num_classes, cfg.device)
        if result.accuracy > best_eval.accuracy:
            best_eval = result
            best_state = clone_state(to_float_state(model.state_dict()))
        logger.log({"round": epoch, **result.as_dict(), "train_loss": round(loss, 5),
                    "clients_selected": len(fleet), "clients_reported": len(fleet),
                    "mb_round": 0.0, "mb_cumulative": round(raw_mb, 3)})
    model.load_state_dict(best_state, strict=False)
    robustness = evaluate_fleet(model, fleet, cfg.model, cfg.data.num_classes, cfg.device)
    result = RunResult(
        name=cfg.name, config=cfg.to_dict(), best_state=best_state,
        final_eval=best_eval, best_eval=best_eval, rounds_run=epochs,
        rounds_to_target=logger.rounds_to_target(cfg.train.target_accuracy),
        mb_per_round=0.0, total_mb=raw_mb, converged=True,
        history=list(logger.records), robustness=robustness.as_dict(),
        privacy={"mode": "none - raw sensor data centralised",
                 "raw_data_egress": True, "raw_data_mb": round(raw_mb, 2)},
        selection={"note": "all vehicles, every epoch"},
        wall_clock_s=time.perf_counter() - started,
    )
    logger.write_summary(result.summary())
    write_vehicle_summary(
        logger.run_dir, evaluate_vehicles(model, fleet, cfg, cfg.device),
        note="final_* = the pooled-data model on each vehicle's own test split")
    return result


def run_local_only(cfg: FLConfig, fleet: Fleet, epochs: Optional[int] = None,
                   max_clients: Optional[int] = None, echo: bool = True) -> RunResult:
    """Lower bound: every vehicle trains alone, nothing is shared (PRD 12.2b)."""
    cfg = cfg.merged({"name": "local_only"})
    seed_everything(cfg.seed)
    started = time.perf_counter()
    logger = MetricsLogger(cfg.name, cfg.out_dir, cfg.fingerprint(), echo=echo)
    logger.write_config(cfg.to_dict())
    write_vehicle_profiles(logger.run_dir, fleet)
    fleet.reset_augmentation()                   # independent of earlier arms
    epochs = epochs or max(1, cfg.train.rounds * cfg.train.local_epochs // 4)
    test_loader = fleet.test_loader()

    vehicles = list(fleet)
    if max_clients is not None and max_clients < len(vehicles):
        step = len(vehicles) / max_clients
        vehicles = [vehicles[int(i * step)] for i in range(max_clients)]

    accuracies, maps, ades, fdes, losses = [], [], [], [], []
    per_client: Dict[int, float] = {}
    own_test: Dict[int, Dict[str, float]] = {}
    best_state, best_accuracy = None, -1.0
    for index, vehicle in enumerate(vehicles, start=1):
        model = build_model(cfg.model, cfg.data.num_classes, cfg.data.traj_horizon,
                            seed=cfg.seed + vehicle.client_id)
        loader = vehicle.loader(cfg.train.batch_size, seed=cfg.seed + vehicle.client_id)
        _train_epochs(model, loader, cfg, epochs, cfg.train.lr, cfg.device)
        result = evaluate(model, test_loader, cfg.model, cfg.data.num_classes, cfg.device)
        accuracies.append(result.accuracy); maps.append(result.mAP50)
        ades.append(result.ade); fdes.append(result.fde); losses.append(result.loss)
        per_client[vehicle.client_id] = result.accuracy
        own_test.update(evaluate_vehicles(model, [vehicle], cfg, cfg.device))
        if result.accuracy > best_accuracy:
            best_accuracy = result.accuracy
            best_state = clone_state(to_float_state(model.state_dict()))
        logger.log({"round": index, **result.as_dict(), "clients_selected": 1,
                    "clients_reported": 1, "mb_round": 0.0, "mb_cumulative": 0.0,
                    "client_id": vehicle.client_id})

    mean = EvalResult(
        accuracy=float(np.mean(accuracies)), mAP50=float(np.mean(maps)),
        ade=float(np.mean(ades)), fde=float(np.mean(fdes)), loss=float(np.mean(losses)),
        balanced_accuracy=float(np.mean(accuracies)), macro_f1=float(np.mean(accuracies)),
        samples=len(fleet.global_test),
    )
    values = np.array(accuracies)
    result = RunResult(
        name=cfg.name, config=cfg.to_dict(), best_state=best_state or {},
        final_eval=mean, best_eval=mean, rounds_run=epochs,
        rounds_to_target=None, mb_per_round=0.0, total_mb=0.0, converged=True,
        history=list(logger.records),
        robustness={"client_acc_mean": round(float(values.mean()), 5),
                    "client_acc_std": round(float(values.std()), 5),
                    "client_acc_min": round(float(values.min()), 5),
                    "client_acc_p10": round(float(np.percentile(values, 10)), 5),
                    "client_acc_spread": round(float(values.max() - values.min()), 5)},
        privacy={"mode": "no communication at all", "raw_data_egress": False},
        selection={"clients_trained": len(vehicles)},
        wall_clock_s=time.perf_counter() - started,
    )
    logger.write_summary({**result.summary(), "per_client_accuracy": per_client})
    write_vehicle_summary(
        logger.run_dir, own_test,
        note="final_* = each vehicle's own locally trained model on its own test split")
    return result


__all__ = ["run_centralized", "run_local_only", "proposed_config", "fedavg_config"]
