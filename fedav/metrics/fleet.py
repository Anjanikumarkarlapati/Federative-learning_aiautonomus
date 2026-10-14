"""Per-vehicle telemetry: who each car is, what it did every round, how it ends up.

Three files land next to ``rounds.jsonl``:

``vehicles.json``
    The fleet as built: one record per vehicle with its region, weather,
    calibration, shard sizes, link uptime, compute, sensor quality, similarity
    to the fleet label mix, and its class histogram.  Written once, at start.
``vehicle_rounds.jsonl``
    One record per (round, vehicle): whether it was selected, whether its update
    arrived (or was dropped / a straggler / corrupt / rejected), its ``q_k``,
    aggregation weight ``alpha_k``, local epochs and loss, and the current global
    model's accuracy on that vehicle's own test split.  Appended round by round,
    so a live run can be watched.
``vehicle_summary.json``
    The final per-vehicle scorecard: participation counts plus the best model's
    accuracy / mAP / ADE / FDE / loss on each vehicle's own test split.

Everything here is simulator-side bookkeeping.  None of it crosses the
simulated V2I link, so it does not change what the RSU learns about a vehicle.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
from torch import nn

from ..data.quality import hellinger_distance
from ..data.scenes import CLASS_NAMES
from .evaluation import evaluate

VEHICLES_FILE = "vehicles.json"
VEHICLE_ROUNDS_FILE = "vehicle_rounds.jsonl"
VEHICLE_SUMMARY_FILE = "vehicle_summary.json"

STATUSES = ("not_selected", "reported", "dropped", "straggler", "corrupt", "rejected")


def vehicle_profiles(fleet) -> List[Dict[str, Any]]:
    """The fleet as built - exact, deterministic from the config and seed."""
    names = list(CLASS_NAMES)
    out = []
    for vehicle in fleet:
        p = vehicle.profile
        hist = np.asarray(p.label_histogram, dtype=np.float64)
        out.append({
            "client_id": int(p.client_id),
            "region": p.region,
            "weather": p.weather,
            "calibrated": bool(p.calibrated),
            "byzantine": bool(p.byzantine),
            "n_train": len(vehicle.train),
            "n_val": len(vehicle.val),
            "n_test": len(vehicle.test),
            "link_uptime": round(float(p.link_uptime), 5),
            "compute_capability": round(float(p.compute_capability), 5),
            "sensor_quality": round(float(p.quality_signal), 5),
            "label_similarity": round(
                1.0 - hellinger_distance(hist, fleet.reference_histogram), 5),
            "dominant_class": names[int(hist.argmax())] if hist.sum() > 0 else None,
            "label_histogram": {names[i]: round(float(hist[i]), 5)
                                for i in range(min(len(names), len(hist)))},
        })
    return out


def write_vehicle_profiles(run_dir: str, fleet) -> str:
    path = os.path.join(run_dir, VEHICLES_FILE)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(vehicle_profiles(fleet), fh, indent=1)
    return path


def evaluate_vehicles(model: nn.Module, fleet, cfg, device: str = "cpu",
                      batch_size: int = 128,
                      model_for: Optional[Callable[[int], nn.Module]] = None
                      ) -> Dict[int, Dict[str, float]]:
    """Full metrics of ``model`` (or ``model_for(client_id)``) on each vehicle's
    own held-out test split."""
    out: Dict[int, Dict[str, float]] = {}
    for vehicle in fleet:
        m = model_for(vehicle.client_id) if model_for else model
        if m is None:
            continue
        result = evaluate(m, vehicle.loader(batch_size, split="test"), cfg.model,
                          cfg.data.num_classes, device)
        out[vehicle.client_id] = {
            "accuracy": round(result.accuracy, 5),
            "balanced_accuracy": round(result.balanced_accuracy, 5),
            "mAP50": round(result.mAP50, 5),
            "ade": round(result.ade, 5),
            "fde": round(result.fde, 5),
            "loss": round(result.loss, 5),
            "test_samples": result.samples,
        }
    return out


def per_vehicle_accuracy(model: nn.Module, fleet, cfg, device: str = "cpu") -> Dict[int, float]:
    """Accuracy only - the cheap per-round variant."""
    return {cid: m["accuracy"] for cid, m in evaluate_vehicles(model, fleet, cfg, device).items()}


class VehicleTracker:
    """Appends one record per vehicle per round and builds the final scorecard."""

    def __init__(self, run_dir: str, fleet):
        self.run_dir = run_dir
        self.ids = [v.client_id for v in fleet]
        self.path = os.path.join(run_dir, VEHICLE_ROUNDS_FILE)
        self.counts = {cid: {s: 0 for s in STATUSES} for cid in self.ids}
        self.q_sum = {cid: 0.0 for cid in self.ids}
        self.alpha_sum = {cid: 0.0 for cid in self.ids}
        self.last_round = {cid: None for cid in self.ids}
        open(self.path, "w").close()

    def log_round(self, round_id: int, events: Dict[int, Dict[str, Any]],
                  accuracy: Optional[Dict[int, float]] = None) -> None:
        """``events`` maps client_id -> {"status": ..., other fields} for the
        vehicles that were selected; every other vehicle is ``not_selected``."""
        accuracy = accuracy or {}
        with open(self.path, "a", encoding="utf-8") as fh:
            for cid in self.ids:
                event = dict(events.get(cid, {"status": "not_selected"}))
                status = event["status"]
                self.counts[cid][status] += 1
                if status == "reported":
                    self.q_sum[cid] += float(event.get("q_k", 0.0))
                    self.alpha_sum[cid] += float(event.get("alpha", 0.0))
                    self.last_round[cid] = round_id
                record = {"round": round_id, "client_id": cid, **event}
                if cid in accuracy:
                    record["accuracy"] = accuracy[cid]
                fh.write(json.dumps(record, default=float) + "\n")

    def participation(self) -> Dict[int, Dict[str, Any]]:
        out = {}
        for cid in self.ids:
            c = self.counts[cid]
            selected = sum(v for k, v in c.items() if k != "not_selected")
            reported = c["reported"]
            out[cid] = {
                "times_selected": selected,
                "times_reported": reported,
                "times_dropped": c["dropped"],
                "times_straggler": c["straggler"],
                "times_corrupt": c["corrupt"],
                "times_rejected": c["rejected"],
                "delivery_rate": round(reported / selected, 4) if selected else None,
                "mean_q": round(self.q_sum[cid] / reported, 5) if reported else None,
                "mean_alpha": round(self.alpha_sum[cid] / reported, 5) if reported else None,
                "last_contribution_round": self.last_round[cid],
            }
        return out


def write_vehicle_summary(run_dir: str, final: Dict[int, Dict[str, float]],
                          participation: Optional[Dict[int, Dict[str, Any]]] = None,
                          note: str = "") -> str:
    """``final`` is evaluate_vehicles() of the model this run reports."""
    ids: Iterable[int] = sorted(set(final) | set(participation or {}))
    rows = []
    for cid in ids:
        rows.append({"client_id": int(cid), **(participation or {}).get(cid, {}),
                     **{f"final_{k}": v for k, v in final.get(cid, {}).items()}})
    path = os.path.join(run_dir, VEHICLE_SUMMARY_FILE)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"note": note, "vehicles": rows}, fh, indent=1)
    return path


__all__ = ["vehicle_profiles", "write_vehicle_profiles", "evaluate_vehicles",
           "per_vehicle_accuracy", "VehicleTracker", "write_vehicle_summary",
           "VEHICLES_FILE", "VEHICLE_ROUNDS_FILE", "VEHICLE_SUMMARY_FILE", "STATUSES"]
