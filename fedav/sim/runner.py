"""SimRunner - orchestrates federated rounds (PRD 9, 13).

Implements the reference loop verbatim::

    1  server initialises w0 and broadcasts it
    2  for r = 1..R:
    3      S <- select connected / trajectory-viable vehicles, |S| = ceil(C N)
    4      for k in S (in parallel):
    5          receive w(r)
    6          for e = 1..E_k:  w_k <- w_k - eta grad F_k(w_k)
    7          (optional) clip + DP noise; mask; encrypt
    8          send w_k(r+1), n_k, q_k
    9      w(r+1) = sum_k [(q_k n_k) / sum_j (q_j n_j)] w_k(r+1)
    10     if ||w(r+1) - w(r)|| <= eps: break
    11     broadcast w(r+1)
    12 return w*

with the PRD's fault handling layered on: dropped uplinks, straggler timeouts,
checksum failures and Byzantine norm filtering all remove a vehicle from the
round instead of stalling or corrupting it, and the best checkpoint is kept so a
run that hits ``R_max`` without converging still returns a usable model
(PRD 16).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from collections import OrderedDict

from ..client.compression import BandwidthController, TopKCompressor
from ..client.serializer import ChecksumError, UpdateSerializer
from ..client.trainer import ClientTrainer, ClientUpdate
from ..config import FLConfig
from ..data.federated import Fleet, build_fleet
from ..metrics.evaluation import EvalResult, evaluate, evaluate_fleet
from ..metrics.logger import MetricsLogger
from ..models.perception import build_model
from ..privacy.dp import PrivacyAccountant
from ..server.aggregator import ContributedUpdate, FedAggregator, has_converged
from ..server.secure import SecureAggregationMasks, SecureChannel
from ..server.selection import ClientSelector
from ..utils.seeding import rng as make_rng, seed_everything
from ..utils.tensors import (StateDict, clone_state, state_add, state_norm,
                             state_numel, state_sub, state_zeros_like,
                             to_float_state, weighted_average)


@dataclass
class RunResult:
    """Everything a run reports back (PRD 12)."""

    name: str
    config: Dict[str, Any]
    best_state: StateDict
    final_eval: EvalResult
    best_eval: EvalResult
    rounds_run: int
    rounds_to_target: Optional[int]
    mb_per_round: float
    total_mb: float
    converged: bool
    history: List[Dict[str, Any]] = field(default_factory=list)
    robustness: Dict[str, float] = field(default_factory=dict)
    privacy: Dict[str, Any] = field(default_factory=dict)
    selection: Dict[str, float] = field(default_factory=dict)
    wall_clock_s: float = 0.0

    def summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "accuracy": round(self.best_eval.accuracy, 5),
            "final_accuracy": round(self.final_eval.accuracy, 5),
            "mAP50": round(self.best_eval.mAP50, 5),
            "ade": round(self.best_eval.ade, 5),
            "fde": round(self.best_eval.fde, 5),
            "rounds_run": self.rounds_run,
            "rounds_to_target": self.rounds_to_target,
            "mb_per_round": round(self.mb_per_round, 4),
            "total_mb": round(self.total_mb, 3),
            "converged": self.converged,
            "robustness": self.robustness,
            "privacy": self.privacy,
            "selection": self.selection,
            "wall_clock_s": round(self.wall_clock_s, 1),
        }


class SimRunner:
    """Runs one federated experiment end to end."""

    def __init__(self, cfg: FLConfig, fleet: Optional[Fleet] = None,
                 logger: Optional[MetricsLogger] = None, echo: bool = True):
        self.cfg = cfg
        seed_everything(cfg.seed)
        self.fleet = fleet if fleet is not None else build_fleet(cfg)
        self.model = build_model(cfg.model, cfg.data.num_classes,
                                 cfg.data.traj_horizon, seed=cfg.seed)
        self.global_state: StateDict = to_float_state(self.model.state_dict())
        self.trainer = ClientTrainer(cfg, self.model, cfg.device)
        self.selector = ClientSelector(cfg.selection, len(self.fleet), seed=cfg.seed)
        self.aggregator = FedAggregator(cfg.aggregation)
        self.serializer = UpdateSerializer(cfg.comm)
        self.channel = SecureChannel()
        self.compressor = TopKCompressor(ratio=cfg.comm.topk_ratio,
                                         error_feedback=cfg.comm.error_feedback,
                                         min_ratio=cfg.comm.min_topk_ratio)
        self.bandwidth = BandwidthController(
            cfg.comm.mb_budget_per_round if cfg.comm.adaptive_sparsity else 0.0,
            initial_ratio=cfg.comm.topk_ratio, min_ratio=cfg.comm.min_topk_ratio,
        )
        self.logger = logger or MetricsLogger(cfg.name, cfg.out_dir,
                                              cfg.fingerprint(), echo=echo)
        self._last_sync: Dict[int, int] = {}      # client -> round whose model it holds
        self._mask_scale = 1e-2                   # matched to the update magnitude
        self.accountant: Optional[PrivacyAccountant] = None
        if cfg.dp.enabled:
            sampling_rate = min(1.0, cfg.selection.fraction)
            self.accountant = PrivacyAccountant.calibrated(
                cfg.dp.epsilon, cfg.dp.delta, sampling_rate,
                cfg.train.rounds, cfg.dp.clip_norm,
            )

    # ------------------------------------------------------------------ round
    def _broadcast(self, state: StateDict, previous: Optional[StateDict],
                   selected: List[int], round_id: int) -> int:
        """Server -> vehicles (FR-06), with honest byte accounting.

        A vehicle can only reconstruct ``w^(r)`` from a delta if it already holds
        ``w^(r-1)``; one that sat out the previous round needs the full model.
        Selection means most rounds carry a mix, so the two groups are costed
        separately.  Under ``comm.multicast`` each group is one RSU transmission
        into the coverage cell; otherwise every vehicle is billed a unicast copy.
        """
        delta_group, full_group = [], []
        for client_id in selected:
            if (self.cfg.comm.broadcast_delta and previous is not None
                    and self._last_sync.get(client_id) == round_id - 1):
                delta_group.append(client_id)
            else:
                full_group.append(client_id)

        total = 0
        if delta_group:
            blob = self.serializer.pack(state_sub(state, previous),
                                        {"round_id": round_id, "kind": "global_delta"})
            size, _ = self.channel.send_down(blob, 1 if self.cfg.comm.multicast
                                             else len(delta_group))
            total += size
        if full_group:
            blob = self.serializer.pack(state, {"round_id": round_id, "kind": "global"})
            size, _ = self.channel.send_down(blob, 1 if self.cfg.comm.multicast
                                             else len(full_group))
            total += size
        for client_id in selected:
            self._last_sync[client_id] = round_id
        return total

    def _collect(self, selected: List[int], round_id: int
                 ) -> Tuple[List[ContributedUpdate], List[ClientUpdate], Dict[str, int]]:
        """Local training + uplink for every selected vehicle."""
        dp_sigma = 0.0
        if self.accountant is not None:
            dp_sigma = (self.accountant.client_sigma(len(selected))
                        if self.cfg.dp.secure_aggregation else self.accountant.sigma)

        contributed: List[ContributedUpdate] = []
        raw_updates: List[ClientUpdate] = []
        stats = {"dropped": 0, "straggler": 0, "corrupt": 0}
        generator = make_rng(self.cfg.seed, round_id, 55)

        for client_id in selected:
            vehicle = self.fleet[client_id]
            update = self.trainer.train(self.global_state, vehicle, round_id,
                                        self.fleet.reference_histogram, dp_sigma)
            if update.duration_s > self.cfg.selection.round_timeout_s:
                stats["straggler"] += 1            # PRD 16: past the round window
                continue
            payload = self.compressor.compress(client_id, update.delta)
            if not self.cfg.comm.send_deltas:
                # Classic FedAvg puts the full local weights on the wire; the
                # server differences them against w^(r) to recover the update.
                payload = state_add(self.global_state, payload, alpha=1.0)
            blob = self.serializer.pack(payload, update.meta())
            delivered, _, _ = self.channel.send_up(blob, vehicle.profile.link_uptime, generator)
            if not delivered:
                stats["dropped"] += 1              # PRD 16: client drops mid-round
                continue
            try:
                delta, meta = self.serializer.unpack(blob)
                if not self.cfg.comm.send_deltas:
                    delta = state_sub(delta, self.global_state)
            except ChecksumError:
                stats["corrupt"] += 1
                continue
            contributed.append(ContributedUpdate(
                client_id=int(meta["client_id"]), delta=delta, n_k=int(meta["n_k"]),
                q_k=float(meta["q_k"]), steps=int(meta.get("steps", 1))))
            raw_updates.append(update)
        return contributed, raw_updates, stats

    def _secure_sum(self, updates: List[ContributedUpdate], alphas: np.ndarray,
                    round_id: int) -> Tuple[StateDict, bool]:
        """Masked weighted sum: the RSU sees only the total (PRD 8.7).

        Each vehicle uploads ``alpha_k dW_k + mask_k``; the pairwise masks cancel
        in the sum, so the server recovers ``sum_k alpha_k dW_k`` exactly and
        nothing about any individual ``dW_k``.

        The mask magnitude is set from the round's own reported update norms -
        already part of the PRD 8.6 message schema, and already exchanged in the
        first phase that establishes the ``alpha`` normaliser, so this costs no
        extra round trip and reveals no individual update beyond the scalar norm
        the protocol already carries. Scale-matching is not cosmetic: a mask fixed
        at 1.0 against a converged update of norm 0.01 is a 10^4 : 1 cancellation
        that blows past the verification tolerance (it did, in 56 of 60 rounds,
        before this was fixed).
        """
        participants = [u.client_id for u in updates]
        expected = weighted_average([u.delta for u in updates], alphas.tolist())
        scale = SecureAggregationMasks.scale_for(
            float(np.dot(alphas, [state_norm(u.delta) for u in updates])),
            state_numel(expected),
        )
        self._mask_scale = scale
        masks = SecureAggregationMasks(round_id=round_id, participants=participants,
                                       secret=self.cfg.seed, scale=scale)
        total = OrderedDict((k, torch.zeros(v.shape, dtype=torch.float64))
                            for k, v in expected.items())
        for update, alpha in zip(updates, alphas):
            masked = masks.apply({k: v * float(alpha) for k, v in update.delta.items()},
                                 update.client_id)
            for key in total:
                total[key] += masked[key]
        recovered = OrderedDict((k, v.to(torch.float32)) for k, v in total.items())
        residual = state_norm(state_sub(recovered, expected))
        return recovered, bool(residual <= 1e-3 * max(state_norm(expected), 1e-12))

    # -------------------------------------------------------------------- run
    def run(self) -> RunResult:
        cfg = self.cfg
        started = time.perf_counter()
        self.logger.write_config(cfg.to_dict())
        previous_broadcast: Optional[StateDict] = None
        best_eval = EvalResult()
        best_state = clone_state(self.global_state)
        converged = False
        rounds_run = 0
        total_mb = 0.0
        test_loader = self.fleet.test_loader()

        for round_id in range(1, cfg.train.rounds + 1):
            round_started = time.perf_counter()
            self.channel.reset()
            rounds_run = round_id

            # (3) client selection
            selection = self.selector.select(self.fleet.profiles, round_id)
            if not selection.selected:
                continue
            # (5) broadcast the current global model
            self._broadcast(self.global_state, previous_broadcast,
                            selection.selected, round_id)
            previous_broadcast = clone_state(self.global_state)

            # (4-8) local training + uplink
            contributed, raw_updates, drop_stats = self._collect(selection.selected, round_id)
            if not contributed:
                self.logger.log({"round": round_id, "clients_selected": len(selection.selected),
                                 "clients_reported": 0, "note": "no updates received",
                                 **{f"drop_{k}": v for k, v in drop_stats.items()}})
                continue

            rejected: List[int] = []
            if cfg.aggregation.byzantine_robust:
                contributed, rejected = FedAggregator.norm_filter(contributed)

            # (9) quality-aware FedAvg, optionally through secure aggregation
            aggregated, alphas = self.aggregator.combine(contributed)
            secure_ok = False
            if cfg.dp.secure_aggregation and not cfg.aggregation.byzantine_robust:
                aggregated, secure_ok = self._secure_sum(contributed, alphas, round_id)
            previous_state = self.global_state
            self.global_state, report = self.aggregator.apply(
                previous_state, aggregated, alphas,
                [u.client_id for u in contributed], rejected,
            )
            if self.accountant is not None:
                self.accountant.step()

            # (10) convergence test
            converged, distance = has_converged(previous_state, self.global_state,
                                                cfg.train.tolerance)

            # metrics
            self.model.load_state_dict(self.global_state, strict=False)
            result = evaluate(self.model, test_loader, cfg.model, cfg.data.num_classes, cfg.device)
            round_mb = self.channel.stats.total_mb
            total_mb += round_mb
            if cfg.comm.adaptive_sparsity and cfg.comm.topk_ratio < 1.0:
                self.compressor.ratio = self.bandwidth.update(round_mb)
            if result.accuracy > best_eval.accuracy:
                best_eval = result
                best_state = clone_state(self.global_state)

            record = {
                "round": round_id,
                **result.as_dict(),
                "clients_selected": len(selection.selected),
                "clients_reported": len(contributed),
                "clients_eligible": selection.eligible,
                "selection_relaxed": selection.relaxed,
                "mb_round": round(round_mb, 5),
                "mb_cumulative": round(total_mb, 4),
                "delta_norm": round(distance, 6),
                "aggregate_norm": round(report.update_norm, 6),
                "effective_clients": round(report.effective_clients, 3),
                "mean_q": round(float(np.mean([u.q_k for u in contributed])), 4),
                "mean_local_epochs": round(float(np.mean([u.local_epochs for u in raw_updates])), 2)
                if raw_updates else 0.0,
                "train_loss": round(float(np.mean([u.train_loss for u in raw_updates])), 5)
                if raw_updates else 0.0,
                "lr": round(self.trainer.learning_rate(round_id), 6),
                "rejected": len(rejected),
                "secure_aggregation_verified": secure_ok,
                "topk_ratio": round(self.compressor.ratio, 4),
                "epsilon_spent": round(self.accountant.spent(), 4) if self.accountant else 0.0,
                "round_seconds": round(time.perf_counter() - round_started, 3),
                **{f"drop_{k}": v for k, v in drop_stats.items()},
            }
            self.logger.log(record)

            if converged:
                break

        # final reporting on the best checkpoint (PRD 16: non-convergence still
        # returns the best model plus diagnostics)
        self.model.load_state_dict(best_state, strict=False)
        final_eval = evaluate(self.model, test_loader, cfg.model, cfg.data.num_classes, cfg.device)
        robustness = evaluate_fleet(self.model, self.fleet, cfg.model,
                                    cfg.data.num_classes, cfg.device)
        rounds = max(rounds_run, 1)
        result = RunResult(
            name=cfg.name,
            config=cfg.to_dict(),
            best_state=best_state,
            final_eval=final_eval,
            best_eval=best_eval if best_eval.samples else final_eval,
            rounds_run=rounds_run,
            rounds_to_target=self.logger.rounds_to_target(cfg.train.target_accuracy),
            mb_per_round=total_mb / rounds,
            total_mb=total_mb,
            converged=converged,
            history=list(self.logger.records),
            robustness=robustness.as_dict(),
            privacy=(self.accountant.summary() if self.accountant
                     else {"mode": "no DP noise", "raw_data_egress": False,
                           "secure_aggregation": cfg.dp.secure_aggregation}),
            selection=self.selector.coverage(),
            wall_clock_s=time.perf_counter() - started,
        )
        self.logger.write_summary(result.summary())
        torch.save({"state_dict": best_state, "config": cfg.to_dict()},
                   os.path.join(self.logger.run_dir, "best_model.pt"))
        return result


def run_experiment(cfg: FLConfig, fleet: Optional[Fleet] = None, echo: bool = True) -> RunResult:
    return SimRunner(cfg, fleet=fleet, echo=echo).run()


__all__ = ["SimRunner", "RunResult", "run_experiment"]
