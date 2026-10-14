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

import math
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
from ..metrics.fleet import (VehicleTracker, evaluate_vehicles, per_vehicle_accuracy,
                             write_vehicle_profiles, write_vehicle_summary)
from ..metrics.logger import MetricsLogger
from ..models.perception import build_model
from ..privacy.dp import PrivacyAccountant
from ..server.aggregator import ContributedUpdate, FedAggregator, has_converged
from ..server.secure import SecureChannel
from ..server.selection import ClientSelector
from ..server import secagg
from ..server.v2x import V2XRadioModel
from ..utils.seeding import rng as make_rng, seed_everything
from ..utils.tensors import (StateDict, clone_state, flatten, state_add, state_norm,
                             state_numel, state_sub, state_zeros_like,
                             to_float_state, unflatten, weighted_average)


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
    radio: Dict[str, float] = field(default_factory=dict)
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
            "radio": self.radio,
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
        self.radio = V2XRadioModel()              # bytes -> airtime / energy
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
        self.vehicles = VehicleTracker(self.logger.run_dir, self.fleet)
        # quality-hiding secure aggregation state (claims K1 / K2)
        self._keys: Dict[int, secagg.KeyPair] = {}
        self._sa_residual: Dict[int, np.ndarray] = {}   # on-vehicle, step-normalised units
        self._prev_update: Optional[np.ndarray] = None  # last broadcast global delta (public)
        self._value_bound: Optional[float] = None
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
                 ) -> Tuple[List[ContributedUpdate], List[ClientUpdate], Dict[str, int],
                            Dict[int, Dict[str, Any]]]:
        """Local training + uplink for every selected vehicle.

        Also returns a per-vehicle event (status, q_k, epochs, loss) for the
        simulator's own telemetry - none of it reaches the aggregator.
        """
        dp_sigma = 0.0
        if self.accountant is not None:
            dp_sigma = (self.accountant.client_sigma(len(selected))
                        if self.cfg.dp.secure_aggregation else self.accountant.sigma)

        contributed: List[ContributedUpdate] = []
        raw_updates: List[ClientUpdate] = []
        stats = {"dropped": 0, "straggler": 0, "corrupt": 0}
        events: Dict[int, Dict[str, Any]] = {}
        generator = make_rng(self.cfg.seed, round_id, 55)

        for client_id in selected:
            vehicle = self.fleet[client_id]
            update = self.trainer.train(self.global_state, vehicle, round_id,
                                        self.fleet.reference_histogram, dp_sigma)
            event = {"q_k": round(float(update.q_k), 5),
                     "local_epochs": int(update.local_epochs),
                     "train_loss": round(float(update.train_loss), 5),
                     "update_norm": round(float(update.update_norm), 6),
                     "train_seconds": round(float(update.duration_s), 3)}
            if update.duration_s > self.cfg.selection.round_timeout_s:
                stats["straggler"] += 1            # PRD 16: past the round window
                events[client_id] = {"status": "straggler", **event}
                continue
            ef_norm = self.cfg.comm.ef_normalize_steps
            payload = self.compressor.compress(
                client_id, update.delta, steps=update.steps, normalize=ef_norm,
                fp16=self.cfg.comm.ef_quantization and self.cfg.comm.fp16)
            sent, sent_steps = payload, (update.steps if ef_norm else 1.0)
            if not self.cfg.comm.send_deltas:
                # Classic FedAvg puts the full local weights on the wire; the
                # server differences them against w^(r) to recover the update.
                payload = state_add(self.global_state, payload, alpha=1.0)
            blob = self.serializer.pack(payload, update.meta())
            delivered, wire_bytes, _ = self.channel.send_up(blob, vehicle.profile.link_uptime,
                                                            generator)
            # Own RNG stream, so adding the radio model leaves every other draw unchanged.
            link = self.radio.sample(wire_bytes, vehicle.profile.region,
                                     make_rng(self.cfg.seed, round_id, 91, client_id))
            event.update({"airtime_s": round(link.airtime_s, 5),
                          "radio_energy_j": round(link.energy_j, 5),
                          "rate_mbps": round(link.rate_mbps, 3),
                          "distance_m": round(link.distance_m, 1)})
            if not delivered:
                stats["dropped"] += 1              # PRD 16: client drops mid-round
                events[client_id] = {"status": "dropped", **event}
                self._nack(client_id, sent, sent_steps)
                continue
            blob = self._maybe_corrupt(blob, round_id, client_id)
            try:
                delta, meta = self.serializer.unpack(blob)
                if not self.cfg.comm.send_deltas:
                    delta = state_sub(delta, self.global_state)
            except ChecksumError:
                stats["corrupt"] += 1
                events[client_id] = {"status": "corrupt", **event}
                self._nack(client_id, sent, sent_steps)
                continue
            events[client_id] = {"status": "reported", "uplink_kb": round(len(blob) / 1024, 2),
                                 **event}
            contributed.append(ContributedUpdate(
                client_id=int(meta["client_id"]), delta=delta, n_k=int(meta["n_k"]),
                q_k=float(meta["q_k"]), steps=int(meta.get("steps", 1))))
            raw_updates.append(update)
        return contributed, raw_updates, stats, events

    # ---------------------------------------------- quality-hiding secure round
    def _weight(self, q_k: float, n_k: int) -> float:
        a = self.cfg.aggregation
        n_ref = max(self.cfg.data.samples_per_client * self.cfg.data.train_share, 1.0)
        q = float(np.clip(q_k, a.q_min, a.q_max)) if a.quality_aware else 1.0
        return float(min(q * n_k / n_ref, a.q_max * a.weight_cap))

    def _secure_round(self, selected: List[int], round_id: int) -> Dict[str, Any]:
        """One round of K1 + K2 (see fedav/server/secagg.py).

        Vehicles train, fold ``w_k`` and ``tau_k`` into a masked fixed-point
        payload over a shared, tiered support, and transmit it. The RSU sums what
        arrives, removes dropped vehicles' masks and normalises. Per-vehicle
        ``q_k`` / ``alpha`` in the telemetry are the simulator's ground truth for
        plotting; the aggregation path never reads them.
        """
        cfg = self.cfg
        dp_sigma = 0.0
        if self.accountant is not None:
            dp_sigma = self.accountant.client_sigma(len(selected))
        keys = list(self.global_state.keys())
        numel = state_numel(self.global_state)
        # the MB-budget controller adapts the top tier exactly as it adapts top-k
        top_ratio = min(self.compressor.ratio, 1.0)
        full = max(1, int(round(top_ratio * numel)))
        smallest = max(1, int(round(min(cfg.comm.min_topk_ratio, top_ratio) * numel)))
        ladder = secagg.tier_ladder(full, smallest, cfg.comm.tier_levels)
        support = secagg.shared_support(self._prev_update, numel, full, cfg.seed, round_id)
        w_max = cfg.aggregation.q_max * cfg.aggregation.weight_cap
        if self._value_bound is None:
            self._value_bound = 0.05 * w_max
        value_fp = secagg.FixedPoint(self._value_bound)
        # public bound on w * tau: weight cap times the most local steps any vehicle can take
        n_ref = max(cfg.data.samples_per_client * cfg.data.train_share, 1.0)
        tau_cap = (max(cfg.train.local_epochs, cfg.train.max_epochs)
                   * math.ceil(cfg.aggregation.weight_cap * n_ref / cfg.train.batch_size + 1))
        scalar_fp = secagg.FixedPoint(float(w_max * tau_cap))

        # key setup: each selected vehicle publishes g^a once; the RSU multicasts the list
        for c in selected:
            if c not in self._keys:
                self._keys[c] = secagg.keypair(cfg.seed, c)
                self.channel.stats.uplink_bytes += secagg.PUBLIC_KEY_BYTES
        self.channel.stats.downlink_bytes += secagg.PUBLIC_KEY_BYTES * len(selected)

        # links this round; the RSU predicts rates from position, not shadowing
        links = {c: self.radio.draw_link(self.fleet[c].profile.region,
                                         make_rng(cfg.seed, round_id, 91, c)) for c in selected}
        predicted = {c: self.radio.predicted_rate_mbps(d, los) for c, (d, los, _) in links.items()}
        tiers = secagg.assign_tiers(predicted, ladder, cfg.comm.airtime_budget_s)
        self.channel.stats.downlink_bytes += 2 * len(selected)          # tier ids

        stats = {"dropped": 0, "straggler": 0, "corrupt": 0}
        events: Dict[int, Dict[str, Any]] = {}
        raw_updates: List[ClientUpdate] = []
        arrived: Dict[int, np.ndarray] = {}
        truth: Dict[int, Tuple[float, float, np.ndarray]] = {}
        clipped = sent_values = scalar_clipped = 0
        generator = make_rng(cfg.seed, round_id, 55)
        pubs = {c: self._keys[c].public for c in selected}

        trained: Dict[int, ClientUpdate] = {}
        for c in selected:
            vehicle = self.fleet[c]
            update = self.trainer.train(self.global_state, vehicle, round_id,
                                        self.fleet.reference_histogram, dp_sigma)
            events[c] = {"q_k": round(float(update.q_k), 5),
                         "local_epochs": int(update.local_epochs),
                         "train_loss": round(float(update.train_loss), 5),
                         "update_norm": round(float(update.update_norm), 6),
                         "train_seconds": round(float(update.duration_s), 3),
                         "tier": int(tiers[c])}
            if update.duration_s > cfg.selection.round_timeout_s:
                stats["straggler"] += 1
                events[c]["status"] = "straggler"
                continue
            trained[c] = update
        # vehicles that timed out never upload: treat them as dropped for masking
        for c in selected:
            if c not in trained:
                continue
            update = trained[c]
            tau = float(max(int(update.steps), 1))
            u = flatten(update.delta, keys).double().numpy()
            if cfg.aggregation.normalize_local_steps:
                u = u / tau
            if c in self._sa_residual and cfg.comm.error_feedback:
                u = u + self._sa_residual[c]
            w = self._weight(update.q_k, update.n_k)
            scalar_clipped += int(w * tau > scalar_fp.bound)
            length = ladder[tiers[c]]
            values = u[support[:length]]
            up = secagg.vehicle_encode(c, round_id, self._keys[c], pubs, tiers, ladder,
                                       tiers[c], values, w, tau, value_fp, scalar_fp)
            clipped += int(np.count_nonzero(np.abs(w * values) > value_fp.bound))
            sent_values += length
            residual = u.copy()
            residual[support[:length]] = up.clipped_excess        # sent part leaves
            delivered, wire_bytes, _ = self.channel.send_up(
                up.payload, self.fleet[c].profile.link_uptime, generator)
            link = self.radio.transmit(wire_bytes, links[c])
            events[c].update({"airtime_s": round(link.airtime_s, 5),
                              "radio_energy_j": round(link.energy_j, 5),
                              "rate_mbps": round(link.rate_mbps, 3),
                              "distance_m": round(link.distance_m, 1),
                              "uplink_kb": round(wire_bytes / 1024, 2)})
            blob = self._maybe_corrupt(up.payload, round_id, c) if delivered else None
            ok = False
            if delivered:
                try:
                    _, _, _, vec = secagg.parse_upload(blob)
                    ok = True
                except secagg.PayloadError:
                    stats["corrupt"] += 1
                    events[c]["status"] = "corrupt"
            else:
                stats["dropped"] += 1
                events[c]["status"] = "dropped"
            if ok:
                arrived[c] = vec
                events[c]["status"] = "reported"
                truth[c] = (w, tau, values - up.clipped_excess)
                raw_updates.append(update)
            elif not cfg.comm.nack_retention:
                pass                                        # sent coordinates are lost
            else:
                residual = u                                # NACK: keep everything
            if cfg.comm.error_feedback:
                self._sa_residual[c] = residual

        out: Dict[str, Any] = {"stats": stats, "events": events, "raw_updates": raw_updates,
                               "arrived": list(arrived), "aggregated": None}
        if not arrived:
            return out
        # every selected vehicle masked against every other, so stragglers count
        # as dropped too and survivors reveal their seeds with them
        sums = secagg.rsu_aggregate(arrived, tiers, self._keys, round_id,
                                    ladder, value_fp, scalar_fp)
        self.channel.stats.uplink_bytes += secagg.SEED_REVEAL_BYTES * sums.seeds_revealed
        d_support = secagg.combine_segments(sums, ladder, cfg.aggregation.normalize_local_steps)
        flat = np.zeros(numel, dtype=np.float64)
        flat[support] = d_support
        aggregated = unflatten(torch.from_numpy(flat).float(), self.global_state)

        # simulator-only oracle: the same aggregate from plaintext values
        expected = np.zeros(full)
        start = 0
        for t, end in enumerate(ladder):
            cover = [c for c in arrived if tiers[c] >= t]
            if cover:
                W = sum(truth[c][0] for c in cover)
                T = sum(truth[c][0] * truth[c][1] for c in cover)
                S = sum(truth[c][0] * truth[c][2][start:end] for c in cover)
                expected[start:end] = ((T / W) if cfg.aggregation.normalize_local_steps
                                       else 1.0) * S / W
            start = end
        err = float(np.linalg.norm(d_support - expected) / max(np.linalg.norm(expected), 1e-12))
        weights = np.array([truth[c][0] for c in arrived])
        out.update({
            "aggregated": aggregated,
            "alphas": weights / weights.sum(),
            "contributed": [ContributedUpdate(client_id=c, delta={}, n_k=self.fleet[c].profile.n_k,
                                              q_k=float(events[c]["q_k"]),
                                              steps=int(truth[c][1])) for c in arrived],
            "secure_ok": err <= 1e-3,
            "secagg": {"secagg_rel_error": round(err, 8),
                       "secagg_clip_frac": round(clipped / max(sent_values, 1), 6),
                       "secagg_scalar_clipped": scalar_clipped,
                       "secagg_value_bound": round(value_fp.bound, 8),
                       "secagg_seeds_revealed": sums.seeds_revealed,
                       "secagg_mean_tier": round(float(np.mean([tiers[c] for c in trained])), 3),
                       "secagg_support": full},
        })
        for c, a in zip(arrived, out["alphas"]):
            events[c]["alpha"] = round(float(a), 5)
        self._value_bound = max(4.0 * w_max * float(np.max(np.abs(d_support))), 1e-6)
        return out

    def _nack(self, client_id: int, sent: StateDict, steps: float = 1.0) -> None:
        if (self.cfg.comm.nack_retention and self.cfg.comm.send_deltas
                and self.compressor.ratio < 1.0 and self.compressor.error_feedback):
            self.compressor.restore(client_id, sent, steps)

    def _maybe_corrupt(self, blob: bytes, round_id: int, client_id: int) -> bytes:
        """Fault injection: flip one byte with probability ``comm.corrupt_prob``."""
        p = self.cfg.comm.corrupt_prob
        if p <= 0:
            return blob
        g = make_rng(self.cfg.seed, round_id, 93, client_id)
        if g.random() >= p:
            return blob
        raw = bytearray(blob)
        pos = int(g.integers(len(raw) // 2, len(raw)))
        raw[pos] ^= 0xFF
        return bytes(raw)

    @staticmethod
    def _radio_totals(events: Dict[int, Dict[str, Any]]) -> Dict[str, float]:
        """Shared-channel uplink cost of the round: every transmission occupies
        the cell, delivered or not."""
        sent = [e for e in events.values() if "airtime_s" in e]
        if not sent:
            return {"uplink_airtime_s": 0.0, "uplink_energy_j": 0.0, "max_vehicle_airtime_s": 0.0}
        return {"uplink_airtime_s": round(sum(e["airtime_s"] for e in sent), 5),
                "uplink_energy_j": round(sum(e["radio_energy_j"] for e in sent), 5),
                "max_vehicle_airtime_s": round(max(e["airtime_s"] for e in sent), 5)}

    def _radio_summary(self, records: List[Dict[str, Any]], rounds: int) -> Dict[str, Any]:
        airtime = sum(r.get("uplink_airtime_s", 0.0) for r in records)
        energy = sum(r.get("uplink_energy_j", 0.0) for r in records)
        return {"model": "3GPP TR 38.901 UMi (TR 37.885 V2I), "
                         f"{self.radio.carrier_ghz} GHz, {self.radio.bandwidth_mhz} MHz, "
                         f"{self.radio.tx_power_dbm} dBm",
                "uplink_airtime_s_per_round": round(airtime / rounds, 4),
                "uplink_airtime_s_total": round(airtime, 3),
                "uplink_energy_j_per_round": round(energy / rounds, 4),
                "uplink_energy_j_total": round(energy, 3)}


    # -------------------------------------------------------------------- run
    def run(self) -> RunResult:
        cfg = self.cfg
        started = time.perf_counter()
        self.fleet.reset_augmentation()          # independent of earlier arms
        self.logger.write_config(cfg.to_dict())
        write_vehicle_profiles(self.logger.run_dir, self.fleet)
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
            secure_round = cfg.dp.secure_aggregation and not cfg.aggregation.byzantine_robust
            secagg_stats: Dict[str, Any] = {}
            if secure_round:
                sr = self._secure_round(selection.selected, round_id)
                raw_updates, drop_stats, events = sr["raw_updates"], sr["stats"], sr["events"]
                contributed = sr.get("contributed", [])
            else:
                contributed, raw_updates, drop_stats, events = self._collect(
                    selection.selected, round_id)
            if not contributed:
                self.vehicles.log_round(round_id, events)
                self.logger.log({"round": round_id, "clients_selected": len(selection.selected),
                                 "clients_reported": 0, "note": "no updates received",
                                 **{f"drop_{k}": v for k, v in drop_stats.items()}})
                continue

            rejected: List[int] = []
            if secure_round:
                aggregated, alphas = sr["aggregated"], sr["alphas"]
                secure_ok, secagg_stats = sr["secure_ok"], sr["secagg"]
            elif cfg.aggregation.byzantine_robust:
                contributed, rejected = FedAggregator.norm_filter(contributed)
                for client_id in rejected:
                    events[client_id]["status"] = "rejected"

            # (9) quality-aware FedAvg; the secure path already aggregated in the masked domain
            if not secure_round:
                aggregated, alphas = self.aggregator.combine(contributed)
                for update, alpha in zip(contributed, alphas):
                    events[update.client_id]["alpha"] = round(float(alpha), 5)
                secure_ok = False
            previous_state = self.global_state
            self.global_state, report = self.aggregator.apply(
                previous_state, aggregated, alphas,
                [u.client_id for u in contributed], rejected,
            )
            if cfg.train.scaffold:
                # delta c_k rides the uplink next to the update: count its bytes
                delta_cs = [u.delta_c for u in raw_updates if u.delta_c is not None]
                for dc in delta_cs:
                    self.channel.stats.uplink_bytes += len(self.serializer.pack(dc, {}))
                self.trainer.scaffold_server_update(delta_cs, len(self.fleet))
            if secure_round:     # every vehicle derives next round's support from this
                self._prev_update = flatten(state_sub(self.global_state, previous_state),
                                            list(self.global_state.keys())).double().numpy()
            if self.accountant is not None:
                self.accountant.step()

            # (10) convergence test
            converged, distance = has_converged(previous_state, self.global_state,
                                                cfg.train.tolerance)

            # metrics
            self.model.load_state_dict(self.global_state, strict=False)
            result = evaluate(self.model, test_loader, cfg.model, cfg.data.num_classes, cfg.device)
            self.vehicles.log_round(round_id, events,
                                    per_vehicle_accuracy(self.model, self.fleet, cfg, cfg.device))
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
                **self._radio_totals(events),
                **secagg_stats,
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
            radio=self._radio_summary(self.logger.records, rounds),
            wall_clock_s=time.perf_counter() - started,
        )
        self.logger.write_summary(result.summary())
        write_vehicle_summary(
            self.logger.run_dir, evaluate_vehicles(self.model, self.fleet, cfg, cfg.device),
            self.vehicles.participation(),
            note="final_* = best global model on each vehicle's own test split")
        torch.save({"state_dict": best_state, "config": cfg.to_dict()},
                   os.path.join(self.logger.run_dir, "best_model.pt"))
        return result


def run_experiment(cfg: FLConfig, fleet: Optional[Fleet] = None, echo: bool = True) -> RunResult:
    return SimRunner(cfg, fleet=fleet, echo=echo).run()


__all__ = ["SimRunner", "RunResult", "run_experiment"]
