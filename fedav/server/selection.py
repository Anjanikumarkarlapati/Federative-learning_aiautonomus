"""ClientSelector - connectivity- and trajectory-aware selection (PRD 8.5, FR-04).

Per round the RSU picks ``S = ceil(C * N)`` vehicles subject to three rules:

1. **Connectivity.** Only vehicles whose predicted V2I link uptime clears
   ``uptime_threshold`` are eligible, so rounds are not lost to dropped
   uplinks (the infra-engineer user story of PRD 5).
2. **Trajectory / geography coverage.** Selection is stratified over regions so
   the round still sees the whole data distribution - discarding weak links
   would otherwise systematically drop whole regions and *worsen* the Non-IID
   problem it is meant to relieve.
3. **Fairness guard.** Any vehicle not selected within the last ``K`` rounds is
   promoted to the front of the queue, bounding starvation and keeping the
   global objective faithful to ``sum_k (n_k/n) F_k`` (PRD 8.2).

Under very low connectivity (PRD 16) the threshold is relaxed progressively
rather than returning an empty round.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..config import SelectionConfig
from ..data.federated import VehicleProfile
from ..utils.seeding import rng as make_rng


@dataclass
class SelectionResult:
    selected: List[int]
    eligible: int
    forced_fairness: List[int] = field(default_factory=list)
    relaxed: bool = False

    def __len__(self) -> int:
        return len(self.selected)


class ClientSelector:
    def __init__(self, cfg: SelectionConfig, num_clients: int, seed: int = 0):
        self.cfg = cfg
        self.num_clients = num_clients
        self.seed = seed
        self.last_seen: Dict[int, int] = {k: -1 for k in range(num_clients)}
        self.times_selected: Dict[int, int] = defaultdict(int)

    # ------------------------------------------------------------- utilities
    def target_size(self) -> int:
        return max(self.cfg.min_clients,
                   int(math.ceil(self.cfg.fraction * self.num_clients)))

    def _link_sample(self, profile: VehicleProfile, round_id: int) -> float:
        """Predicted uptime for this round: the vehicle's long-run uptime with
        round-to-round jitter, i.e. mobility through the RSU's coverage cell."""
        if not self.cfg.seed_jitter:
            return profile.link_uptime
        generator = make_rng(self.seed, round_id, profile.client_id, 3)
        jitter = float(generator.normal(0.0, 0.08))
        return float(np.clip(profile.link_uptime + jitter, 0.0, 1.0))

    # -------------------------------------------------------------- selection
    def select(self, profiles: Sequence[VehicleProfile], round_id: int) -> SelectionResult:
        cfg = self.cfg
        target = min(self.target_size(), len(profiles))
        if cfg.mode == "uniform":
            return self._select_uniform(profiles, round_id, target)
        if cfg.mode == "kst_ucb":
            return self._select_kst_ucb(profiles, round_id, target)
        uptimes = {p.client_id: self._link_sample(p, round_id) for p in profiles}

        threshold, relaxed = cfg.uptime_threshold, False
        eligible = [p for p in profiles if uptimes[p.client_id] >= threshold]
        while len(eligible) < target and threshold > 0.0:
            threshold *= 0.7                      # PRD 16: very low connectivity
            relaxed = True
            eligible = [p for p in profiles if uptimes[p.client_id] >= threshold]
        if not eligible:
            eligible = list(profiles)
            relaxed = True

        # 1. fairness guard - starved vehicles first. It runs *before* the uptime
        #    gate: a vehicle with a weak link is still owed a slot every K rounds,
        #    otherwise the bound only holds for vehicles that never needed it.
        starved = [p for p in profiles
                   if round_id - self.last_seen[p.client_id] > cfg.fairness_k
                   or self.last_seen[p.client_id] < 0]
        starved.sort(key=lambda p: (self.last_seen[p.client_id], -uptimes[p.client_id]))
        chosen: List[int] = [p.client_id for p in starved[:target]]
        forced = list(chosen)

        # 2. trajectory-aware stratification over regions
        remaining = [p for p in eligible if p.client_id not in set(chosen)]
        if cfg.trajectory_aware and len(chosen) < target:
            buckets: Dict[str, List[VehicleProfile]] = defaultdict(list)
            for p in remaining:
                buckets[p.region].append(p)
            for bucket in buckets.values():
                bucket.sort(key=lambda p: -self._score(p, uptimes[p.client_id]))
            regions = sorted(buckets)
            idx = 0
            while len(chosen) < target and any(buckets[r] for r in regions):
                region = regions[idx % len(regions)]
                idx += 1
                if buckets[region]:
                    chosen.append(buckets[region].pop(0).client_id)
        # 3. fill the rest by score
        if len(chosen) < target:
            pool = [p for p in remaining if p.client_id not in set(chosen)]
            pool.sort(key=lambda p: -self._score(p, uptimes[p.client_id]))
            chosen.extend(p.client_id for p in pool[:target - len(chosen)])

        chosen = chosen[:target]
        for client_id in chosen:
            self.last_seen[client_id] = round_id
            self.times_selected[client_id] += 1
        return SelectionResult(selected=sorted(chosen), eligible=len(eligible),
                               forced_fairness=forced, relaxed=relaxed)

    def _select_uniform(self, profiles: Sequence[VehicleProfile], round_id: int,
                        target: int) -> SelectionResult:
        """McMahan FedAvg: ``target`` vehicles uniformly at random, no link or
        quality information used."""
        generator = make_rng(self.seed, round_id, 17)
        ids = [p.client_id for p in profiles]
        chosen = sorted(int(c) for c in generator.choice(ids, size=target, replace=False))
        for client_id in chosen:
            self.last_seen[client_id] = round_id
            self.times_selected[client_id] += 1
        return SelectionResult(selected=chosen, eligible=len(profiles))

    def _select_kst_ucb(self, profiles: Sequence[VehicleProfile], round_id: int,
                        target: int) -> SelectionResult:
        """Quality floor + UCB staleness exploration + rare-hazard scout reserve.

        * Vehicles whose reported sensor health is below ``quality_floor`` are
          excluded outright; link uptime is part of the score, not a gate (mud-occluded or faulty sensors). The floor uses the
          sensor-health signal the vehicle already beacons, not q_k, which stays
          hidden under secure aggregation.
        * The remaining vehicles are ranked by the usual link/quality score plus
          ``ucb_beta * sqrt(ln(t+1) / (N_k(t)+1))``, so a rarely chosen vehicle's
          bonus grows until it is picked.
        * ``rarity_reserve`` of the cohort goes first to the vehicles with the
          highest rare-scene share (classes beyond the common six, or label
          divergence from the fleet when there are none).
        """
        cfg = self.cfg
        uptimes = {p.client_id: self._link_sample(p, round_id) for p in profiles}
        # link uptime enters through the score, not as a gate, so the UCB term can
        # always reach a weak-link vehicle eventually
        eligible = [p for p in profiles if p.quality_signal >= cfg.quality_floor]
        relaxed = len(eligible) < target
        if relaxed:                                # never return a short cohort
            eligible = list(profiles)

        def ucb(p: VehicleProfile) -> float:
            n = self.times_selected[p.client_id]
            return self._score(p, uptimes[p.client_id]) + cfg.ucb_beta * math.sqrt(
                math.log(round_id + 1) / (n + 1))

        def rarity(p: VehicleProfile) -> float:
            h = np.asarray(p.label_histogram, dtype=np.float64)
            return float(h[6:].sum()) if h.size > 6 else 0.0

        chosen: List[int] = []
        reserve = int(round(cfg.rarity_reserve * target))
        if reserve and any(rarity(p) > 0 for p in eligible):
            # the 2x-reserve rarest vehicles form the scout pool; UCB rotates
            # the reserved slots through it instead of always taking the rarest
            pool = sorted(eligible, key=lambda p: -rarity(p))[:2 * reserve]
            chosen = [p.client_id for p in sorted(pool, key=lambda p: -ucb(p))[:reserve]]
        rest = sorted((p for p in eligible if p.client_id not in chosen), key=lambda p: -ucb(p))
        chosen += [p.client_id for p in rest[:target - len(chosen)]]
        for client_id in chosen:
            self.last_seen[client_id] = round_id
            self.times_selected[client_id] += 1
        return SelectionResult(selected=sorted(chosen), eligible=len(eligible), relaxed=relaxed)

    def _score(self, profile: VehicleProfile, uptime: float) -> float:
        """Prefer stable links and larger, healthier shards; mild inverse-count
        term keeps participation spread across the fleet."""
        freshness = 1.0 / (1.0 + self.times_selected[profile.client_id])
        return float(0.55 * uptime + 0.25 * profile.quality_signal
                     + 0.10 * math.log1p(profile.n_k) / 10.0 + 0.10 * freshness)

    def coverage(self) -> Dict[str, float]:
        counts = np.array([self.times_selected[k] for k in range(self.num_clients)])
        return {
            "clients_never_selected": int((counts == 0).sum()),
            "selection_gini": float(_gini(counts)),
            "mean_participation": float(counts.mean()) if counts.size else 0.0,
        }


def _gini(values: np.ndarray) -> float:
    if values.size == 0 or values.sum() == 0:
        return 0.0
    sorted_values = np.sort(values.astype(np.float64))
    n = sorted_values.size
    index = np.arange(1, n + 1)
    return float((2 * (index * sorted_values).sum()) / (n * sorted_values.sum()) - (n + 1) / n)


__all__ = ["ClientSelector", "SelectionResult"]
