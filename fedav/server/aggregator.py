"""FedAggregator - quality-aware FedAvg on the RSU (PRD 8.4, 8.8, FR-03/06/07).

Standard FedAvg forms the size-weighted mean

    w^{t+1} = sum_k (n_k / n) w_k^{t+1}

This aggregator implements the PRD's enhancement, replacing the size weight
with the quality-aware weight

    alpha_k = q_k n_k / sum_j q_j n_j ,      w^{t+1} = sum_k alpha_k w_k^{t+1}

and adds two stabilisers for heterogeneous fleets (PRD 8.8):

* **server momentum** ``v <- beta v + d``, ``w <- w + v`` on the aggregated
  delta, which damps the oscillation caused by clients pulling in different
  directions between rounds;
* an optional **coordinate-wise trimmed mean** for Byzantine robustness
  (PRD 15/16), which discards the extreme tails per coordinate before averaging.

Because clients transmit deltas, aggregation is a weighted mean of deltas
followed by ``w^{t+1} = w^{t} + d`` - algebraically identical to averaging the
weights themselves whenever every participant started the round from ``w^t``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..config import AggregationConfig
from ..data.quality import aggregation_weights
from ..utils.tensors import (StateDict, flatten, state_add, state_norm, state_scale,
                             state_zeros_like, trimmed_mean, unflatten, weighted_average)


@dataclass
class AggregationReport:
    weights: Dict[int, float]
    delta_norm: float
    update_norm: float
    participants: int
    rejected: List[int] = field(default_factory=list)
    effective_clients: float = 0.0       # 1 / sum(alpha^2), participation entropy


@dataclass
class ContributedUpdate:
    """A verified update as it reaches the aggregator."""

    client_id: int
    delta: StateDict
    n_k: int
    q_k: float


class FedAggregator:
    def __init__(self, cfg: AggregationConfig):
        self.cfg = cfg
        self.momentum_buffer: Optional[StateDict] = None

    # ------------------------------------------------------------- weighting
    def weights_for(self, updates: Sequence[ContributedUpdate]) -> np.ndarray:
        n_k = [u.n_k for u in updates]
        q_k = [u.q_k for u in updates] if self.cfg.quality_aware else None
        return aggregation_weights(n_k, q_k)

    # ----------------------------------------------------------- aggregation
    def combine(self, updates: Sequence[ContributedUpdate]
                ) -> Tuple[StateDict, np.ndarray]:
        """Weighted mean of the received deltas: ``sum_k alpha_k dW_k``."""
        if not updates:
            raise ValueError("no updates received this round")
        alphas = self.weights_for(updates)
        deltas = [u.delta for u in updates]
        if self.cfg.byzantine_robust and len(updates) >= 4:
            reference = deltas[0]
            stack = torch.stack([flatten(d, list(reference.keys())) for d in deltas])
            robust = trimmed_mean(stack, torch.tensor(alphas, dtype=torch.float32),
                                  self.cfg.trim_fraction)
            return unflatten(robust, reference), alphas
        return weighted_average(deltas, alphas.tolist()), alphas

    def apply(self, global_state: StateDict, aggregated: StateDict,
              alphas: np.ndarray, client_ids: Sequence[int],
              rejected: Optional[List[int]] = None) -> Tuple[StateDict, AggregationReport]:
        """Server momentum + ``w^{t+1} = w^t + d`` (PRD 8.4 / 8.8)."""
        applied = aggregated
        if self.cfg.server_momentum > 0:
            if self.momentum_buffer is None:
                self.momentum_buffer = state_zeros_like(aggregated)
            beta = float(self.cfg.server_momentum)
            self.momentum_buffer = {
                key: beta * self.momentum_buffer[key] + aggregated[key] for key in aggregated
            }
            # Convex blend of the momentum direction and the fresh aggregate keeps
            # the effective step size comparable to plain FedAvg while inheriting
            # the smoothed direction.
            applied = {key: (1 - beta) * self.momentum_buffer[key] + beta * aggregated[key]
                       for key in aggregated}

        new_state = state_add(global_state, applied, alpha=1.0)
        report = AggregationReport(
            weights={int(cid): float(a) for cid, a in zip(client_ids, alphas)},
            delta_norm=state_norm(applied),
            update_norm=state_norm(aggregated),
            participants=len(client_ids),
            rejected=list(rejected or []),
            effective_clients=float(1.0 / np.square(alphas).sum()) if alphas.size else 0.0,
        )
        return new_state, report

    def aggregate(self, global_state: StateDict,
                  updates: Sequence[ContributedUpdate],
                  rejected: Optional[List[int]] = None) -> Tuple[StateDict, AggregationReport]:
        aggregated, alphas = self.combine(updates)
        return self.apply(global_state, aggregated, alphas,
                          [u.client_id for u in updates], rejected)

    # ------------------------------------------------------------ robustness
    @staticmethod
    def norm_filter(updates: Sequence[ContributedUpdate], factor: float = 3.0
                    ) -> Tuple[List[ContributedUpdate], List[int]]:
        """Reject updates whose norm exceeds ``factor`` x the median norm.

        First line of defence against poisoned updates (PRD 16); the trimmed
        mean then handles whatever slips through.
        """
        if len(updates) < 4:
            return list(updates), []
        norms = np.array([state_norm(u.delta) for u in updates])
        median = float(np.median(norms))
        limit = factor * median if median > 0 else float("inf")
        keep, rejected = [], []
        for update, norm in zip(updates, norms):
            if norm <= limit:
                keep.append(update)
            else:
                rejected.append(update.client_id)
        return keep, rejected

    def reset(self) -> None:
        self.momentum_buffer = None


def has_converged(previous: StateDict, current: StateDict, tolerance: float) -> Tuple[bool, float]:
    """FR-07 / PRD 8.9: stop when ||w^{t+1} - w^t||_2 <= epsilon."""
    from ..utils.tensors import state_sub
    distance = state_norm(state_sub(current, previous))
    return distance <= tolerance, distance


__all__ = ["FedAggregator", "ContributedUpdate", "AggregationReport", "has_converged"]
