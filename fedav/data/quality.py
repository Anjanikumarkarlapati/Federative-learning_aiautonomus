"""Client quality score ``q_k`` (PRD 8.4).

    q_k = similarity_k x quality_k,  q_k in [q_min, q_max]

``similarity_k`` measures how close the vehicle's label/feature distribution is
to the fleet reference; ``quality_k`` is an onboard sensor-health signal (blur /
occlusion / rain / calibration flag).  Both are computed **on the vehicle** - the
only thing that crosses the V2I link is the resulting scalar (PRD 8.6), so no
raw data and no per-sample statistics leave the car.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def _normalise(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64).clip(min=0.0)
    total = p.sum()
    return p / total if total > 0 else np.full_like(p, 1.0 / max(len(p), 1))


def hellinger_distance(p: Sequence[float], q: Sequence[float]) -> float:
    """H(p, q) = (1/sqrt(2)) * ||sqrt(p) - sqrt(q)||_2, in [0, 1]."""
    p, q = _normalise(np.asarray(p)), _normalise(np.asarray(q))
    return float(np.clip(np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2.0), 0.0, 1.0))


def jensen_shannon_distance(p: Sequence[float], q: Sequence[float]) -> float:
    """sqrt of the Jensen-Shannon divergence (base 2), in [0, 1]."""
    p, q = _normalise(np.asarray(p)), _normalise(np.asarray(q))
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / np.clip(b[mask], 1e-12, None))))
    return float(np.sqrt(np.clip(0.5 * _kl(p, m) + 0.5 * _kl(q, m), 0.0, 1.0)))


def similarity_score(histogram: Sequence[float], reference: Sequence[float],
                     metric: str = "hellinger") -> float:
    """1 - distance(client histogram, fleet reference), in [0, 1]."""
    distance = (hellinger_distance if metric == "hellinger" else jensen_shannon_distance)(
        histogram, reference
    )
    return float(np.clip(1.0 - distance, 0.0, 1.0))


def compute_quality_weight(histogram: Sequence[float], reference: Sequence[float],
                           quality_signal: float, q_min: float = 0.05,
                           q_max: float = 1.0, metric: str = "hellinger",
                           similarity_floor: float = 0.25) -> float:
    """q_k = clamp(similarity_k x quality_k).

    ``similarity_floor`` keeps a genuinely rare-but-clean region from being
    silenced entirely: a client that is merely *different* still contributes,
    while a client that is different **and** sensor-degraded is strongly damped.
    This is the "clamp q_k range" mitigation of PRD 18.
    """
    similarity = similarity_score(histogram, reference, metric=metric)
    similarity = similarity_floor + (1.0 - similarity_floor) * similarity
    quality = float(np.clip(quality_signal, 0.0, 1.0))
    return float(np.clip(similarity * quality, q_min, q_max))


def aggregation_weights(n_k: Sequence[float], q_k: Optional[Sequence[float]] = None
                        ) -> np.ndarray:
    """alpha_k = q_k n_k / sum_j q_j n_j  (PRD 8.4); plain FedAvg when q is None.

    Always returns a simplex vector - ``sum(alpha) == 1`` - which is the
    invariant the aggregation unit test of PRD 17 checks.
    """
    n = np.asarray(n_k, dtype=np.float64)
    if n.size == 0:
        return n
    w = n if q_k is None else n * np.asarray(q_k, dtype=np.float64)
    total = w.sum()
    if total <= 0:                       # degenerate: fall back to uniform
        return np.full(n.size, 1.0 / n.size)
    return w / total


__all__ = ["hellinger_distance", "jensen_shannon_distance", "similarity_score",
           "compute_quality_weight", "aggregation_weights"]
