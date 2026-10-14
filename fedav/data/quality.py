"""Client quality score ``q_k`` (PRD 8.4).

    q_k = similarity_k x quality_k,  q_k in [q_min, q_max]

``similarity_k`` measures how close the vehicle's label/feature distribution is
to the fleet reference; ``quality_k`` is an onboard sensor-health signal (blur /
occlusion / rain / calibration flag).  Both are computed **on the vehicle** - the
only thing that crosses the V2I link is the resulting scalar (PRD 8.6), so no
raw data and no per-sample statistics leave the car.

One caveat worth stating plainly: the *fleet reference* the vehicle compares
itself against has to come from somewhere.  PRD 8.4 defines similarity as a
"label/feature histogram distance", which presumes the server holds a reference
histogram; here it is the size-weighted mean of the vehicles' own histograms.
A histogram is far less revealing than raw sensor data, but it is not nothing -
in a deployment it should be collected under secure aggregation (the server then
sees only the fleet-level sum, never one vehicle's class mixture) or replaced by
a fixed reference agreed offline.  The simulator computes it directly for
convenience; ``Fleet.reference_histogram`` is the single place to change.
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


# ------------------------------------------------ KST quality (optional score)
def kl_divergence(p: Sequence[float], q: Sequence[float], eps: float = 1e-6) -> float:
    """KL(p || q) in nats, with both distributions smoothed by ``eps``."""
    p, q = _normalise(np.asarray(p) + eps), _normalise(np.asarray(q) + eps)
    return float(np.sum(p * np.log(p / q)))


def kinematic_novelty(mean_speed: float, yaw_rate_var: float, speed_ref: float = 0.25,
                      yaw_ref: float = 0.1, floor: float = 0.2) -> float:
    """K_k in [floor, 1]: a vehicle crawling in stationary traffic on a straight
    road sees little new; speed and manoeuvring (yaw-rate variance) raise it."""
    k = 0.6 * (1.0 - np.exp(-max(mean_speed, 0.0) / speed_ref)) \
        + 0.4 * (1.0 - np.exp(-max(yaw_rate_var, 0.0) / yaw_ref))
    return float(np.clip(k, floor, 1.0))


def sensor_fidelity(snr_db: float, occlusion_rate: float, calibrated: bool = True,
                    snr_mid: float = 6.0, snr_scale: float = 3.0) -> float:
    """S_k in (0, 1]: logistic in image SNR (dB), times the unoccluded share,
    times 0.55 for a de-calibrated sensor."""
    s = 1.0 / (1.0 + np.exp(-(snr_db - snr_mid) / snr_scale))
    s *= 1.0 - float(np.clip(occlusion_rate, 0.0, 1.0))
    if not calibrated:
        s *= 0.55
    return float(np.clip(s, 0.02, 1.0))


def rare_scene_bonus(histogram: Sequence[float], prior: Sequence[float],
                     strength: float = 0.5, kl_ref: float = 1.0) -> float:
    """R_k in [1, 1 + strength]: grows with KL(local labels || fleet prior), so a
    vehicle that sees what the fleet rarely sees is not averaged away."""
    return 1.0 + strength * min(kl_divergence(histogram, prior) / kl_ref, 1.0)


def compute_vehicle_quality(mean_speed: float, yaw_rate_var: float, snr_db: float,
                            occlusion_rate: float, calibrated: bool,
                            histogram: Sequence[float], prior: Sequence[float],
                            q_min: float = 0.05, q_max: float = 1.0,
                            rarity_strength: float = 0.5) -> float:
    """Kinematic-spatio-temporal quality q_k = K_k * S_k * R_k / (1 + strength).

    Monotone: non-decreasing in speed, yaw-rate variance, SNR and label
    divergence; non-increasing in occlusion. Every input is measured on the
    vehicle; only q_k leaves it (inside the masked payload under secure
    aggregation).
    """
    q = (kinematic_novelty(mean_speed, yaw_rate_var)
         * sensor_fidelity(snr_db, occlusion_rate, calibrated)
         * rare_scene_bonus(histogram, prior, rarity_strength)
         / (1.0 + rarity_strength))
    return float(np.clip(q, q_min, q_max))


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
           "compute_quality_weight", "aggregation_weights", "kl_divergence",
           "kinematic_novelty", "sensor_fidelity", "rare_scene_bonus",
           "compute_vehicle_quality"]
