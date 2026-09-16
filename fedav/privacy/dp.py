"""Differential privacy for client updates (PRD 8.7, FR-05).

Pipeline per selected vehicle, per round:

1. **Clip** the update to an L2 ball of radius ``C``:
   ``dW <- dW * min(1, C / ||dW||_2)``.  This bounds the *sensitivity* of the
   aggregate to one vehicle: replacing a client changes the sum by at most
   ``2C`` for an unbounded neighbourhood, ``C`` for the add/remove
   neighbourhood used here.
2. **Perturb** with Gaussian noise ``N(0, sigma^2 I)`` where
   ``sigma = z * C`` and ``z`` (the noise multiplier) is calibrated so that the
   composition over all ``R`` rounds satisfies the configured ``(eps, delta)``.
3. Because the server only ever observes the *sum* under secure aggregation
   (``secure.py``), the noise can be divided across the ``|S|`` participants -
   the aggregate carries the full ``sigma`` while each client adds
   ``sigma / sqrt(|S|)``.

Two calibrations are provided: the classical analytic Gaussian mechanism
(Balle & Wang, 2018) for a single release, and an RDP accountant for the
sub-sampled Gaussian mechanism (Mironov et al., 2019) composed over rounds -
the latter is what ``calibrate_noise_multiplier`` uses, since federated training
is exactly a sub-sampled Gaussian mechanism repeated ``R`` times.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch

from ..utils.tensors import StateDict, state_norm, state_scale

DEFAULT_ALPHAS: Tuple[float, ...] = tuple(
    [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)) + [128, 256]
)


def _logsumexp(values: Sequence[float]) -> float:
    """Numerically stable log(sum(exp(v))) - the binomial terms overflow for
    large alpha / small sigma otherwise."""
    top = max(values)
    if top == float("-inf"):
        return top
    return top + math.log(sum(math.exp(v - top) for v in values))


def _phi(t: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(t / math.sqrt(2.0)))


def clip_state(update: StateDict, clip_norm: float) -> Tuple[StateDict, float, bool]:
    """Project ``update`` onto the L2 ball of radius ``clip_norm``."""
    norm = state_norm(update)
    if clip_norm <= 0 or norm <= clip_norm or norm == 0.0:
        return update, norm, False
    return state_scale(update, clip_norm / norm), norm, True


def add_gaussian_noise(update: StateDict, sigma: float,
                       generator: Optional[torch.Generator] = None) -> StateDict:
    """Add i.i.d. ``N(0, sigma^2)`` noise to every coordinate."""
    if sigma <= 0:
        return update
    noised = {}
    for key, value in update.items():
        noise = torch.normal(mean=0.0, std=float(sigma), size=value.shape,
                             generator=generator, dtype=torch.float32)
        noised[key] = value.to(torch.float32) + noise
    return noised


def analytic_gaussian_sigma(sensitivity: float, epsilon: float, delta: float,
                            tol: float = 1e-12) -> float:
    """Exact sigma for a *single* (eps, delta)-DP Gaussian release.

    Solves Balle & Wang's condition
    ``Phi(s/2 - eps/s) - e^eps * Phi(-s/2 - eps/s) = delta`` for ``s = Delta/sigma``
    by bisection; unlike the textbook ``sqrt(2 ln(1.25/delta))/eps`` bound this
    stays tight (and valid) for ``eps > 1``, which the PRD default eps = 8 is.
    """
    if epsilon <= 0 or not (0 < delta < 1):
        raise ValueError("epsilon must be > 0 and delta in (0, 1)")

    def budget(s: float) -> float:
        return _phi(s / 2 - epsilon / s) - math.exp(epsilon) * _phi(-s / 2 - epsilon / s)

    lo, hi = 1e-6, 1.0
    while budget(hi) < delta:                 # grow until the bracket contains the root
        hi *= 2.0
        if hi > 1e6:
            break
    while hi - lo > tol * max(1.0, hi):
        mid = 0.5 * (lo + hi)
        if budget(mid) < delta:
            lo = mid
        else:
            hi = mid
    s = 0.5 * (lo + hi)
    return float(sensitivity / s)


def rdp_subsampled_gaussian(q: float, noise_multiplier: float,
                            alphas: Sequence[float] = DEFAULT_ALPHAS) -> List[float]:
    """RDP epsilon(alpha) of one sub-sampled Gaussian step (Poisson sampling rate q).

    Integer alphas use the exact binomial expansion; fractional alphas are
    bounded with the standard ``q^2 alpha / sigma^2`` small-q bound.
    """
    if noise_multiplier <= 0:
        return [float("inf")] * len(alphas)
    q = min(max(q, 0.0), 1.0)
    if q == 0.0:
        return [0.0] * len(alphas)
    sigma2 = noise_multiplier ** 2
    out: List[float] = []
    for alpha in alphas:
        if q == 1.0:
            out.append(alpha / (2.0 * sigma2))
            continue
        if float(alpha).is_integer() and alpha >= 2:
            a = int(alpha)
            log_terms = [
                math.lgamma(a + 1) - math.lgamma(k + 1) - math.lgamma(a - k + 1)
                + (a - k) * math.log1p(-q) + k * math.log(q)
                + (k * k - k) / (2.0 * sigma2)
                for k in range(a + 1)
            ]
            out.append(_logsumexp(log_terms) / (a - 1))
        else:
            out.append(float(q * q * alpha / sigma2))
    return out


def rdp_to_dp(rdp: Sequence[float], alphas: Sequence[float], delta: float
              ) -> Tuple[float, float]:
    """Convert composed RDP to (eps, delta)-DP, returning (eps, best alpha)."""
    best_eps, best_alpha = float("inf"), float(alphas[0])
    for eps_rdp, alpha in zip(rdp, alphas):
        if alpha <= 1:
            continue
        eps = eps_rdp + math.log1p(-1.0 / alpha) - (math.log(delta) + math.log(alpha)) / (alpha - 1)
        if eps < best_eps:
            best_eps, best_alpha = eps, float(alpha)
    return float(max(best_eps, 0.0)), best_alpha


def compose_epsilon(q: float, noise_multiplier: float, steps: int, delta: float,
                    alphas: Sequence[float] = DEFAULT_ALPHAS) -> float:
    """Total privacy loss after ``steps`` sub-sampled Gaussian rounds."""
    if steps <= 0:
        return 0.0
    per_step = rdp_subsampled_gaussian(q, noise_multiplier, alphas)
    composed = [value * steps for value in per_step]
    eps, _ = rdp_to_dp(composed, alphas, delta)
    return eps


def calibrate_noise_multiplier(target_epsilon: float, delta: float, q: float, steps: int,
                               lo: float = 0.2, hi: float = 512.0, tol: float = 1e-3) -> float:
    """Smallest ``z`` whose ``steps``-fold composition stays within the budget."""
    if steps <= 0 or target_epsilon <= 0:
        return 0.0
    if compose_epsilon(q, hi, steps, delta) > target_epsilon:
        return float(hi)
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if compose_epsilon(q, mid, steps, delta) > target_epsilon:
            lo = mid
        else:
            hi = mid
    return float(hi)


@dataclass
class PrivacyAccountant:
    """Tracks the (eps, delta) spent over the federated run (PRD 8.7 / 20)."""

    target_epsilon: float
    delta: float
    sampling_rate: float
    clip_norm: float = 1.0
    steps: int = 0
    noise_multiplier: float = 0.0

    @classmethod
    def calibrated(cls, target_epsilon: float, delta: float, sampling_rate: float,
                   rounds: int, clip_norm: float = 1.0) -> "PrivacyAccountant":
        z = calibrate_noise_multiplier(target_epsilon, delta, sampling_rate, rounds)
        return cls(target_epsilon=target_epsilon, delta=delta,
                   sampling_rate=sampling_rate, clip_norm=clip_norm, noise_multiplier=z)

    @property
    def sigma(self) -> float:
        """Std-dev of the noise added to the *aggregate* sum of clipped updates."""
        return float(self.noise_multiplier * self.clip_norm)

    def client_sigma(self, num_participants: int) -> float:
        """Per-client share under secure aggregation: sum of |S| independent
        N(0, sigma^2/|S|) draws is exactly N(0, sigma^2)."""
        return float(self.sigma / math.sqrt(max(int(num_participants), 1)))

    def step(self, count: int = 1) -> None:
        self.steps += int(count)

    def spent(self) -> float:
        return compose_epsilon(self.sampling_rate, self.noise_multiplier,
                               self.steps, self.delta)

    def summary(self) -> dict:
        return {
            "epsilon_spent": round(self.spent(), 4),
            "epsilon_budget": self.target_epsilon,
            "delta": self.delta,
            "noise_multiplier": round(self.noise_multiplier, 4),
            "clip_norm": self.clip_norm,
            "sampling_rate": round(self.sampling_rate, 4),
            "rounds_accounted": self.steps,
        }


__all__ = ["clip_state", "add_gaussian_noise", "analytic_gaussian_sigma",
           "rdp_subsampled_gaussian", "rdp_to_dp", "compose_epsilon",
           "calibrate_noise_multiplier", "PrivacyAccountant"]
