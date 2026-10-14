"""On-vehicle update compression (PRD 8.6, NFR "<= 9 MB/round").

float16 + gzip alone buys a fixed ~2x.  To hit a hard V2I budget the uplink also
uses **top-k sparsification with error feedback**: the vehicle transmits only
the ``k`` largest-magnitude coordinates of its delta and *keeps the remainder*
in a local residual buffer that is added back into the next round's update.

    d_k^r  = dW_k^r + e_k^{r-1}
    s_k^r  = top_k(d_k^r)                      <- transmitted
    e_k^r  = d_k^r - s_k^r                     <- stays on the vehicle

Error feedback is what makes this lossless in the limit: no coordinate is ever
discarded, only delayed, so the sum of transmitted updates tracks the sum of
true updates and convergence is preserved.  The residual never leaves the car,
so the privacy property of FR-02 is untouched.

Zeroed coordinates cost almost nothing on the wire because gzip collapses the
resulting runs of zero bytes, so no separate sparse index format is needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

from ..utils.tensors import StateDict, state_zeros_like


@dataclass
class TopKCompressor:
    """Per-vehicle residual buffers (state lives on the vehicle, not the RSU)."""

    ratio: float = 1.0
    error_feedback: bool = True
    min_ratio: float = 0.02
    residuals: Dict[int, StateDict] = field(default_factory=dict)

    def compress(self, client_id: int, update: StateDict, steps: float = 1.0,
                 normalize: bool = False, fp16: bool = False) -> StateDict:
        """Top-k with error feedback.

        ``normalize``: work in step-normalised units, ``u = dW / tau + e``, and
        return ``tau * s`` so a FedNova aggregator that divides by ``tau`` sees
        exactly ``s`` - without it the residual, built from rounds with
        different ``tau``, is mis-scaled once the aggregator normalises.
        ``fp16``: round the output to half precision here, so the residual also
        carries the quantisation error and it is sent next round.
        """
        if self.ratio >= 1.0:
            return update
        tau = float(max(steps, 1.0)) if normalize else 1.0
        if tau != 1.0:
            update = {k: v / tau for k, v in update.items()}
        if self.error_feedback:
            residual = self.residuals.get(client_id)
            if residual is not None:
                update = {k: v + residual[k] for k, v in update.items()}
        sparse, kept = self._top_k(update, max(self.ratio, self.min_ratio))
        out = {k: v * tau for k, v in sparse.items()} if tau != 1.0 else sparse
        if fp16:
            out = {k: v.half().float() for k, v in out.items()}
        if self.error_feedback:
            self.residuals[client_id] = {k: update[k] - out[k] / tau for k in update}
        self.last_density = kept
        return out

    @staticmethod
    def _top_k(update: StateDict, ratio: float):
        flat = torch.cat([v.reshape(-1) for v in update.values()])
        total = flat.numel()
        k = max(1, int(round(ratio * total)))
        if k >= total:
            return update, 1.0
        threshold = torch.kthvalue(flat.abs(), total - k).values
        sparse = {key: torch.where(value.abs() > threshold, value, torch.zeros_like(value))
                  for key, value in update.items()}
        density = float(sum(int((v != 0).sum()) for v in sparse.values())) / total
        return sparse, density

    def restore(self, client_id: int, sent: StateDict, steps: float = 1.0) -> None:
        """NACK: the upload never arrived, so its coordinates go back into the
        residual instead of being lost - the next upload carries them.
        ``steps`` undoes the ``tau`` scaling of a step-normalised upload."""
        tau = float(max(steps, 1.0))
        sent = {k: v / tau for k, v in sent.items()} if tau != 1.0 else dict(sent)
        residual = self.residuals.get(client_id)
        self.residuals[client_id] = (sent if residual is None
                                     else {k: residual[k] + sent[k] for k in residual})

    def clear(self, client_id: Optional[int] = None) -> None:
        if client_id is None:
            self.residuals.clear()
        else:
            self.residuals.pop(client_id, None)


class BandwidthController:
    """Closed loop that keeps MB/round inside the configured budget.

    Raises sparsity when a round overruns the budget and relaxes it again when
    there is headroom, so the framework meets the non-functional requirement
    without hand-tuning the compression ratio per fleet size.
    """

    def __init__(self, budget_mb: float, initial_ratio: float = 1.0,
                 min_ratio: float = 0.02, tighten: float = 0.7, relax: float = 1.15):
        self.budget_mb = float(budget_mb)
        self.ratio = float(initial_ratio)
        self.min_ratio = float(min_ratio)
        self.tighten = float(tighten)
        self.relax = float(relax)
        self.history: list = []

    def update(self, observed_mb: float) -> float:
        self.history.append(observed_mb)
        if self.budget_mb <= 0:
            return self.ratio
        if observed_mb > self.budget_mb:
            self.ratio = max(self.min_ratio, self.ratio * self.tighten)
        elif observed_mb < 0.6 * self.budget_mb and self.ratio < 1.0:
            self.ratio = min(1.0, self.ratio * self.relax)
        return self.ratio


__all__ = ["TopKCompressor", "BandwidthController"]
