"""Secure aggregation + V2I transport simulation (PRD 8.6 / 8.7 / 13).

**Secure aggregation.** Every pair of participating vehicles (i, j) derives a
shared seed and expands it into a pseudo-random mask ``m_ij``.  Vehicle i adds
``+m_ij`` for j > i and ``-m_ij`` for j < i, so

    sum_k (dW_k + mask_k) = sum_k dW_k

exactly, while any *individual* masked update is computationally
indistinguishable from noise.  The server therefore learns only the sum, never a
single vehicle's contribution - the property the compliance persona of PRD 4
asks for.  (Production deployments add Shamir-shared dropout recovery; the
simulator instead re-derives masks over the set that actually reported.)

**SecureChannel.** Simulates the TLS-protected V2I hop: per-message TLS/record
overhead, bandwidth-derived latency, link-uptime-driven drops and byte
accounting for the MB/round metric of PRD 12.1.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..utils.tensors import StateDict, state_add, state_zeros_like

TLS_OVERHEAD_BYTES = 29 + 5      # AEAD tag + record header, per message
HANDSHAKE_BYTES = 4_400          # amortised TLS 1.3 handshake per round


def _pair_seed(round_id: int, a: int, b: int, secret: int) -> int:
    lo, hi = (a, b) if a < b else (b, a)
    digest = hashlib.sha256(f"{secret}:{round_id}:{lo}:{hi}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


@dataclass
class SecureAggregationMasks:
    """Pairwise cancelling masks for one round."""

    round_id: int
    participants: Sequence[int]
    secret: int = 0xA17C
    scale: float = 1.0

    def mask_for(self, client_id: int, reference: StateDict) -> StateDict:
        mask = state_zeros_like(reference)
        for other in self.participants:
            if other == client_id:
                continue
            seed = _pair_seed(self.round_id, client_id, other, self.secret)
            generator = torch.Generator().manual_seed(seed)
            sign = 1.0 if client_id < other else -1.0
            for key, value in reference.items():
                mask[key] += sign * self.scale * torch.randn(
                    value.shape, generator=generator, dtype=torch.float32
                )
        return mask

    def apply(self, update: StateDict, client_id: int) -> StateDict:
        return state_add(update, self.mask_for(client_id, update), alpha=1.0)


def verify_masks_cancel(masks: SecureAggregationMasks, reference: StateDict,
                        atol: float = 1e-4) -> bool:
    """Unit-test helper: the masks of all participants must sum to ~zero."""
    total = state_zeros_like(reference)
    for client_id in masks.participants:
        mask = masks.mask_for(client_id, reference)
        for key in total:
            total[key] += mask[key]
    return all(float(v.abs().max()) <= atol for v in total.values())


@dataclass
class ChannelStats:
    uplink_bytes: int = 0
    downlink_bytes: int = 0
    messages: int = 0
    drops: int = 0

    @property
    def uplink_mb(self) -> float:
        return self.uplink_bytes / (1024.0 * 1024.0)

    @property
    def downlink_mb(self) -> float:
        return self.downlink_bytes / (1024.0 * 1024.0)

    @property
    def total_mb(self) -> float:
        return self.uplink_mb + self.downlink_mb

    def as_dict(self) -> Dict[str, float]:
        return {
            "uplink_mb": round(self.uplink_mb, 4),
            "downlink_mb": round(self.downlink_mb, 4),
            "total_mb": round(self.total_mb, 4),
            "messages": self.messages,
            "drops": self.drops,
        }


@dataclass
class SecureChannel:
    """TLS-encrypted V2I link with bandwidth, latency and drop simulation."""

    uplink_mbps: float = 12.0
    downlink_mbps: float = 40.0
    base_latency_s: float = 0.035
    stats: ChannelStats = field(default_factory=ChannelStats)

    def _transfer(self, payload: bytes, mbps: float) -> Tuple[int, float]:
        size = len(payload) + TLS_OVERHEAD_BYTES
        seconds = self.base_latency_s + (size * 8.0) / (mbps * 1e6)
        return size, seconds

    def send_up(self, payload: bytes, link_uptime: float = 1.0,
                generator: Optional[np.random.Generator] = None) -> Tuple[bool, int, float]:
        """Vehicle -> RSU.  Returns (delivered, bytes_on_wire, seconds)."""
        size, seconds = self._transfer(payload, self.uplink_mbps)
        self.stats.messages += 1
        self.stats.uplink_bytes += size
        rng = generator or np.random.default_rng()
        delivered = bool(rng.random() <= float(np.clip(link_uptime, 0.0, 1.0)))
        if not delivered:
            self.stats.drops += 1
        return delivered, size, seconds

    def send_down(self, payload: bytes, num_clients: int = 1) -> Tuple[int, float]:
        """RSU -> vehicles (broadcast counted once per recipient)."""
        size, seconds = self._transfer(payload, self.downlink_mbps)
        total = size * max(int(num_clients), 1)
        self.stats.messages += max(int(num_clients), 1)
        self.stats.downlink_bytes += total
        return total, seconds

    def handshake(self, num_clients: int) -> int:
        cost = HANDSHAKE_BYTES * max(int(num_clients), 0)
        self.stats.uplink_bytes += cost // 2
        self.stats.downlink_bytes += cost // 2
        return cost

    def reset(self) -> ChannelStats:
        previous, self.stats = self.stats, ChannelStats()
        return previous


__all__ = ["SecureAggregationMasks", "SecureChannel", "ChannelStats",
           "verify_masks_cancel", "TLS_OVERHEAD_BYTES"]
