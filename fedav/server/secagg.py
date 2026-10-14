"""Quality-hiding, budget-tiered secure aggregation over V2I (claims K1 + K2).

What the road-side unit (RSU) learns
------------------------------------
Only round-level *sums*. Each vehicle k folds its private aggregation weight
``w_k = q_k * n_k / n_ref`` (sensor-quality score times shard size) and its
local step count ``tau_k`` into the masked payload itself:

    x_k = [ w_k * u_k[support] ,  w_k ,  w_k * tau_k ]        u_k = dW_k / tau_k

and adds pairwise masks that cancel in the sum. The RSU recovers

    S = sum_k w_k u_k,   W = sum_k w_k,   T = sum_k w_k tau_k

and forms the normalised (FedNova) quality-weighted update ``d = (T/W) * S/W``
without ever seeing an individual q_k, n_k, tau_k or update. A plaintext
quality-aware aggregator has to receive each q_k; this one does not.

Why the payload stays small
---------------------------
Masks make every transmitted coordinate look random, so per-vehicle top-k
cannot be combined with masking without sending the whole dense vector. Here
every vehicle transmits the *same, publicly known* ordered support: the
coordinates where the previous round's global update was largest (all vehicles
already hold it from the multicast broadcast; round 1 uses a seeded random
support). No index list is sent. Whatever is not transmitted stays in the
vehicle's error-feedback residual.

Budget tiers (K2)
-----------------
The RSU assigns each vehicle a *prefix length* of that ordered support from a
small public ladder L_1 < ... < L_m, so a vehicle on a fast link sends more
coordinates and one on a slow link fewer, keeping every vehicle's airtime inside
an equal share of the round's airtime budget. Weights are sent per segment
``[L_{t-1}, L_t)`` so each segment is normalised by the weight of exactly the
vehicles that covered it. A pair of vehicles masks only the coordinates both of
them send, so the masks still cancel.

Arithmetic
----------
Values are fixed-point integers in the ring Z_{2^32}; masks are uniform ring
elements from a PRG, so cancellation is *exact* (no floating-point residue).
Update coordinates are clipped to a public bound ``B`` broadcast by the RSU;
the clipped excess returns to the vehicle's residual, so nothing is lost.

Pair seeds come from Diffie-Hellman over the RFC 3526 2048-bit MODP group: each
vehicle publishes ``g^a mod p`` once, and a pair's seed is
``SHA-256(g^{ab} mod p || round)``. The RSU relays public keys but cannot derive
any pair seed. (The simulator derives private keys from the run seed for
reproducibility; a deployment draws them in the vehicle's secure element.)

Dropouts
--------
If a selected vehicle's upload is lost or fails its CRC check, the survivors
reveal the pair seeds they shared with it (32 bytes per dropped peer, counted on
the uplink), and the RSU removes those mask terms. This is the unmasking step of
Bonawitz et al. (CCS 2017) without the Shamir layer that guards against a
malicious RSU; the dropped vehicle's input was never received, so nothing about
it is revealed.
"""
from __future__ import annotations

import hashlib
import math
import struct
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# RFC 3526, group 14 (2048-bit MODP), generator 2.
MODP_P = int(
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22"
    "514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6"
    "F44C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3D"
    "C2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB"
    "9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3BE39E772C180E8603"
    "9B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA0510"
    "15728E5A8AACAA68FFFFFFFFFFFFFFFF", 16)
MODP_G = 2
PUBLIC_KEY_BYTES = 256
SEED_REVEAL_BYTES = 32
RING_BITS = 32
_RING = 1 << RING_BITS
_MAGIC = b"FLSA"
_HEADER = struct.Struct("<4sIIIII")   # magic, client, round, tier, n_values, crc32


# ----------------------------------------------------------------------- keys
@dataclass(frozen=True)
class KeyPair:
    private: int
    public: int


def keypair(seed: int, client_id: int) -> KeyPair:
    digest = hashlib.sha256(f"flav-dh:{seed}:{client_id}".encode()).digest()
    private = int.from_bytes(digest * 8, "big") % (MODP_P - 3) + 2
    return KeyPair(private, pow(MODP_G, private, MODP_P))


def pair_seed(own: KeyPair, peer_public: int, round_id: int) -> int:
    shared = pow(peer_public, own.private, MODP_P)
    digest = hashlib.sha256(shared.to_bytes(PUBLIC_KEY_BYTES, "big")
                            + round_id.to_bytes(8, "big")).digest()
    return int.from_bytes(digest[:8], "big")


def prg(seed: int, n: int) -> np.ndarray:
    """``n`` uniform elements of Z_{2^32}."""
    return np.random.default_rng(seed).integers(0, _RING, size=n, dtype=np.uint64).astype(np.uint32)


# ------------------------------------------------------------- fixed point
@dataclass(frozen=True)
class FixedPoint:
    """Signed fixed point in Z_{2^32} with headroom for ``max_parties`` summands."""

    bound: float          # |value| <= bound before encoding (values are clipped)
    max_parties: int = 64

    @property
    def scale(self) -> float:
        headroom = int(math.ceil(math.log2(max(self.max_parties, 2))))
        return float(2 ** (RING_BITS - 1 - headroom)) / self.bound

    def encode(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (ring elements, clipped excess in real units)."""
        clipped = np.clip(values, -self.bound, self.bound)
        q = np.rint(clipped * self.scale).astype(np.int64)
        return (q % _RING).astype(np.uint32), values - q / self.scale

    def decode(self, ring: np.ndarray) -> np.ndarray:
        signed = ring.astype(np.int64)
        signed[signed >= _RING // 2] -= _RING
        return signed / self.scale


# ---------------------------------------------------------------- support
def shared_support(previous_update: Optional[np.ndarray], numel: int, size: int,
                   seed: int, round_id: int) -> np.ndarray:
    """Ordered coordinate list every vehicle can compute on its own.

    Largest-magnitude coordinates of the previous global update first; round 1
    (or an all-zero update) falls back to a seeded random permutation.
    """
    size = int(min(max(size, 1), numel))
    if previous_update is None or not np.any(previous_update):
        perm = np.random.default_rng([seed, round_id, 29]).permutation(numel)
        return perm[:size]
    magnitude = np.abs(previous_update)
    top = np.argpartition(-magnitude, size - 1)[:size]
    return top[np.argsort(-magnitude[top], kind="stable")]


def tier_ladder(max_size: int, min_size: int, levels: int = 5) -> List[int]:
    """Public prefix lengths, geometric from ``min_size`` to ``max_size``."""
    if levels <= 1 or max_size <= min_size:
        return [int(max_size)]
    ratio = (max_size / min_size) ** (1.0 / (levels - 1))
    ladder = sorted({int(round(min_size * ratio ** i)) for i in range(levels)})
    ladder[-1] = int(max_size)
    return ladder


def assign_tiers(rates_mbps: Dict[int, float], ladder: Sequence[int],
                 airtime_budget_s: float, bytes_per_value: int = 4,
                 overhead_bytes: int = 0) -> Dict[int, int]:
    """Airtime-equalising tier choice: each vehicle gets the longest prefix it
    can send within an equal share of the round's uplink airtime budget.
    Returns client -> tier index (0-based). ``airtime_budget_s <= 0`` -> top tier."""
    top = len(ladder) - 1
    if airtime_budget_s <= 0 or not rates_mbps:
        return {c: top for c in rates_mbps}
    share = airtime_budget_s / len(rates_mbps)
    tiers = {}
    for client, rate in rates_mbps.items():
        allowed = share * rate * 1e6 / 8.0 - overhead_bytes
        tier = 0
        for t, length in enumerate(ladder):
            if (length + 2 * (t + 1)) * bytes_per_value <= allowed:
                tier = t
        tiers[client] = tier
    return tiers


# ------------------------------------------------------------------ vehicle
@dataclass
class MaskedUpload:
    client_id: int
    tier: int
    payload: bytes
    clipped_excess: np.ndarray        # stays on the vehicle (error feedback)


def vehicle_encode(client_id: int, round_id: int, keys: KeyPair,
                   peers: Dict[int, int], peer_tiers: Dict[int, int],
                   ladder: Sequence[int], tier: int, values: np.ndarray,
                   weight: float, steps: float, value_fp: FixedPoint,
                   scalar_fp: FixedPoint) -> MaskedUpload:
    """Vehicle side: weight, clip, encode and mask its prefix of the support.

    ``values`` is the vehicle's update (already step-normalised) on the first
    ``ladder[tier]`` support coordinates. ``peers`` maps every other selected
    vehicle to its public key; ``peer_tiers`` to its assigned tier.
    """
    length = ladder[tier]
    ring_vals, excess = value_fp.encode(weight * values[:length])
    seg = np.array([weight, weight * steps] * (tier + 1), dtype=np.float64)
    ring_seg, _ = scalar_fp.encode(seg)
    vec = np.concatenate([ring_vals, ring_seg])
    for peer, public in peers.items():
        if peer == client_id:
            continue
        shared_tier = min(tier, peer_tiers[peer])
        common = ladder[shared_tier]
        mask = prg(pair_seed(keys, public, round_id), ladder[-1] + 2 * len(ladder))
        m = np.concatenate([mask[:common], mask[ladder[-1]:ladder[-1] + 2 * (shared_tier + 1)]])
        idx = np.concatenate([np.arange(common), length + np.arange(2 * (shared_tier + 1))])
        if client_id < peer:
            vec[idx] = vec[idx] + m          # uint32 arithmetic wraps mod 2^32
        else:
            vec[idx] = vec[idx] - m
    body = vec.astype("<u4").tobytes()
    header = _HEADER.pack(_MAGIC, client_id, round_id, tier, vec.size,
                          zlib.crc32(body) & 0xFFFFFFFF)
    # excess in the *unweighted* update domain, for the residual
    return MaskedUpload(client_id, tier, header + body,
                        excess / weight if weight > 0 else np.zeros_like(excess))


class PayloadError(ValueError):
    pass


def parse_upload(raw: bytes) -> Tuple[int, int, int, np.ndarray]:
    magic, client, round_id, tier, n, crc = _HEADER.unpack_from(raw, 0)
    body = raw[_HEADER.size:]
    if magic != _MAGIC or (zlib.crc32(body) & 0xFFFFFFFF) != crc or len(body) != 4 * n:
        raise PayloadError(f"bad secure-aggregation payload from client {client}")
    return client, round_id, tier, np.frombuffer(body, dtype="<u4").astype(np.uint32)


# ---------------------------------------------------------------------- RSU
@dataclass
class RecoveredSums:
    S: np.ndarray                 # sum w u, over the ordered support (length ladder[-1])
    W: np.ndarray                 # per-segment weight sums
    T: np.ndarray                 # per-segment sum w * tau
    seeds_revealed: int = 0


def rsu_aggregate(uploads: Dict[int, np.ndarray], tiers: Dict[int, int],
                  selected_keys: Dict[int, KeyPair], round_id: int,
                  ladder: Sequence[int], value_fp: FixedPoint,
                  scalar_fp: FixedPoint) -> RecoveredSums:
    """Sum the ring vectors that arrived and remove dropped vehicles' masks.

    ``selected_keys`` holds the *surviving vehicles'* key pairs only so the
    simulator can compute the seeds they reveal for each dropped peer; the RSU
    never reads a private key otherwise.
    """
    full = ladder[-1]
    m = len(ladder)
    S = np.zeros(full, dtype=np.uint32)
    seg = np.zeros(2 * m, dtype=np.uint32)
    for client, vec in uploads.items():
        length = ladder[tiers[client]]
        S[:length] += vec[:length]
        seg[:2 * (tiers[client] + 1)] += vec[length:]

    dropped = [c for c in tiers if c not in uploads]
    revealed = 0
    for survivor in uploads:
        for gone in dropped:
            seed = pair_seed(selected_keys[survivor], selected_keys[gone].public, round_id)
            revealed += 1
            shared_tier = min(tiers[survivor], tiers[gone])
            common = ladder[shared_tier]
            mask = prg(seed, full + 2 * m)
            sign_add = survivor < gone
            if sign_add:
                S[:common] -= mask[:common]
                seg[:2 * (shared_tier + 1)] -= mask[full:full + 2 * (shared_tier + 1)]
            else:
                S[:common] += mask[:common]
                seg[:2 * (shared_tier + 1)] += mask[full:full + 2 * (shared_tier + 1)]

    scalars = scalar_fp.decode(seg)
    return RecoveredSums(S=value_fp.decode(S), W=scalars[0::2], T=scalars[1::2],
                         seeds_revealed=revealed)


def combine_segments(sums: RecoveredSums, ladder: Sequence[int],
                     normalize_steps: bool) -> np.ndarray:
    """Per-segment weighted mean, FedNova-rescaled: d_t = (T_t/W_t) * S_t/W_t."""
    out = np.zeros(ladder[-1], dtype=np.float64)
    start = 0
    for t, end in enumerate(ladder):
        W = sums.W[t]
        if W > 1e-12:
            factor = (sums.T[t] / W) if normalize_steps else 1.0
            out[start:end] = factor * sums.S[start:end] / W
        start = end
    return out


__all__ = ["KeyPair", "keypair", "pair_seed", "prg", "FixedPoint", "shared_support",
           "tier_ladder", "assign_tiers", "vehicle_encode", "parse_upload", "PayloadError",
           "rsu_aggregate", "combine_segments", "RecoveredSums", "MaskedUpload",
           "PUBLIC_KEY_BYTES", "SEED_REVEAL_BYTES"]
