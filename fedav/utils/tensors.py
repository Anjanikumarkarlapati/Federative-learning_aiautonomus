"""Parameter-vector utilities shared by clients, server and privacy modules.

Everything in the framework operates on an ordered ``state_dict`` of float
tensors.  Flattening to a single 1-D vector makes the FedAvg math (PRD 8.4),
the DP clipping (PRD 8.7) and the convergence test (PRD 8.9) exact and easy to
unit-test.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, List, Sequence

import torch

StateDict = Dict[str, torch.Tensor]


def clone_state(state: StateDict) -> StateDict:
    return OrderedDict((k, v.detach().clone()) for k, v in state.items())


def to_float_state(state: StateDict) -> StateDict:
    """Keep only floating point entries (buffers such as num_batches_tracked
    are integer counters and must not be averaged as if they were weights)."""
    return OrderedDict(
        (k, v) for k, v in state.items() if torch.is_floating_point(v)
    )


def flatten(state: StateDict, keys: Sequence[str] | None = None) -> torch.Tensor:
    keys = list(state.keys()) if keys is None else list(keys)
    if not keys:
        return torch.zeros(0)
    return torch.cat([state[k].reshape(-1).to(torch.float32) for k in keys])


def unflatten(vector: torch.Tensor, reference: StateDict) -> StateDict:
    out: StateDict = OrderedDict()
    offset = 0
    for key, ref in reference.items():
        count = ref.numel()
        out[key] = vector[offset: offset + count].view_as(ref).to(ref.dtype)
        offset += count
    if offset != vector.numel():
        raise ValueError(f"vector size {vector.numel()} != reference size {offset}")
    return out


def state_sub(a: StateDict, b: StateDict) -> StateDict:
    """Element-wise a - b (used to form the transmitted delta dW)."""
    return OrderedDict((k, (a[k].to(torch.float32) - b[k].to(torch.float32))) for k in a)


def state_add(a: StateDict, b: StateDict, alpha: float = 1.0) -> StateDict:
    return OrderedDict(
        (k, (a[k].to(torch.float32) + alpha * b[k].to(torch.float32))) for k in a
    )


def state_scale(a: StateDict, alpha: float) -> StateDict:
    return OrderedDict((k, a[k].to(torch.float32) * alpha) for k in a)


def state_zeros_like(a: StateDict) -> StateDict:
    return OrderedDict((k, torch.zeros_like(v, dtype=torch.float32)) for k, v in a.items())


def state_norm(a: StateDict) -> float:
    """L2 norm of the flattened state - the quantity used by FR-07/PRD 8.9."""
    total = 0.0
    for value in a.values():
        total += float(value.to(torch.float32).pow(2).sum())
    return total ** 0.5


def weighted_average(states: Sequence[StateDict], weights: Sequence[float]) -> StateDict:
    """Sum_k alpha_k * state_k with sum(alpha) expected to be 1."""
    if not states:
        raise ValueError("no states to average")
    if len(states) != len(weights):
        raise ValueError("states/weights length mismatch")
    out = state_zeros_like(states[0])
    for state, weight in zip(states, weights):
        for key in out:
            out[key] += state[key].to(torch.float32) * float(weight)
    return out


def num_parameters(module: torch.nn.Module, trainable_only: bool = True) -> int:
    params: Iterable[torch.nn.Parameter] = module.parameters()
    return sum(p.numel() for p in params if (p.requires_grad or not trainable_only))


def state_numel(state: StateDict) -> int:
    return int(sum(v.numel() for v in state.values()))


def trimmed_mean(stack: torch.Tensor, weights: torch.Tensor, trim_fraction: float) -> torch.Tensor:
    """Coordinate-wise weighted trimmed mean (PRD 15/16 - Byzantine robustness).

    ``stack`` is [K, D]; for each coordinate the ``trim`` smallest and ``trim``
    largest values are dropped and the remaining weights are renormalised.
    """
    k = stack.shape[0]
    trim = int(trim_fraction * k)
    if k <= 2 or trim <= 0:
        w = (weights / weights.sum()).unsqueeze(1)
        return (stack * w).sum(dim=0)
    trim = min(trim, (k - 1) // 2)
    order = torch.argsort(stack, dim=0)
    ranks = torch.argsort(order, dim=0)                       # rank of each row
    keep = (ranks >= trim) & (ranks < k - trim)               # [K, D] mask
    w = weights.unsqueeze(1).expand_as(stack) * keep
    denom = w.sum(dim=0).clamp_min(1e-12)
    return (stack * w).sum(dim=0) / denom


__all__ = [
    "StateDict", "clone_state", "to_float_state", "flatten", "unflatten",
    "state_sub", "state_add", "state_scale", "state_zeros_like", "state_norm",
    "weighted_average", "num_parameters", "state_numel", "trimmed_mean",
]
