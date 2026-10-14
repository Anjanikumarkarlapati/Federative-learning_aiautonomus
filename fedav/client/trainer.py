"""ClientTrainer - on-vehicle local training (PRD 8.3, 9, 13, FR-01/02/05/10).

One round on vehicle k:

    receive w^(r)                                     (no raw data ever leaves)
    for e = 1..E_k:  w_k <- w_k - eta * grad F_k(w_k)  [+ mu (w_k - w^(r)) if FedProx]
    dW_k = w_k - w^(r)
    clip + Gaussian noise (optional DP) -> secure-aggregation mask
    send (client_id, n_k, q_k, dW_k, round_id, checksum)

``E_k`` is scaled to the vehicle's onboard compute (FR-10) so a weak ECU does
not stall the round, and the proximal term of FedProx damps client drift when
the local shard is extremely skewed (PRD 8.8).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn

from ..config import FLConfig
from ..data.federated import Vehicle
from ..data.quality import compute_quality_weight, compute_vehicle_quality
from ..models.perception import multitask_loss
from ..privacy.dp import add_gaussian_noise, clip_state
from ..utils.seeding import derive_seed
from ..utils.tensors import StateDict, state_norm, state_sub, to_float_state

REFERENCE_STEP_SECONDS = 0.012      # simulated wall-clock of one SGD step at capability 1.0


@dataclass
class ClientUpdate:
    """What a vehicle hands to the serializer (PRD 8.6)."""

    client_id: int
    round_id: int
    n_k: int
    q_k: float
    delta: StateDict
    local_epochs: int
    steps: int
    train_loss: float
    loss_parts: Dict[str, float] = field(default_factory=dict)
    update_norm: float = 0.0
    clipped: bool = False
    duration_s: float = 0.0
    byzantine: bool = False
    delta_c: Optional[StateDict] = None     # SCAFFOLD control-variate change, if enabled

    def meta(self) -> Dict[str, Any]:
        return {
            "client_id": self.client_id,
            "round_id": self.round_id,
            "n_k": self.n_k,
            "q_k": round(float(self.q_k), 6),
            "local_epochs": self.local_epochs,
            "steps": self.steps,
            "train_loss": round(float(self.train_loss), 6),
            "update_norm": round(float(self.update_norm), 6),
        }


class ClientTrainer:
    """Stateless trainer reused for every vehicle (one model instance, reloaded)."""

    def __init__(self, cfg: FLConfig, model: nn.Module, device: str = "cpu"):
        self.cfg = cfg
        self.model = model
        self.device = torch.device(device)
        # SCAFFOLD (Karimireddy et al. 2020): server control c and per-vehicle
        # c_k over the trainable parameters. c_k lives on the vehicle.
        self.c_global: Optional[Dict[str, torch.Tensor]] = None
        self.c_local: Dict[int, Dict[str, torch.Tensor]] = {}

    def _scaffold_controls(self, client_id: int):
        names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        if self.c_global is None:
            self.c_global = {n: torch.zeros_like(p, device="cpu")
                             for n, p in self.model.named_parameters() if p.requires_grad}
        c_k = self.c_local.get(client_id) or {n: torch.zeros_like(self.c_global[n]) for n in names}
        return names, c_k

    def scaffold_server_update(self, delta_cs: List[StateDict], num_clients: int) -> None:
        """c <- c + (1/N) sum_{k in S} delta c_k."""
        if self.c_global is None or not delta_cs:
            return
        for n in self.c_global:
            self.c_global[n] += sum(d[n] for d in delta_cs) / float(max(num_clients, 1))

    # ------------------------------------------------------------- scheduling
    def local_epochs_for(self, vehicle: Vehicle) -> int:
        """FR-10: E_k scaled to onboard compute, clamped to [min, max]."""
        cfg = self.cfg.train
        if not cfg.adaptive_epochs:
            return int(cfg.local_epochs)
        scaled = cfg.local_epochs * float(vehicle.profile.compute_capability)
        return int(np.clip(round(scaled), cfg.min_epochs, cfg.max_epochs))

    def learning_rate(self, round_id: int) -> float:
        """Cosine-decayed eta (PRD 10)."""
        cfg = self.cfg.train
        if not cfg.cosine_decay or cfg.rounds <= 1:
            return cfg.lr
        progress = min(max((round_id - 1) / max(cfg.rounds - 1, 1), 0.0), 1.0)
        return float(0.5 * cfg.lr * (1.0 + math.cos(math.pi * progress)))

    # ---------------------------------------------------------------- quality
    def quality_weight(self, vehicle: Vehicle, reference_histogram: np.ndarray) -> float:
        """q_k computed on-vehicle; only this scalar is transmitted (PRD 8.4/8.6)."""
        agg = self.cfg.aggregation
        if agg.quality_score == "kst":
            return compute_vehicle_quality(
                **vehicle.kst_inputs(), histogram=vehicle.profile.label_histogram,
                prior=reference_histogram, q_min=agg.q_min, q_max=agg.q_max,
                rarity_strength=agg.kst_rarity_strength)
        return compute_quality_weight(
            vehicle.profile.label_histogram,
            reference_histogram,
            vehicle.profile.quality_signal,
            q_min=agg.q_min,
            q_max=agg.q_max,
        )

    # ----------------------------------------------------------------- train
    def train(self, global_state: StateDict, vehicle: Vehicle, round_id: int,
              reference_histogram: np.ndarray,
              dp_sigma: float = 0.0) -> ClientUpdate:
        cfg = self.cfg
        torch.manual_seed(derive_seed(round_id, vehicle.client_id, base=cfg.seed))
        self.model.load_state_dict(global_state, strict=False)
        self.model.to(self.device).train()

        anchor = {k: v.detach().clone() for k, v in self.model.state_dict().items()
                  if torch.is_floating_point(v)}
        prox_ref = None
        if cfg.train.fedprox_mu > 0:
            prox_ref = [p.detach().clone() for p in self.model.parameters()]

        lr = self.learning_rate(round_id)
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=lr, momentum=cfg.train.momentum,
            weight_decay=cfg.train.weight_decay,
        )
        scaffold = cfg.train.scaffold
        if scaffold:
            sc_names, c_k = self._scaffold_controls(vehicle.client_id)
            correction = {n: (self.c_global[n] - c_k[n]).to(self.device) for n in sc_names}
            sc_params = dict(self.model.named_parameters())
        epochs = self.local_epochs_for(vehicle)
        loader = vehicle.loader(cfg.train.batch_size,
                                seed=derive_seed(round_id, vehicle.client_id, 7, base=cfg.seed))

        total_loss, steps = 0.0, 0
        parts_acc = {"cls": 0.0, "box": 0.0, "traj": 0.0}
        started = time.perf_counter()
        for _ in range(epochs):
            for images, labels, boxes, trajectories in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                boxes = boxes.to(self.device)
                trajectories = trajectories.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                outputs = self.model(images)
                loss, parts = multitask_loss(outputs, labels, boxes, trajectories, cfg.model)
                if prox_ref is not None:                    # FedProx: + mu/2 ||w_k - w||^2
                    prox = sum(((p - ref) ** 2).sum()
                               for p, ref in zip(self.model.parameters(), prox_ref))
                    loss = loss + 0.5 * cfg.train.fedprox_mu * prox
                loss.backward()
                if scaffold:                                # g <- g - c_k + c
                    for n in sc_names:
                        if sc_params[n].grad is not None:
                            sc_params[n].grad.add_(correction[n])
                if cfg.train.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.train.grad_clip)
                optimizer.step()
                total_loss += float(loss.detach())
                for key in parts_acc:
                    parts_acc[key] += parts[key]
                steps += 1
        wall = time.perf_counter() - started

        local_state = to_float_state(self.model.state_dict())
        delta = state_sub(local_state, anchor)              # dW transmitted, not w
        delta_c = None
        if scaffold and steps > 0:
            # option II: c_k+ = c_k - c + (x - y_k) / (K * eta_eff); with client
            # momentum beta the per-step displacement is ~ eta / (1 - beta)
            eta_eff = lr / max(1.0 - cfg.train.momentum, 1e-3)
            new_c = {n: c_k[n] - self.c_global[n] - delta[n].cpu() / (steps * eta_eff)
                     for n in sc_names}
            delta_c = {n: new_c[n] - c_k[n] for n in sc_names}
            self.c_local[vehicle.client_id] = new_c
        if vehicle.profile.byzantine:
            delta = self._poison(delta, round_id, vehicle.client_id)

        clipped = False
        norm_before = 0.0
        if cfg.dp.enabled:
            delta, norm_before, clipped = clip_state(delta, cfg.dp.clip_norm)
            if dp_sigma > 0:
                generator = torch.Generator().manual_seed(
                    derive_seed(round_id, vehicle.client_id, 91, base=cfg.seed)
                )
                delta = add_gaussian_noise(delta, dp_sigma, generator)
        update_norm = norm_before or state_norm(delta)

        # Simulated onboard duration: real wall-clock is machine dependent, so
        # the straggler logic uses a capability-normalised estimate instead.
        simulated = steps * REFERENCE_STEP_SECONDS / max(
            float(vehicle.profile.compute_capability), 1e-3)
        return ClientUpdate(
            client_id=vehicle.client_id,
            round_id=round_id,
            n_k=vehicle.n_k,
            q_k=self.quality_weight(vehicle, reference_histogram),
            delta=delta,
            local_epochs=epochs,
            steps=steps,
            train_loss=total_loss / max(steps, 1),
            loss_parts={k: v / max(steps, 1) for k, v in parts_acc.items()},
            update_norm=float(update_norm),
            clipped=clipped,
            duration_s=float(max(simulated, wall * 0.0)),
            byzantine=vehicle.profile.byzantine,
            delta_c=delta_c,
        )

    # ------------------------------------------------------------- adversary
    def _poison(self, delta: StateDict, round_id: int, client_id: int) -> StateDict:
        """Sign-flip + magnitude boost attack, used to exercise the robust
        aggregator of PRD 16 ('Byzantine / poisoned update')."""
        generator = torch.Generator().manual_seed(derive_seed(round_id, client_id, 13))
        return {
            key: -6.0 * value + torch.randn(value.shape, generator=generator) * 0.05
            for key, value in delta.items()
        }


__all__ = ["ClientTrainer", "ClientUpdate", "REFERENCE_STEP_SECONDS"]
