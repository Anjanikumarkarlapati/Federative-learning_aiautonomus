"""Client model: a lightweight multi-task perception network (PRD 8.1).

Design constraints taken straight from the PRD:

* MobileNetV3-style inverted-residual backbone with squeeze-excite, sized for
  onboard compute, hard-capped at ``model.max_parameters`` (default 6M).
* Three heads on a shared trunk, matching the two primary tasks: a **centre
  heatmap** whose spatial soft-argmax localises the object, a **size head** for
  (w, h), and a **trajectory head** for the short-horizon path - plus the class
  logits.  Global average pooling alone throws away the spatial information a
  box needs, so localisation goes through the heatmap: the softmax over the
  feature grid is differentiable and its expectation is sub-cell accurate, and
  the same attention map pools an object-focused embedding for the classifier.
* **GroupNorm, not BatchNorm.** Batch statistics are client-specific under
  Non-IID data and averaging them across vehicles is a well-known source of
  FedAvg divergence; GroupNorm is batch-independent, so every tensor in the
  state dict is safe to average (PRD 8.8).
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import ModelConfig
from ..utils.tensors import num_parameters


def _round_channels(channels: int, width: float, divisor: int = 8) -> int:
    scaled = max(divisor, int(channels * width + divisor / 2) // divisor * divisor)
    return int(scaled if scaled >= 0.9 * channels * width else scaled + divisor)


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.relu(self.fc1(scale), inplace=True)
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


def _norm(channels: int) -> nn.GroupNorm:
    groups = 1
    for candidate in (8, 4, 2):
        if channels % candidate == 0:
            groups = candidate
            break
    return nn.GroupNorm(groups, channels)


class InvertedResidual(nn.Module):
    """MobileNetV3 block: 1x1 expand -> depthwise kxk -> SE -> 1x1 project."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, expansion: int = 4,
                 kernel: int = 3, use_se: bool = True):
        super().__init__()
        hidden = in_ch * expansion
        self.use_residual = stride == 1 and in_ch == out_ch
        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1, bias=False), _norm(hidden), nn.Hardswish(inplace=True)
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel, stride=stride, padding=kernel // 2,
                      groups=hidden, bias=False),
            _norm(hidden), nn.Hardswish(inplace=True),
        )
        self.se = SqueezeExcite(hidden) if use_se else nn.Identity()
        self.project = nn.Sequential(nn.Conv2d(hidden, out_ch, 1, bias=False), _norm(out_ch))

    def forward(self, x: Tensor) -> Tensor:
        out = self.project(self.se(self.depthwise(self.expand(x))))
        return x + out if self.use_residual else out


class AVPerceptionNet(nn.Module):
    """Shared global model trained federatively across the fleet."""

    def __init__(self, cfg: ModelConfig, num_classes: int, traj_horizon: int):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.traj_horizon = traj_horizon

        stem = _round_channels(cfg.stem_channels, cfg.width)
        stages = [_round_channels(c, cfg.width) for c in cfg.stage_channels]
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem, 3, stride=1, padding=1, bias=False),
            _norm(stem), nn.Hardswish(inplace=True),
        )
        blocks = []
        in_ch = stem
        for i, out_ch in enumerate(stages):
            blocks.append(InvertedResidual(in_ch, out_ch, stride=2, expansion=4,
                                           kernel=5 if i else 3))
            blocks.append(InvertedResidual(out_ch, out_ch, stride=1, expansion=4))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        embed = _round_channels(cfg.embed_dim, cfg.width)
        self.neck = nn.Sequential(
            nn.Conv2d(in_ch, embed, 1, bias=False), _norm(embed), nn.Hardswish(inplace=True)
        )
        # Centre heatmap -> spatial soft-argmax -> (cx, cy)
        self.heatmap = nn.Conv2d(embed, 1, 3, padding=1)
        self.dropout = nn.Dropout(cfg.dropout)
        feat_dim = 2 * embed                      # [avg-pooled context | attention-pooled object]
        self.cls_head = nn.Sequential(
            nn.Linear(feat_dim, embed), nn.Hardswish(inplace=True),
            nn.Linear(embed, num_classes),
        )
        self.size_head = nn.Sequential(
            nn.Linear(feat_dim, embed // 2), nn.Hardswish(inplace=True),
            nn.Linear(embed // 2, cfg.box_dim - 2),
        )
        self.traj_head = nn.Sequential(
            nn.Linear(feat_dim + num_classes + 2, embed), nn.Hardswish(inplace=True),
            nn.Linear(embed, 2 * traj_horizon),
        )
        self._init_weights()
        count = num_parameters(self)
        if count > cfg.max_parameters:
            raise ValueError(
                f"model has {count:,} parameters, above the PRD 8.1 cap of "
                f"{cfg.max_parameters:,}; lower model.width or stage_channels"
            )
    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        features = self.neck(self.blocks(self.stem(x)))            # [B, E, h, w]
        b, _, h, w = features.shape

        # --- spatial soft-argmax over the centre heatmap ---------------------
        heat = self.heatmap(features).flatten(2)                   # [B, 1, h*w]
        attention = torch.softmax(heat, dim=-1)                    # [B, 1, h*w]
        grid_y, grid_x = torch.meshgrid(
            (torch.arange(h, device=x.device, dtype=features.dtype) + 0.5) / h,
            (torch.arange(w, device=x.device, dtype=features.dtype) + 0.5) / w,
            indexing="ij",
        )
        coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=0)   # [2, h*w]
        centre = attention.squeeze(1) @ coords.t()                 # [B, 2] = (cx, cy)

        # --- pooled descriptors ---------------------------------------------
        flat = features.flatten(2)                                 # [B, E, h*w]
        attended = (flat * attention).sum(dim=-1)                  # object-focused
        averaged = flat.mean(dim=-1)                               # scene context
        pooled = self.dropout(torch.cat([averaged, attended], dim=1))

        logits = self.cls_head(pooled)
        size = torch.sigmoid(self.size_head(pooled))               # (w, h) in (0, 1)
        box = torch.cat([centre, size], dim=1)
        traj = self.traj_head(torch.cat([pooled, logits.detach(), centre.detach()], dim=1))
        return {"logits": logits, "box": box, "trajectory": traj,
                "embedding": pooled, "heatmap": heat.view(b, 1, h, w)}

    @torch.no_grad()
    def predict(self, x: Tensor) -> Dict[str, Tensor]:
        self.eval()
        out = self(x)
        out["probs"] = F.softmax(out["logits"], dim=1)
        out["pred"] = out["probs"].argmax(dim=1)
        return out


def multitask_loss(outputs: Dict[str, Tensor], labels: Tensor, boxes: Tensor,
                   trajectories: Tensor, cfg: ModelConfig,
                   background_index: int = 0) -> Tuple[Tensor, Dict[str, float]]:
    """L = CE(class) + lambda_box * SmoothL1(box) + lambda_traj * SmoothL1(traj).

    Box and trajectory terms are masked on background samples: there is no
    object to localise or to predict a path for, so regressing them would inject
    pure noise into the shared representation.
    """
    cls_loss = F.cross_entropy(outputs["logits"], labels)
    foreground = (labels != background_index)
    if foreground.any():
        pred_box = outputs["box"][foreground]
        true_box = boxes[foreground]
        centre_loss = F.smooth_l1_loss(pred_box[:, :2], true_box[:, :2], beta=0.05)
        size_loss = F.smooth_l1_loss(pred_box[:, 2:], true_box[:, 2:], beta=0.05)
        box_loss = 2.0 * centre_loss + size_loss
        traj_loss = F.smooth_l1_loss(
            outputs["trajectory"][foreground], trajectories[foreground], beta=0.5
        )
    else:
        box_loss = outputs["box"].sum() * 0.0
        traj_loss = outputs["trajectory"].sum() * 0.0
    total = cls_loss + cfg.lambda_box * box_loss + cfg.lambda_traj * traj_loss
    parts = {"cls": cls_loss.detach().item(), "box": box_loss.detach().item(),
             "traj": traj_loss.detach().item()}
    return total, parts


def build_model(cfg: ModelConfig, num_classes: int, traj_horizon: int,
                seed: int | None = None) -> AVPerceptionNet:
    if seed is not None:
        torch.manual_seed(int(seed))
    return AVPerceptionNet(cfg, num_classes, traj_horizon)


__all__ = ["AVPerceptionNet", "build_model", "multitask_loss", "InvertedResidual"]
