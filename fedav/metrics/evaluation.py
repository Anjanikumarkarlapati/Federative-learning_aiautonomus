"""Evaluation metrics (PRD 12.1).

* **Accuracy** - top-1 over the global held-out test set.
* **mAP@0.5** - a detection score: a prediction counts as a true positive when
  the predicted class matches *and* the predicted box overlaps the ground truth
  with IoU >= 0.5.  Average precision uses all-point interpolation over the
  precision/recall curve obtained by sorting detections by confidence, then
  averaged over foreground classes.
* **ADE / FDE** - average and final displacement error of the predicted
  trajectory, the standard short-horizon prediction metrics.
* **Non-IID robustness** - accuracy evaluated on every vehicle's *local* test
  split; the dispersion of that vector (std, min, worst-decile) is what
  PRD 12.1 calls "robustness to Non-IID".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from ..config import ModelConfig
from ..models.perception import multitask_loss

BACKGROUND = 0


def box_iou_cxcywh(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU between two [N, 4] sets of (cx, cy, w, h) boxes, element-wise."""
    def corners(x: torch.Tensor) -> torch.Tensor:
        cx, cy, w, h = x.unbind(dim=-1)
        return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)

    pa, pb = corners(a), corners(b)
    x1 = torch.maximum(pa[:, 0], pb[:, 0])
    y1 = torch.maximum(pa[:, 1], pb[:, 1])
    x2 = torch.minimum(pa[:, 2], pb[:, 2])
    y2 = torch.minimum(pa[:, 3], pb[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (pa[:, 2] - pa[:, 0]).clamp(min=0) * (pa[:, 3] - pa[:, 1]).clamp(min=0)
    area_b = (pb[:, 2] - pb[:, 0]).clamp(min=0) * (pb[:, 3] - pb[:, 1]).clamp(min=0)
    union = (area_a + area_b - inter).clamp(min=1e-9)
    return inter / union


def average_precision(scores: np.ndarray, matched: np.ndarray, num_positives: int) -> float:
    """All-point-interpolated AP from confidence scores and TP/FP flags."""
    if num_positives == 0:
        return float("nan")
    if scores.size == 0:
        return 0.0
    order = np.argsort(-scores)
    tp = matched[order].astype(np.float64)
    fp = 1.0 - tp
    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
    recall = tp_cum / num_positives
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    # monotone-decreasing precision envelope
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[precision[0] if precision.size else 0.0], precision])
    return float(np.sum(np.diff(recall) * precision[1:]))


@dataclass
class EvalResult:
    accuracy: float = 0.0
    balanced_accuracy: float = 0.0
    macro_f1: float = 0.0
    mAP50: float = 0.0
    ade: float = 0.0
    fde: float = 0.0
    loss: float = 0.0
    per_class_ap: Dict[int, float] = field(default_factory=dict)
    samples: int = 0

    def as_dict(self) -> Dict[str, float]:
        return {
            "accuracy": round(self.accuracy, 5),
            "balanced_accuracy": round(self.balanced_accuracy, 5),
            "macro_f1": round(self.macro_f1, 5),
            "mAP50": round(self.mAP50, 5),
            "ade": round(self.ade, 5),
            "fde": round(self.fde, 5),
            "loss": round(self.loss, 5),
            "samples": self.samples,
        }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, cfg: ModelConfig,
             num_classes: int, device: str = "cpu",
             iou_threshold: float = 0.5) -> EvalResult:
    model.eval().to(device)
    correct = total = 0
    loss_sum = 0.0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    scores: List[np.ndarray] = []
    matched: List[np.ndarray] = []
    pred_classes: List[np.ndarray] = []
    positives = np.zeros(num_classes, dtype=np.int64)
    ade_sum = fde_sum = fg_count = 0.0

    for images, labels, boxes, trajectories in loader:
        images, labels = images.to(device), labels.to(device)
        boxes, trajectories = boxes.to(device), trajectories.to(device)
        outputs = model(images)
        loss, _ = multitask_loss(outputs, labels, boxes, trajectories, cfg)
        loss_sum += float(loss) * labels.numel()

        probs = F.softmax(outputs["logits"], dim=1)
        confidence, prediction = probs.max(dim=1)
        correct += int((prediction == labels).sum())
        total += int(labels.numel())
        for t, p in zip(labels.tolist(), prediction.tolist()):
            confusion[t, p] += 1

        iou = box_iou_cxcywh(outputs["box"], boxes)
        is_tp = (prediction == labels) & (iou >= iou_threshold) & (labels != BACKGROUND)
        foreground = labels != BACKGROUND
        keep = prediction != BACKGROUND
        scores.append(confidence[keep].cpu().numpy())
        matched.append(is_tp[keep].cpu().numpy())
        pred_classes.append(prediction[keep].cpu().numpy())
        positives += np.bincount(labels[foreground].cpu().numpy(), minlength=num_classes)

        if foreground.any():
            pred_traj = outputs["trajectory"][foreground].view(-1, trajectories.shape[1] // 2, 2)
            true_traj = trajectories[foreground].view_as(pred_traj)
            displacement = torch.linalg.norm(pred_traj - true_traj, dim=-1)
            ade_sum += float(displacement.mean(dim=1).sum())
            fde_sum += float(displacement[:, -1].sum())
            fg_count += int(foreground.sum())

    all_scores = np.concatenate(scores) if scores else np.zeros(0)
    all_matched = np.concatenate(matched) if matched else np.zeros(0, dtype=bool)
    all_pred = np.concatenate(pred_classes) if pred_classes else np.zeros(0, dtype=np.int64)

    per_class: Dict[int, float] = {}
    for cls in range(num_classes):
        if cls == BACKGROUND or positives[cls] == 0:
            continue
        mask = all_pred == cls
        per_class[cls] = average_precision(all_scores[mask], all_matched[mask], int(positives[cls]))
    valid = [v for v in per_class.values() if not np.isnan(v)]

    recalls, f1s = [], []
    for cls in range(num_classes):
        support, predicted = confusion[cls].sum(), confusion[:, cls].sum()
        if support == 0 and predicted == 0:
            continue
        recall = confusion[cls, cls] / support if support else 0.0
        precision = confusion[cls, cls] / predicted if predicted else 0.0
        if support:
            recalls.append(recall)
        f1s.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))

    return EvalResult(
        accuracy=correct / max(total, 1),
        balanced_accuracy=float(np.mean(recalls)) if recalls else 0.0,
        macro_f1=float(np.mean(f1s)) if f1s else 0.0,
        mAP50=float(np.mean(valid)) if valid else 0.0,
        ade=ade_sum / max(fg_count, 1.0),
        fde=fde_sum / max(fg_count, 1.0),
        loss=loss_sum / max(total, 1),
        per_class_ap=per_class,
        samples=total,
    )


@dataclass
class RobustnessResult:
    mean: float
    std: float
    minimum: float
    p10: float
    spread: float
    per_client: Dict[int, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, float]:
        return {
            "client_acc_mean": round(self.mean, 5),
            "client_acc_std": round(self.std, 5),
            "client_acc_min": round(self.minimum, 5),
            "client_acc_p10": round(self.p10, 5),
            "client_acc_spread": round(self.spread, 5),
        }


@torch.no_grad()
def evaluate_fleet(model: nn.Module, fleet, cfg: ModelConfig, num_classes: int,
                   device: str = "cpu", batch_size: int = 128) -> RobustnessResult:
    """Per-vehicle accuracy on local test splits -> Non-IID robustness (PRD 12.1)."""
    per_client: Dict[int, float] = {}
    for vehicle in fleet:
        loader = vehicle.loader(batch_size, split="test")
        per_client[vehicle.client_id] = evaluate(model, loader, cfg, num_classes, device).accuracy
    values = np.array(list(per_client.values()), dtype=np.float64)
    if values.size == 0:
        return RobustnessResult(0, 0, 0, 0, 0, {})
    return RobustnessResult(
        mean=float(values.mean()), std=float(values.std()), minimum=float(values.min()),
        p10=float(np.percentile(values, 10)),
        spread=float(values.max() - values.min()), per_client=per_client,
    )


__all__ = ["evaluate", "evaluate_fleet", "EvalResult", "RobustnessResult",
           "box_iou_cxcywh", "average_precision"]
