"""Fleet construction: Non-IID partitioning + per-vehicle datasets (PRD 11.2).

Three skew axes are combined, as the PRD requires:

1. **Label skew** - ``p_k ~ Dirichlet(alpha * 1)`` over the class simplex.
   Smaller ``alpha`` concentrates mass on fewer classes.
2. **Feature skew** - each vehicle is assigned a (region, weather) context that
   changes image statistics, plus a sensor-calibration flag.
3. **Quantity skew** - ``n_k`` is log-normal around the configured mean.

Each vehicle keeps a 70 / 15 / 15 train / val / test split (PRD 11.3) and the
server holds a separate IID global test set for fair reporting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import DataConfig, FLConfig
from ..utils.seeding import rng as make_rng
from .scenes import CLASS_NAMES, REGIONS, WEATHERS, SceneContext, generate_samples


class SceneDataset(Dataset):
    """Tensor dataset of (image, label, box, trajectory) with optional augmentation."""

    def __init__(self, arrays: Dict[str, np.ndarray], augment: bool = False, seed: int = 0):
        self.images = torch.from_numpy(np.ascontiguousarray(arrays["images"]))
        self.labels = torch.from_numpy(np.ascontiguousarray(arrays["labels"]))
        self.boxes = torch.from_numpy(np.ascontiguousarray(arrays["boxes"]))
        self.trajectories = torch.from_numpy(np.ascontiguousarray(arrays["trajectories"]))
        self.augment = augment
        self._seed = int(seed)
        self._gen = torch.Generator().manual_seed(self._seed)

    def reset_augmentation(self) -> None:
        """Rewind the augmentation stream to its construction seed."""
        self._gen.manual_seed(self._seed)

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, idx: int):
        image = self.images[idx]
        box = self.boxes[idx]
        traj = self.trajectories[idx]
        if self.augment:
            if torch.rand(1, generator=self._gen).item() < 0.5:
                # horizontal flip: mirror the image, the box centre and the
                # lateral component of every future waypoint.
                image = torch.flip(image, dims=[2])
                box = box.clone(); box[0] = 1.0 - box[0]
                traj = traj.clone(); traj[1::2] = -traj[1::2]
            jitter = 1.0 + 0.10 * (torch.rand(1, generator=self._gen).item() - 0.5)
            image = (image * jitter).clamp(0.0, 1.0)
        return image, self.labels[idx], box, traj

    def label_histogram(self, num_classes: int = len(CLASS_NAMES)) -> np.ndarray:
        hist = np.bincount(self.labels.numpy(), minlength=num_classes).astype(np.float64)
        total = hist.sum()
        return hist / total if total > 0 else hist


@dataclass
class VehicleProfile:
    """Everything the server may know about a vehicle *without* seeing its data."""

    client_id: int
    region: str
    weather: str
    calibrated: bool
    n_k: int                       # local training-set size
    link_uptime: float             # fraction of the round the V2I link is up
    compute_capability: float      # relative onboard TOPS (1.0 = reference)
    quality_signal: float          # sensor-health score in [0, 1]
    byzantine: bool = False
    label_histogram: np.ndarray = field(default_factory=lambda: np.zeros(len(CLASS_NAMES)))


@dataclass
class Vehicle:
    profile: VehicleProfile
    train: SceneDataset
    val: SceneDataset
    test: SceneDataset

    @property
    def client_id(self) -> int:
        return self.profile.client_id

    @property
    def n_k(self) -> int:
        return len(self.train)

    def loader(self, batch_size: int, seed: int = 0, split: str = "train") -> DataLoader:
        dataset = {"train": self.train, "val": self.val, "test": self.test}[split]
        shuffle = split == "train"
        generator = torch.Generator().manual_seed(int(seed)) if shuffle else None
        return DataLoader(
            dataset,
            batch_size=min(batch_size, max(1, len(dataset))),
            shuffle=shuffle,
            generator=generator,
            drop_last=False,
        )


@dataclass
class Fleet:
    vehicles: List[Vehicle]
    global_test: SceneDataset
    reference_histogram: np.ndarray

    def __len__(self) -> int:
        return len(self.vehicles)

    def __iter__(self):
        return iter(self.vehicles)

    def __getitem__(self, idx: int) -> Vehicle:
        return self.vehicles[idx]

    @property
    def profiles(self) -> List[VehicleProfile]:
        return [v.profile for v in self.vehicles]

    def reset_augmentation(self) -> None:
        """Rewind every vehicle's augmentation stream.

        The fleet is shared across benchmark arms, and each training set draws
        its flips/jitter from a stateful generator - without a rewind, an arm's
        result depends on which arms ran before it on the same fleet.
        """
        for vehicle in self.vehicles:
            vehicle.train.reset_augmentation()

    @property
    def total_samples(self) -> int:
        """n = sum_k n_k (PRD 8.2)."""
        return int(sum(v.n_k for v in self.vehicles))

    def test_loader(self, batch_size: int = 128) -> DataLoader:
        return DataLoader(self.global_test, batch_size=batch_size, shuffle=False)

    def pooled_train(self, augment: bool = False, seed: int = 0) -> SceneDataset:
        """Union of every local training shard - only the centralized baseline
        (PRD 12.2a) is allowed to use this; it violates the privacy goal and
        exists purely as an upper-bound reference."""
        arrays = {
            key: np.concatenate([getattr(v.train, attr).numpy() for v in self.vehicles])
            for key, attr in (("images", "images"), ("labels", "labels"),
                              ("boxes", "boxes"), ("trajectories", "trajectories"))
        }
        return SceneDataset(arrays, augment=augment, seed=seed)


def dirichlet_class_probs(num_clients: int, num_classes: int, alpha: float,
                          generator: np.random.Generator) -> np.ndarray:
    """p_k ~ Dir(alpha), one row per client (PRD 11.2, label skew)."""
    alpha = max(float(alpha), 1e-3)
    return generator.dirichlet(np.full(num_classes, alpha), size=num_clients)


def quantity_skew_sizes(num_clients: int, mean: int, sigma: float,
                        generator: np.random.Generator, minimum: int = 40) -> np.ndarray:
    """Log-normal n_k with mean preserved (PRD 11.2, quantity skew)."""
    if sigma <= 0:
        return np.full(num_clients, int(mean), dtype=np.int64)
    raw = generator.lognormal(mean=0.0, sigma=float(sigma), size=num_clients)
    raw = raw / raw.mean() * float(mean)
    return np.maximum(minimum, raw.round()).astype(np.int64)


def _split_arrays(arrays: Dict[str, np.ndarray], shares: Tuple[float, float, float],
                  generator: np.random.Generator) -> List[Dict[str, np.ndarray]]:
    n = len(arrays["labels"])
    order = generator.permutation(n)
    n_train = max(1, int(round(shares[0] * n)))
    n_val = max(1, int(round(shares[1] * n))) if n - n_train > 1 else 0
    bounds = [order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:]]
    out = []
    for idx in bounds:
        if len(idx) == 0:                       # keep every split non-empty
            idx = bounds[0][:1]
        out.append({k: v[idx] for k, v in arrays.items()})
    return out


def build_fleet(cfg: FLConfig, byzantine_ids: Optional[List[int]] = None) -> Fleet:
    """Materialise ``cfg.num_clients`` vehicles with Non-IID local data."""
    data: DataConfig = cfg.data
    generator = make_rng(cfg.seed, 0xF1EE7)
    num_classes = data.num_classes
    probs = dirichlet_class_probs(cfg.num_clients, num_classes, data.dirichlet_alpha, generator)
    sizes = quantity_skew_sizes(cfg.num_clients, data.samples_per_client,
                                data.quantity_skew, generator)
    byz = set(byzantine_ids or [])

    vehicles: List[Vehicle] = []
    for k in range(cfg.num_clients):
        client_rng = make_rng(cfg.seed, k, 17)
        if data.feature_skew:
            region = REGIONS[k % len(REGIONS)]
            weather = WEATHERS[int(client_rng.integers(0, len(WEATHERS)))]
            calibrated = bool(client_rng.random() > 0.12)
        else:
            region, weather, calibrated = "urban", "clear", True
        label_noise = 0.0 if calibrated else float(client_rng.uniform(0.25, 0.45))
        label_noise += {"clear": 0.0, "rain": 0.05, "fog": 0.10, "night": 0.08}[weather]
        ctx = SceneContext(region=region, weather=weather, calibrated=calibrated,
                           label_noise=label_noise)

        n_total = int(sizes[k])
        arrays = generate_samples(n_total, probs[k], ctx, data.image_size,
                                  data.traj_horizon, client_rng)
        shares = (data.train_share, data.val_share, data.test_share)
        tr, va, te = _split_arrays(arrays, shares, client_rng)
        train_ds = SceneDataset(tr, augment=data.augment, seed=cfg.seed + k)
        profile = VehicleProfile(
            client_id=k,
            region=region,
            weather=weather,
            calibrated=calibrated,
            n_k=len(train_ds),
            link_uptime=float(np.clip(client_rng.beta(5, 2), 0.05, 1.0)),
            compute_capability=float(np.clip(client_rng.lognormal(0.0, 0.45), 0.2, 3.0)),
            quality_signal=ctx.quality_score(),
            byzantine=k in byz,
            label_histogram=train_ds.label_histogram(num_classes),
        )
        vehicles.append(Vehicle(profile=profile, train=train_ds,
                                val=SceneDataset(va), test=SceneDataset(te)))

    # Global held-out test set: IID over classes and environments so that every
    # method is scored on the same, unbiased distribution (PRD 11.3 / 12.1).
    test_rng = make_rng(cfg.seed, 0xBEEF)
    chunks = []
    per_ctx = max(1, data.global_test_size // (len(REGIONS) * len(WEATHERS)))
    for region in REGIONS:
        for weather in WEATHERS:
            ctx = SceneContext(region=region, weather=weather, calibrated=True, label_noise=0.0)
            chunks.append(generate_samples(per_ctx, np.ones(num_classes), ctx,
                                           data.image_size, data.traj_horizon, test_rng))
    merged = {key: np.concatenate([c[key] for c in chunks]) for key in chunks[0]}
    global_test = SceneDataset(merged)

    # Fleet reference label distribution, broadcast to clients so they can score
    # their own similarity locally.  See the caveat in data/quality.py: in a
    # deployment this reference is collected under secure aggregation rather
    # than assembled from per-vehicle histograms in the clear.
    weights = np.array([v.n_k for v in vehicles], dtype=np.float64)
    stacked = np.stack([v.profile.label_histogram for v in vehicles])
    reference = (stacked * weights[:, None]).sum(axis=0) / weights.sum()
    return Fleet(vehicles=vehicles, global_test=global_test, reference_histogram=reference)


__all__ = ["SceneDataset", "Vehicle", "VehicleProfile", "Fleet", "build_fleet",
           "dirichlet_class_probs", "quantity_skew_sizes"]
