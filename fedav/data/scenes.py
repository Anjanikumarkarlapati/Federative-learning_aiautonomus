"""Procedural AV driving-scene generator.

PRD section 11 names KITTI / nuScenes / CARLA as the data sources.  KITTI and
nuScenes cannot be redistributed and CARLA needs a GPU simulator, so this module
ships a *CARLA-style synthetic generator* that reproduces the properties the
framework actually has to cope with:

* a multi-task label (class + 2-D box + short-horizon trajectory), exactly the
  task pair of PRD 8.1;
* controllable **feature skew** - region (urban / highway / rural) and weather
  (clear / rain / fog / night) change the image statistics;
* controllable **label skew** - the class mixture per client is drawn from a
  Dirichlet;
* a genuine **data-quality axis** - rain/fog/night blur and de-calibrated
  sensors inject label noise, which is what the quality-aware weighting of
  PRD 8.4 is designed to discount.

Every cue the labels depend on is drawn into the image, so the task is fully
learnable: the object rectangle carries the class, its motion streak encodes
speed and heading, and the lane-marking slant encodes the yaw rate that bends
the future trajectory.  ``adapters.py`` documents how to swap this generator for
real KITTI / nuScenes tensors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

CLASS_NAMES = ("background", "car", "truck", "pedestrian", "cyclist", "traffic_sign")
REGIONS = ("urban", "highway", "rural")
WEATHERS = ("clear", "rain", "fog", "night")

# (height, width) of the object box in normalised image units, and base speed.
CLASS_SHAPE: Dict[str, Tuple[float, float, float]] = {
    "background":   (0.10, 0.10, 0.00),
    "car":          (0.22, 0.34, 0.55),
    "truck":        (0.34, 0.46, 0.35),
    "pedestrian":   (0.30, 0.12, 0.10),
    "cyclist":      (0.26, 0.18, 0.25),
    "traffic_sign": (0.18, 0.16, 0.00),
}
# Appearance is deliberately *not* a giveaway: the palette is tight and every
# sample jitters its colour, so class identity has to come from shape, size and
# motion as well as appearance - otherwise the task saturates and the baselines
# of PRD 12.2 become indistinguishable.
CLASS_COLOR: Dict[str, Tuple[float, float, float]] = {
    "background":   (0.45, 0.45, 0.45),
    "car":          (0.66, 0.42, 0.40),
    "truck":        (0.42, 0.48, 0.64),
    "pedestrian":   (0.68, 0.64, 0.42),
    "cyclist":      (0.44, 0.62, 0.52),
    "traffic_sign": (0.68, 0.54, 0.40),
}
COLOR_JITTER = 0.16
REGION_SPEED = {"urban": 0.55, "highway": 1.25, "rural": 0.85}
REGION_CURVE = {"urban": 0.55, "highway": 0.12, "rural": 0.95}
WEATHER_FX = {
    # (brightness, contrast, blur strength, noise sigma, fog blend, rain streaks)
    "clear": (1.00, 1.00, 0.00, 0.010, 0.00, 0),
    "rain":  (0.82, 0.88, 0.35, 0.045, 0.10, 7),
    "fog":   (0.95, 0.62, 0.55, 0.030, 0.38, 0),
    "night": (0.48, 0.90, 0.25, 0.055, 0.05, 0),
}
DT = 0.5  # seconds between predicted waypoints


@dataclass
class SceneContext:
    """Environment of one vehicle (drives the feature skew of PRD 11.2)."""

    region: str = "urban"
    weather: str = "clear"
    calibrated: bool = True
    label_noise: float = 0.0        # probability of a corrupted annotation

    @property
    def blur(self) -> float:
        return WEATHER_FX[self.weather][2] + (0.0 if self.calibrated else 0.25)

    @property
    def occlusion(self) -> float:
        return {"clear": 0.05, "rain": 0.18, "fog": 0.30, "night": 0.22}[self.weather]

    @property
    def rain_score(self) -> float:
        return {"clear": 0.0, "rain": 1.0, "fog": 0.45, "night": 0.2}[self.weather]

    def quality_score(self) -> float:
        """Onboard sensor-health signal in [0, 1] (the ``quality_k`` of PRD 8.4).

        Computed from diagnostics the vehicle already has - blur, occlusion,
        rain intensity and the calibration flag - never from labels.
        """
        q = 1.0 - (0.45 * self.blur + 0.30 * self.occlusion + 0.20 * self.rain_score)
        if not self.calibrated:
            q *= 0.55
        return float(np.clip(q, 0.02, 1.0))


def _box_filter(img: np.ndarray, strength: float) -> np.ndarray:
    """Cheap separable 3-tap blur, applied ``strength``-proportionally."""
    if strength <= 0:
        return img
    k = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    out = img.copy()
    for axis in (1, 2):
        pad = np.pad(out, [(0, 0), (1, 1) if axis == 1 else (0, 0),
                           (1, 1) if axis == 2 else (0, 0)], mode="edge")
        acc = np.zeros_like(out)
        for i, weight in enumerate(k):
            sl = [slice(None)] * 3
            sl[axis] = slice(i, i + out.shape[axis])
            acc += weight * pad[tuple(sl)]
        out = acc
    return (1.0 - strength) * img + strength * out


def _draw_rect(img: np.ndarray, cx: float, cy: float, w: float, h: float,
               color: Tuple[float, float, float], alpha: float = 1.0) -> None:
    size = img.shape[1]
    x0 = int(round((cx - w / 2) * size)); x1 = int(round((cx + w / 2) * size))
    y0 = int(round((cy - h / 2) * size)); y1 = int(round((cy + h / 2) * size))
    x0, x1 = max(0, x0), min(size, max(x0 + 1, x1))
    y0, y1 = max(0, y0), min(size, max(y0 + 1, y1))
    if x1 <= x0 or y1 <= y0:
        return
    patch = np.array(color, dtype=np.float32).reshape(3, 1, 1)
    img[:, y0:y1, x0:x1] = (1 - alpha) * img[:, y0:y1, x0:x1] + alpha * patch


def render_scene(rng: np.random.Generator, cls_idx: int, ctx: SceneContext,
                 size: int, horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render one scene and return (image[3,S,S], box[4], trajectory[2*horizon])."""
    name = CLASS_NAMES[cls_idx]
    base_h, base_w, base_v = CLASS_SHAPE[name]

    # ---- geometry ---------------------------------------------------------
    h = float(np.clip(base_h * rng.uniform(0.8, 1.2), 0.05, 0.9))
    w = float(np.clip(base_w * rng.uniform(0.8, 1.2), 0.05, 0.9))
    cx = float(rng.uniform(0.18, 0.82))
    cy = float(rng.uniform(0.30, 0.78))

    # ---- kinematics: speed from class+region, yaw rate from road curvature --
    speed = base_v * REGION_SPEED[ctx.region] * rng.uniform(0.75, 1.25)
    curve = REGION_CURVE[ctx.region] * rng.uniform(-1.0, 1.0)
    heading = float(rng.uniform(-0.35, 0.35))

    # ---- background: road, sky, lane markings slanted by the yaw rate ------
    img = np.empty((3, size, size), dtype=np.float32)
    ys = np.linspace(0.0, 1.0, size, dtype=np.float32).reshape(1, size, 1)
    sky = np.array([0.55, 0.62, 0.72], dtype=np.float32).reshape(3, 1, 1)
    road = np.array([0.30, 0.30, 0.32], dtype=np.float32).reshape(3, 1, 1)
    img[:] = sky * (1 - ys) + road * ys
    for lane_x in (0.32, 0.68):
        for row in range(int(0.45 * size), size):
            t = (row / size - 0.45) / 0.55
            x = int(np.clip((lane_x + curve * 0.22 * t * t) * size, 0, size - 1))
            img[:, row, x] = 0.92

    # ---- motion streak encodes speed and heading --------------------------
    color = tuple(float(np.clip(c + rng.normal(0.0, COLOR_JITTER), 0.05, 0.95))
                  for c in CLASS_COLOR[name])
    steps = int(np.clip(speed * 6.0, 0, 6))
    for s in range(steps, 0, -1):
        frac = s / max(steps, 1)
        _draw_rect(img, cx - 0.06 * s * np.cos(heading), cy - 0.03 * s * np.sin(heading),
                   w * 0.9, h * 0.9, color, alpha=0.25 * (1 - frac))
    if name != "background":
        _draw_rect(img, cx, cy, w, h, color, alpha=1.0)

    # ---- weather / sensor degradation -------------------------------------
    bright, contrast, blur, noise, fog, streaks = WEATHER_FX[ctx.weather]
    if not ctx.calibrated:
        bright *= rng.uniform(0.7, 1.3)
    img = _box_filter(img, min(0.9, blur + (0.2 if not ctx.calibrated else 0.0)))
    img = (img - 0.5) * contrast + 0.5
    img *= bright
    if fog > 0:
        img = (1 - fog) * img + fog * 0.72
    for _ in range(streaks):
        x = int(rng.integers(0, size)); y0 = int(rng.integers(0, size - 4))
        img[:, y0:y0 + 4, x] = np.minimum(1.0, img[:, y0:y0 + 4, x] + 0.35)
    if ctx.occlusion > 0 and rng.random() < ctx.occlusion:
        ox, oy = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)
        _draw_rect(img, ox, oy, 0.22, 0.22, (0.15, 0.15, 0.15), alpha=0.85)
    img += rng.normal(0.0, noise, img.shape).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)

    # ---- trajectory: constant-speed, constant-yaw-rate rollout ------------
    traj = np.zeros((horizon, 2), dtype=np.float32)
    x, y, th = 0.0, 0.0, heading
    for t in range(horizon):
        th += curve * 0.35 * DT
        x += speed * DT * float(np.cos(th))
        y += speed * DT * float(np.sin(th))
        traj[t] = (x, y)
    box = np.array([cx, cy, w, h], dtype=np.float32)
    return img, box, traj.reshape(-1)


def generate_samples(n: int, class_probs: np.ndarray, ctx: SceneContext,
                     size: int, horizon: int, rng: np.random.Generator
                     ) -> Dict[str, np.ndarray]:
    """Generate ``n`` scenes whose class mixture follows ``class_probs``."""
    probs = np.asarray(class_probs, dtype=np.float64)
    probs = probs / probs.sum()
    labels = rng.choice(len(CLASS_NAMES), size=n, p=probs)
    images = np.empty((n, 3, size, size), dtype=np.float32)
    boxes = np.empty((n, 4), dtype=np.float32)
    trajs = np.empty((n, 2 * horizon), dtype=np.float32)
    for i, cls_idx in enumerate(labels):
        images[i], boxes[i], trajs[i] = render_scene(rng, int(cls_idx), ctx, size, horizon)

    # De-calibrated / poor-visibility vehicles produce noisy annotations.
    if ctx.label_noise > 0:
        flip = rng.random(n) < ctx.label_noise
        labels = labels.copy()
        labels[flip] = rng.integers(0, len(CLASS_NAMES), size=int(flip.sum()))
        boxes[flip] += rng.normal(0, 0.12, (int(flip.sum()), 4)).astype(np.float32)
        trajs[flip] += rng.normal(0, 0.6, (int(flip.sum()), trajs.shape[1])).astype(np.float32)
        boxes = np.clip(boxes, 0.01, 0.99)
    return {
        "images": images,
        "labels": labels.astype(np.int64),
        "boxes": boxes.astype(np.float32),
        "trajectories": trajs.astype(np.float32),
    }


__all__ = [
    "CLASS_NAMES", "REGIONS", "WEATHERS", "SceneContext",
    "render_scene", "generate_samples", "DT",
]
