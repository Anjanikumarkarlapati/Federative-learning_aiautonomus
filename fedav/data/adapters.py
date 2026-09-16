"""Hooks for real AV datasets (PRD 11.1).

The framework only ever sees a :class:`~fedav.data.federated.SceneDataset`,
i.e. four aligned arrays::

    images        float32 [N, 3, S, S]   normalised camera tensor
    labels        int64   [N]            class id
    boxes         float32 [N, 4]         (cx, cy, w, h), normalised to [0, 1]
    trajectories  float32 [N, 2 * T]     future waypoints in ego metres

To train on KITTI / nuScenes / CARLA instead of the synthetic generator, write a
loader that produces those four arrays per vehicle and hand them to
``SceneDataset``; nothing else in the pipeline changes.  Sketches below show the
mapping for each source.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .federated import SceneDataset


def from_arrays(images: np.ndarray, labels: np.ndarray, boxes: np.ndarray,
                trajectories: np.ndarray, augment: bool = False, seed: int = 0
                ) -> SceneDataset:
    """Wrap externally produced arrays in the framework's dataset contract."""
    n = len(labels)
    if not (len(images) == len(boxes) == len(trajectories) == n):
        raise ValueError("images/labels/boxes/trajectories must be the same length")
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must be [N, 3, H, W]")
    if boxes.shape[1] != 4:
        raise ValueError("boxes must be [N, 4] as (cx, cy, w, h)")
    arrays: Dict[str, np.ndarray] = {
        "images": images.astype(np.float32),
        "labels": labels.astype(np.int64),
        "boxes": boxes.astype(np.float32),
        "trajectories": trajectories.astype(np.float32),
    }
    return SceneDataset(arrays, augment=augment, seed=seed)


KITTI_NOTES = """
KITTI (object detection benchmark)
  images       : image_2/*.png  -> resize to S x S, scale to [0, 1]
  labels       : label_2/*.txt  -> type field mapped onto CLASS_NAMES
  boxes        : the 2-D bbox (x1, y1, x2, y2) -> (cx, cy, w, h) / (W, H)
  trajectories : KITTI has no future track, so either use the tracking split
                 (label_02 with track ids) or set lambda_traj = 0.
  federation   : partition by drive sequence -> natural feature skew.
"""

NUSCENES_NOTES = """
nuScenes (multi-sensor, multi-city)
  images       : CAM_FRONT samples
  labels       : sample_annotation.category_name -> CLASS_NAMES
  boxes        : project the 3-D box into the image plane, then normalise
  trajectories : future ego/agent poses at 2 Hz for T steps, in ego frame
  federation   : partition by (location, weather/time-of-day) -> the exact
                 region x weather skew the synthetic generator emulates.
"""

CARLA_NOTES = """
CARLA (closed-loop simulator)
  images       : RGB camera sensor stream
  labels/boxes : ground-truth actor bounding boxes from the world snapshot
  trajectories : replayed future waypoints of the tracked actor
  federation   : one client per (town, weather preset) so alpha-style skew is
                 set directly by the scenario configuration.
"""

__all__ = ["from_arrays", "KITTI_NOTES", "NUSCENES_NOTES", "CARLA_NOTES"]
