"""Configuration objects for the FL-AV framework.

Defaults mirror PRD section 10 (Hyperparameters & Defaults).  Configs are plain
dataclasses so they can be (de)serialised to YAML/JSON and hashed for
reproducibility (PRD section 15 - Reproducibility).
"""
from __future__ import annotations

import json
import hashlib
from dataclasses import MISSING, dataclass, field, asdict, fields, is_dataclass
from typing import Any, Dict, Optional


@dataclass
class DataConfig:
    """Synthetic AV driving-scene generation + Non-IID partitioning (PRD 11)."""

    num_classes: int = 6
    image_size: int = 32
    traj_horizon: int = 6                 # future waypoints per agent
    samples_per_client: int = 220         # mean; quantity skew varies it
    quantity_skew: float = 0.45           # lognormal sigma on n_k (0 -> equal sizes)
    dirichlet_alpha: float = 0.4          # label skew; smaller -> more skewed
    feature_skew: bool = True             # region / weather / time-of-day skew
    train_share: float = 0.70
    val_share: float = 0.15
    test_share: float = 0.15              # global held-out test set
    global_test_size: int = 1800
    augment: bool = True


@dataclass
class ModelConfig:
    """Client model (PRD 8.1): lightweight CNN, target <= ~6M parameters."""

    width: float = 1.0
    stem_channels: int = 24
    stage_channels: tuple = (32, 64, 128)
    embed_dim: int = 192
    dropout: float = 0.05
    box_dim: int = 4                      # (cx, cy, w, h) normalised
    lambda_box: float = 1.0
    lambda_traj: float = 0.5
    max_parameters: int = 6_000_000


@dataclass
class DPConfig:
    """Differential privacy (PRD 8.7 / 10)."""

    enabled: bool = False
    clip_norm: float = 1.0
    epsilon: float = 8.0
    delta: float = 1e-5
    secure_aggregation: bool = True       # server sees only the masked sum


@dataclass
class SelectionConfig:
    """Connectivity / trajectory aware client selection (PRD 8.5)."""

    fraction: float = 0.3                 # C
    uptime_threshold: float = 0.55        # minimum V2I link uptime to be eligible
    trajectory_aware: bool = True         # spread selection over regions
    fairness_k: int = 12                  # every client picked at least once per K rounds
    min_clients: int = 2
    round_timeout_s: float = 45.0         # straggler cut-off
    seed_jitter: bool = True


@dataclass
class AggregationConfig:
    """Quality-aware FedAvg (PRD 8.4) + robustness (PRD 15/16)."""

    quality_aware: bool = True
    normalize_local_steps: bool = True    # FedNova: remove objective inconsistency
    server_momentum: float = 0.9          # beta
    byzantine_robust: bool = False        # coordinate-wise trimmed mean
    trim_fraction: float = 0.1
    q_min: float = 0.05                   # clamp for q_k (PRD 18: q_k mis-estimation)
    q_max: float = 1.0


@dataclass
class TrainConfig:
    """Local training loop (PRD 8.3 / 9 / 10)."""

    rounds: int = 60                      # R_max
    local_epochs: int = 5                 # E
    adaptive_epochs: bool = True          # FR-10: scale E to onboard compute
    min_epochs: int = 1
    max_epochs: int = 8
    lr: float = 0.01                      # eta
    cosine_decay: bool = True
    momentum: float = 0.9                 # beta (client SGD)
    weight_decay: float = 5e-4
    batch_size: int = 32                  # B
    fedprox_mu: float = 0.0               # mu; > 0 enables proximal term
    tolerance: float = 1e-3               # epsilon, ||w^{t+1} - w^t||
    target_accuracy: float = 0.90         # PRD 12.3 acceptance bar
    grad_clip: float = 5.0


@dataclass
class CommConfig:
    """V2I communication simulation (PRD 8.6 / 15)."""

    fp16: bool = True
    gzip_level: int = 6
    send_deltas: bool = True              # transmit dW rather than full weights
    broadcast_delta: bool = True          # server ships a delta when possible
    multicast: bool = True                # one RSU broadcast reaches the whole cell
    topk_ratio: float = 1.0               # < 1 enables top-k sparsified uplink
    error_feedback: bool = True           # residual stays on the vehicle
    min_topk_ratio: float = 0.02
    adaptive_sparsity: bool = True        # close the loop on the MB budget
    mb_budget_per_round: float = 9.0      # NFR: <= 9 MB/round at target config


@dataclass
class FLConfig:
    """Top-level experiment configuration."""

    name: str = "proposed"
    num_clients: int = 50                 # N
    seed: int = 1337
    device: str = "cpu"
    workers: int = 0
    out_dir: str = "runs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    dp: DPConfig = field(default_factory=DPConfig)
    comm: CommConfig = field(default_factory=CommConfig)

    # ---------------------------------------------------------------- helpers
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        """Stable hash of the config - used to tag runs for reproducibility."""
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FLConfig":
        return _build(cls, data or {})

    @classmethod
    def from_yaml(cls, path: str) -> "FLConfig":
        return cls.from_dict(_load_yaml(path))

    def merged(self, overrides: Dict[str, Any]) -> "FLConfig":
        """Return a copy with a (possibly nested) override dict applied."""
        base = self.to_dict()
        _deep_update(base, overrides or {})
        return FLConfig.from_dict(base)


def _deep_update(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _build(cls, data: Dict[str, Any]):
    """Recursively construct a dataclass from a nested dict.

    ``from __future__ import annotations`` turns ``f.type`` into a string, so we
    detect nested dataclasses through the field's ``default_factory`` instead.
    """
    kwargs: Dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        factory = getattr(f, "default_factory", None)
        nested_type = None
        if callable(factory) and factory is not MISSING:
            probe = factory()
            if is_dataclass(probe):
                nested_type = type(probe)
        if nested_type is not None and isinstance(value, dict):
            kwargs[f.name] = _build(nested_type, value)
        elif isinstance(f.default, tuple) and isinstance(value, list):
            kwargs[f.name] = tuple(value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def _load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML with PyYAML when available, else a tiny nested-dict parser.

    The fallback keeps the framework dependency-light: it understands the
    two-level ``key:`` / ``  key: value`` layout used by ``configs/*.yaml``.
    """
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except ImportError:
        return _mini_yaml(open(path, "r", encoding="utf-8").read())


def _mini_yaml(text: str) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack = [(-1, root)]
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        key, _, value = line.strip().partition(":")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        value = value.strip()
        if value == "":
            node: Dict[str, Any] = {}
            parent[key.strip()] = node
            stack.append((indent, node))
        else:
            parent[key.strip()] = _coerce(value)
    return root


def _coerce(value: str) -> Any:
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"null", "none", "~"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [_coerce(v.strip()) for v in inner.split(",")] if inner else []
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip("'\"")


def default_config(**overrides: Any) -> FLConfig:
    cfg = FLConfig()
    return cfg.merged(overrides) if overrides else cfg


__all__ = [
    "FLConfig", "DataConfig", "ModelConfig", "TrainConfig", "SelectionConfig",
    "AggregationConfig", "DPConfig", "CommConfig", "default_config",
]
