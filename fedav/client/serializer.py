"""UpdateSerializer - wire format for model updates (PRD 8.6 / 13).

Payload per round, vehicle -> RSU::

    client_id, n_k, q_k, round_id, model_delta (dW), checksum

The tensor payload is flattened in a fixed key order, optionally cast to
float16 and gzip-compressed, which is the compression scheme the PRD names and
the lever that takes communication cost below the 9 MB/round budget of
PRD 12.3.  A CRC32 checksum over the compressed bytes detects corruption on the
lossy V2I link; ``unpack`` raises on mismatch so the aggregator can drop the
update instead of poisoning the global model.
"""
from __future__ import annotations

import gzip
import json
import struct
import zlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np
import torch

from ..config import CommConfig
from ..utils.tensors import StateDict

MAGIC = b"FLAV"
_HEADER = struct.Struct("<4sI")


class ChecksumError(ValueError):
    """Raised when a received payload fails its CRC32 check."""


@dataclass
class UpdateSerializer:
    cfg: CommConfig = field(default_factory=CommConfig)

    # ------------------------------------------------------------------ pack
    def pack(self, update: StateDict, meta: Dict[str, Any]) -> bytes:
        keys = list(update.keys())
        dtype = np.float16 if self.cfg.fp16 else np.float32
        flat = np.concatenate(
            [update[k].detach().to(torch.float32).reshape(-1).numpy() for k in keys]
        ).astype(dtype) if keys else np.zeros(0, dtype=dtype)
        payload = gzip.compress(flat.tobytes(), compresslevel=self.cfg.gzip_level)
        header = {
            "keys": [[k, list(update[k].shape)] for k in keys],
            "dtype": np.dtype(dtype).str,
            "numel": int(flat.size),
            "meta": meta,
            "checksum": zlib.crc32(payload) & 0xFFFFFFFF,
        }
        blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
        return _HEADER.pack(MAGIC, len(blob)) + blob + payload

    # ---------------------------------------------------------------- unpack
    def unpack(self, raw: bytes) -> Tuple[StateDict, Dict[str, Any]]:
        magic, header_len = _HEADER.unpack_from(raw, 0)
        if magic != MAGIC:
            raise ValueError("not a FLAV update payload")
        start = _HEADER.size
        header = json.loads(raw[start:start + header_len].decode("utf-8"))
        payload = raw[start + header_len:]
        if (zlib.crc32(payload) & 0xFFFFFFFF) != header["checksum"]:
            raise ChecksumError(
                f"checksum mismatch for client {header['meta'].get('client_id')} "
                f"round {header['meta'].get('round_id')}"
            )
        flat = np.frombuffer(gzip.decompress(payload), dtype=np.dtype(header["dtype"]))
        if flat.size != header["numel"]:
            raise ChecksumError("payload length mismatch after decompression")
        flat = flat.astype(np.float32)
        state: StateDict = OrderedDict()
        offset = 0
        for key, shape in header["keys"]:
            count = int(np.prod(shape)) if shape else 1
            state[key] = torch.from_numpy(
                np.array(flat[offset:offset + count], dtype=np.float32)
            ).view(*shape)
            offset += count
        return state, header["meta"]

    # ----------------------------------------------------------------- utils
    @staticmethod
    def size_mb(raw: bytes) -> float:
        return len(raw) / (1024.0 * 1024.0)

    def uncompressed_mb(self, update: StateDict) -> float:
        numel = sum(v.numel() for v in update.values())
        return numel * 4.0 / (1024.0 * 1024.0)

    def compression_ratio(self, update: StateDict, raw: bytes) -> float:
        base = self.uncompressed_mb(update)
        return float(base / self.size_mb(raw)) if len(raw) else 1.0


__all__ = ["UpdateSerializer", "ChecksumError"]
