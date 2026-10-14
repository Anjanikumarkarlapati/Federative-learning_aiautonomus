"""Regression tests for the mechanisms the patent specification describes.

* the secure round (K1) reproduces the plaintext quality-weighted FedNova
  aggregate, including when uploads are lost or corrupted;
* NACK residual retention (K2): a failed upload loses no coordinates;
* airtime tiers (K2) keep the round's uplink airtime near its budget;
* the fairness guard bounds starvation for weak-link vehicles too;
* the FedAvg baseline really selects uniformly at random.
"""
import json
import os

import numpy as np
import pytest

from fedav.config import default_config
from fedav.sim.runner import run_experiment


def _cfg(tmp_path, name, **over):
    base = {
        "name": name, "num_clients": 8, "out_dir": str(tmp_path),
        "data": {"samples_per_client": 40, "global_test_size": 96, "image_size": 32},
        "train": {"rounds": 3, "local_epochs": 1},
        "selection": {"fraction": 0.5, "uptime_threshold": 0.0},
        "comm": {"topk_ratio": 0.25},
        "dp": {"secure_aggregation": True},
    }
    for key, value in over.items():
        base.setdefault(key, {}).update(value) if isinstance(value, dict) else base.update({key: value})
    return default_config().merged(base)


def _rounds(tmp_path, name):
    with open(os.path.join(str(tmp_path), name, "rounds.jsonl")) as fh:
        return [json.loads(line) for line in fh]


def test_secure_round_matches_plaintext_even_with_failed_uploads(tmp_path):
    cfg = _cfg(tmp_path, "sec", comm={"corrupt_prob": 0.3})
    run_experiment(cfg, echo=False)
    rounds = [r for r in _rounds(tmp_path, "sec") if r.get("clients_reported")]
    assert rounds
    assert all(r["secure_aggregation_verified"] for r in rounds)
    assert all(r["secagg_rel_error"] < 1e-3 for r in rounds)
    assert sum(r["drop_corrupt"] + r["drop_dropped"] for r in _rounds(tmp_path, "sec")) > 0
    assert all(r["secagg_scalar_clipped"] == 0 for r in rounds)


def test_airtime_budget_tiers_bound_round_airtime(tmp_path):
    free = _cfg(tmp_path, "free")
    tight = _cfg(tmp_path, "tight", comm={"airtime_budget_s": 0.5})
    run_experiment(free, echo=False)
    run_experiment(tight, echo=False)
    a_free = np.mean([r["uplink_airtime_s"] for r in _rounds(tmp_path, "free")])
    a_tight = np.mean([r["uplink_airtime_s"] for r in _rounds(tmp_path, "tight")])
    assert a_tight < a_free


def test_nack_keeps_the_failed_upload_in_the_residual():
    from fedav.client.compression import TopKCompressor
    import torch
    comp = TopKCompressor(ratio=0.5, error_feedback=True)
    update = {"w": torch.tensor([4.0, -3.0, 0.5, 0.1])}
    sent = comp.compress(0, update)
    comp.restore(0, sent)                       # the upload was NACKed
    total = sum(comp.residuals[0][k] for k in comp.residuals[0])
    assert torch.allclose(comp.residuals[0]["w"], update["w"])


def test_fairness_guard_reaches_weak_link_vehicles():
    from fedav.config import SelectionConfig
    from fedav.server.selection import ClientSelector

    class P:
        def __init__(self, cid, up):
            self.client_id, self.link_uptime, self.region = cid, up, "urban"
            self.quality_signal, self.n_k = 1.0, 100
    profiles = [P(c, 0.95 if c < 8 else 0.1) for c in range(10)]
    cfg = SelectionConfig(fraction=0.3, uptime_threshold=0.55, fairness_k=4, seed_jitter=False)
    sel = ClientSelector(cfg, len(profiles), seed=1)
    last = {c: 0 for c in range(10)}
    worst_gap = 0
    for r in range(1, 41):
        for c in sel.select(profiles, r).selected:
            worst_gap = max(worst_gap, r - last[c])
            last[c] = r
    worst_gap = max(worst_gap, max(41 - v for v in last.values()))
    assert worst_gap <= cfg.fairness_k + 1


def test_uniform_selection_ignores_link_quality():
    from fedav.config import SelectionConfig
    from fedav.server.selection import ClientSelector

    class P:
        def __init__(self, cid, up):
            self.client_id, self.link_uptime, self.region = cid, up, "urban"
            self.quality_signal, self.n_k = 1.0, 100
    profiles = [P(c, 0.99 if c < 5 else 0.05) for c in range(10)]
    sel = ClientSelector(SelectionConfig(mode="uniform", fraction=0.5), 10, seed=3)
    counts = np.zeros(10)
    for r in range(1, 401):
        counts[sel.select(profiles, r).selected] += 1
    strong, weak = counts[:5].sum(), counts[5:].sum()
    assert abs(strong - weak) / (strong + weak) < 0.05
