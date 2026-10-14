"""Invariants of the optional mechanisms (all off by default).

The six invariants: capped-simplex weights, KST-quality monotonicity, FedNova
removing step-count bias, error-feedback bookkeeping, analytic-Gaussian
dominance, and exact Z_2^32 secure aggregation (see also tests/test_secagg.py).
"""
import math
import tempfile

import numpy as np
import pytest
import torch

from fedav.client.compression import TopKCompressor
from fedav.config import AggregationConfig, default_config
from fedav.data import quality as Q
from fedav.privacy.dp import analytic_gaussian_sigma
from fedav.server import secagg
from fedav.server.aggregator import ContributedUpdate, FedAggregator, capped_simplex


# 1. simplex ------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(5))
def test_capped_simplex_is_on_the_simplex_and_under_the_cap(seed):
    w = np.random.default_rng(seed).lognormal(0, 1.5, 15)
    cap = 2.5 / 15
    a = capped_simplex(w, cap)
    assert abs(a.sum() - 1.0) < 1e-12
    assert (a >= 0).all() and a.max() <= cap + 1e-12
    # weights below the cap keep their relative order
    order = np.argsort(w)
    assert np.all(np.diff(a[order]) >= -1e-12)


def test_capped_simplex_infeasible_cap_returns_uniform():
    np.testing.assert_allclose(capped_simplex(np.array([5.0, 1.0, 1.0]), 0.2), [1 / 3] * 3)


# 2. KST monotonicity ---------------------------------------------------------
BASE = dict(mean_speed=0.3, yaw_rate_var=0.1, snr_db=8.0, occlusion_rate=0.2, calibrated=True,
            histogram=[0.5, 0.3, 0.2], prior=[0.4, 0.4, 0.2])


@pytest.mark.parametrize("key,lo,hi", [("mean_speed", 0.05, 0.8), ("yaw_rate_var", 0.0, 0.5),
                                       ("snr_db", 0.0, 20.0)])
def test_kst_quality_increases_with_motion_and_snr(key, lo, hi):
    q_lo = Q.compute_vehicle_quality(**{**BASE, key: lo})
    q_hi = Q.compute_vehicle_quality(**{**BASE, key: hi})
    assert q_hi > q_lo


def test_kst_quality_falls_with_occlusion_and_miscalibration():
    assert Q.compute_vehicle_quality(**{**BASE, "occlusion_rate": 0.5}) < Q.compute_vehicle_quality(**BASE)
    assert Q.compute_vehicle_quality(**{**BASE, "calibrated": False}) < Q.compute_vehicle_quality(**BASE)


def test_kst_rarity_bonus_rewards_divergence():
    common = Q.rare_scene_bonus([0.4, 0.4, 0.2], [0.4, 0.4, 0.2])
    rare = Q.rare_scene_bonus([0.05, 0.05, 0.9], [0.4, 0.4, 0.2])
    assert common == pytest.approx(1.0) and rare > common


# 3. FedNova ------------------------------------------------------------------
def test_fednova_removes_step_count_bias():
    g1, g2 = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
    ups = [ContributedUpdate(0, {"w": 1 * g1}, n_k=100, q_k=1.0, steps=1),
           ContributedUpdate(1, {"w": 8 * g2}, n_k=100, q_k=1.0, steps=8)]
    plain = FedAggregator(AggregationConfig(normalize_local_steps=False, server_momentum=0))
    nova = FedAggregator(AggregationConfig(normalize_local_steps=True, server_momentum=0))
    d_plain, _ = plain.combine(ups)
    d_nova, _ = nova.combine(ups)
    # equal weights: the right direction is (g1 + g2) / 2
    cos = float(torch.nn.functional.cosine_similarity(d_nova["w"], g1 + g2, dim=0))
    assert cos == pytest.approx(1.0)
    assert float(d_plain["w"][1]) / float(d_plain["w"][0]) == pytest.approx(8.0)


# 4. error feedback -----------------------------------------------------------
@pytest.mark.parametrize("normalize,fp16", [(False, False), (True, False), (True, True)])
def test_error_feedback_loses_nothing_and_stays_bounded(normalize, fp16):
    comp = TopKCompressor(ratio=0.1, error_feedback=True)
    g = torch.Generator().manual_seed(0)
    tau, total, sent = 4.0, torch.zeros(200), torch.zeros(200)
    norms = []
    for _ in range(60):
        dw = torch.randn(200, generator=g) * 0.01
        out = comp.compress(0, {"w": dw}, steps=tau, normalize=normalize, fp16=fp16)["w"]
        scale = tau if normalize else 1.0
        total += dw / scale
        sent += out / scale
        norms.append(float(comp.residuals[0]["w"].norm()))
    # telescoping: everything not yet sent is exactly the residual
    assert torch.allclose(total, sent + comp.residuals[0]["w"], atol=1e-5)
    # contraction: the residual does not keep growing
    assert max(norms[-10:]) < 3 * max(norms[10:20])


def test_nack_restore_undoes_step_scaling():
    comp = TopKCompressor(ratio=0.5, error_feedback=True)
    dw = {"w": torch.tensor([4.0, -3.0, 0.5, 0.1])}
    out = comp.compress(0, dw, steps=2.0, normalize=True)
    comp.restore(0, out, steps=2.0)
    assert torch.allclose(comp.residuals[0]["w"], dw["w"] / 2.0)


# 5. analytic Gaussian --------------------------------------------------------
@pytest.mark.parametrize("eps", [0.1, 0.5, 0.9])
def test_analytic_gaussian_needs_no_more_noise_than_classical(eps):
    delta = 1e-5
    classical = math.sqrt(2 * math.log(1.25 / delta)) / eps
    assert analytic_gaussian_sigma(1.0, eps, delta) <= classical


# 6. Z_2^32 secure aggregation ------------------------------------------------
def test_ring_masks_cancel_exactly():
    keys = {c: secagg.keypair(3, c) for c in range(5)}
    pubs = {c: k.public for c, k in keys.items()}
    tiers = {c: 0 for c in range(5)}
    zero = np.zeros(16)
    total = np.zeros(18, dtype=np.uint32)
    for c in range(5):
        up = secagg.vehicle_encode(c, 1, keys[c], pubs, tiers, [16], 0, zero, 0.0, 0.0,
                                   secagg.FixedPoint(1.0), secagg.FixedPoint(64.0))
        total += secagg.parse_upload(up.payload)[3]
    assert not total.any()                       # exact zero, not "small"


# options in a full run -------------------------------------------------------
def test_all_options_run_end_to_end():
    from fedav.sim.runner import run_experiment
    cfg = default_config().merged({
        "name": "opts", "num_clients": 8, "out_dir": tempfile.mkdtemp(),
        "data": {"samples_per_client": 40, "global_test_size": 96, "num_classes": 9},
        "train": {"rounds": 3, "local_epochs": 1, "scaffold": True, "grad_clip": 2.5},
        "selection": {"fraction": 0.5, "mode": "kst_ucb"},
        "model": {"focal_gamma": 1.5},
        "aggregation": {"quality_score": "kst", "alpha_cap_ratio": 2.5,
                        "norm_downweight": 2.5, "nesterov": True},
        "comm": {"topk_ratio": 0.25, "ef_normalize_steps": True, "ef_quantization": True},
        "dp": {"secure_aggregation": False},
    })
    hist = run_experiment(cfg, echo=False).history
    assert all(np.isfinite(r["loss"]) for r in hist if "loss" in r)
    assert all("rare_class_accuracy" in r for r in hist if "accuracy" in r)
