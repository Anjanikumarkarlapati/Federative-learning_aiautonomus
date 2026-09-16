"""Unit tests for the aggregation / privacy / metric mathematics (PRD 17)."""
import math
from collections import OrderedDict

import numpy as np
import pytest
import torch

from fedav.config import AggregationConfig
from fedav.data.quality import (aggregation_weights, compute_quality_weight,
                                hellinger_distance, jensen_shannon_distance)
from fedav.metrics.evaluation import average_precision, box_iou_cxcywh
from fedav.privacy.dp import (analytic_gaussian_sigma, clip_state, compose_epsilon,
                              calibrate_noise_multiplier, PrivacyAccountant)
from fedav.server.aggregator import ContributedUpdate, FedAggregator, has_converged
from fedav.utils.tensors import state_norm, trimmed_mean, weighted_average


# ----------------------------------------------------------------- weighting
def test_aggregation_weights_sum_to_one():
    for n_k, q_k in [([10, 20, 30], None), ([5, 5, 90], [0.1, 0.9, 0.5]),
                     ([1], [1.0]), ([7, 7], [0.0, 0.0])]:
        weights = aggregation_weights(n_k, q_k)
        assert math.isclose(weights.sum(), 1.0, rel_tol=1e-12)
        assert (weights >= 0).all()


def test_quality_weight_downweights_low_quality_clients():
    reference = [0.2] * 5
    clean = compute_quality_weight([0.2] * 5, reference, quality_signal=0.95)
    degraded = compute_quality_weight([0.2] * 5, reference, quality_signal=0.25)
    off_distribution = compute_quality_weight([1.0, 0, 0, 0, 0], reference, quality_signal=0.95)
    assert clean > degraded
    assert clean > off_distribution
    assert 0.0 <= off_distribution <= 1.0


def test_distribution_distances_are_metricish():
    p, q = [0.5, 0.5], [0.5, 0.5]
    assert hellinger_distance(p, q) == pytest.approx(0.0, abs=1e-12)
    assert hellinger_distance([1, 0], [0, 1]) == pytest.approx(1.0)
    assert 0.0 <= jensen_shannon_distance([0.7, 0.3], [0.2, 0.8]) <= 1.0


def test_quality_aware_weights_beat_size_weights_for_noisy_client():
    """A large but sensor-degraded shard must lose influence."""
    n_k, q_k = [1000, 100], [0.1, 0.9]
    plain = aggregation_weights(n_k)
    aware = aggregation_weights(n_k, q_k)
    assert plain[0] > aware[0]


# ---------------------------------------------------------------- aggregation
def _updates(values, n_k=None, q_k=None):
    n_k = n_k or [10] * len(values)
    q_k = q_k or [1.0] * len(values)
    return [ContributedUpdate(i, OrderedDict(w=torch.full((4,), float(v))), n, q)
            for i, (v, n, q) in enumerate(zip(values, n_k, q_k))]


def test_fedavg_matches_closed_form():
    updates = _updates([1.0, 3.0], n_k=[10, 30])
    aggregator = FedAggregator(AggregationConfig(quality_aware=False, server_momentum=0.0))
    state, report = aggregator.aggregate(OrderedDict(w=torch.zeros(4)), updates)
    # (10/40)*1 + (30/40)*3 = 2.5
    assert torch.allclose(state["w"], torch.full((4,), 2.5))
    assert math.isclose(sum(report.weights.values()), 1.0, rel_tol=1e-9)


def test_quality_aware_aggregation_matches_closed_form():
    updates = _updates([1.0, 3.0], n_k=[10, 30], q_k=[0.9, 0.1])
    aggregator = FedAggregator(AggregationConfig(quality_aware=True, server_momentum=0.0))
    state, _ = aggregator.aggregate(OrderedDict(w=torch.zeros(4)), updates)
    expected = (0.9 * 10 * 1.0 + 0.1 * 30 * 3.0) / (0.9 * 10 + 0.1 * 30)
    assert torch.allclose(state["w"], torch.full((4,), expected), atol=1e-6)


def test_weighted_average_is_convex_combination():
    states = [OrderedDict(w=torch.tensor([0.0])), OrderedDict(w=torch.tensor([10.0]))]
    out = weighted_average(states, [0.25, 0.75])
    assert out["w"].item() == pytest.approx(7.5)


def test_trimmed_mean_rejects_outliers():
    stack = torch.tensor([[1.0], [2.0], [3.0], [1000.0]])
    weights = torch.ones(4)
    assert trimmed_mean(stack, weights, 0.25).item() == pytest.approx(2.5)
    assert trimmed_mean(stack, weights, 0.0).item() > 100


def test_norm_filter_drops_poisoned_update():
    updates = _updates([1.0, 1.1, 0.9, 1.0, 50.0])
    keep, rejected = FedAggregator.norm_filter(updates)
    assert rejected == [4]
    assert len(keep) == 4


def test_server_momentum_preserves_step_scale():
    """The momentum blend must not blow the step up on the first round."""
    updates = _updates([1.0, 1.0])
    plain = FedAggregator(AggregationConfig(server_momentum=0.0))
    momentum = FedAggregator(AggregationConfig(server_momentum=0.9))
    zero = OrderedDict(w=torch.zeros(4))
    a, _ = plain.aggregate(zero, updates)
    b, _ = momentum.aggregate(zero, updates)
    assert torch.allclose(a["w"], b["w"], atol=1e-6)


def test_convergence_criterion():
    a = OrderedDict(w=torch.zeros(3))
    b = OrderedDict(w=torch.full((3,), 1e-5))
    converged, distance = has_converged(a, b, 1e-3)
    assert converged and distance < 1e-3
    converged, distance = has_converged(a, OrderedDict(w=torch.ones(3)), 1e-3)
    assert not converged and distance == pytest.approx(math.sqrt(3))


# -------------------------------------------------------------------- privacy
def test_clipping_bounds_the_norm():
    update = OrderedDict(w=torch.full((100,), 5.0))
    clipped, before, was_clipped = clip_state(update, 1.0)
    assert was_clipped and before > 1.0
    assert state_norm(clipped) == pytest.approx(1.0, rel=1e-5)


def test_clipping_is_identity_below_the_bound():
    update = OrderedDict(w=torch.tensor([0.1, 0.1]))
    clipped, _, was_clipped = clip_state(update, 10.0)
    assert not was_clipped and torch.allclose(clipped["w"], update["w"])


def test_analytic_gaussian_is_tighter_than_classical_bound():
    classical = math.sqrt(2 * math.log(1.25 / 1e-5)) / 1.0
    assert analytic_gaussian_sigma(1.0, 1.0, 1e-5) < classical


def test_epsilon_decreases_with_noise_and_grows_with_rounds():
    low = compose_epsilon(0.3, 4.0, 60, 1e-5)
    high = compose_epsilon(0.3, 1.0, 60, 1e-5)
    assert low < high
    assert compose_epsilon(0.3, 2.0, 10, 1e-5) < compose_epsilon(0.3, 2.0, 60, 1e-5)


def test_calibrated_noise_meets_the_budget():
    z = calibrate_noise_multiplier(8.0, 1e-5, 0.3, 60)
    assert compose_epsilon(0.3, z, 60, 1e-5) <= 8.0 + 1e-6


def test_accountant_reports_spend_within_budget():
    accountant = PrivacyAccountant.calibrated(8.0, 1e-5, 0.3, 30, clip_norm=1.0)
    accountant.step(30)
    summary = accountant.summary()
    assert summary["epsilon_spent"] <= summary["epsilon_budget"] + 1e-6
    # secure aggregation splits the noise across participants
    assert accountant.client_sigma(9) == pytest.approx(accountant.sigma / 3.0)


# -------------------------------------------------------------------- metrics
def test_box_iou_matches_hand_computation():
    a = torch.tensor([[0.5, 0.5, 0.4, 0.4]])
    b = torch.tensor([[0.6, 0.5, 0.4, 0.4]])
    # overlap 0.3 x 0.4 = 0.12, union = 0.16 + 0.16 - 0.12 = 0.20
    assert float(box_iou_cxcywh(a, b)) == pytest.approx(0.6, rel=1e-5)
    assert float(box_iou_cxcywh(a, a)) == pytest.approx(1.0)


def test_average_precision_all_point_interpolation():
    ap = average_precision(np.array([0.9, 0.8, 0.7]), np.array([1, 0, 1]), 2)
    assert ap == pytest.approx(0.5 * 1.0 + 0.5 * (2 / 3), rel=1e-6)
    assert average_precision(np.zeros(0), np.zeros(0), 5) == 0.0
