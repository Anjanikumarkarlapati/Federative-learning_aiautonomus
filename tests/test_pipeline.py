"""Component and integration tests for the federated pipeline (PRD 17)."""
from collections import OrderedDict

import numpy as np
import pytest
import torch

from fedav.client.compression import BandwidthController, TopKCompressor
from fedav.client.serializer import ChecksumError, UpdateSerializer
from fedav.client.trainer import ClientTrainer
from fedav.config import CommConfig, ModelConfig, SelectionConfig, default_config
from fedav.data.federated import VehicleProfile, build_fleet, dirichlet_class_probs
from fedav.metrics.evaluation import evaluate
from fedav.models.perception import build_model, multitask_loss
from fedav.server.secure import SecureAggregationMasks, SecureChannel, verify_masks_cancel
from fedav.server.selection import ClientSelector
from fedav.utils.tensors import num_parameters, state_norm, state_sub, to_float_state
from tests.conftest import tiny_config


# --------------------------------------------------------------------- model
def test_model_respects_the_parameter_cap():
    model = build_model(ModelConfig(), 6, 6, seed=0)
    assert num_parameters(model) <= ModelConfig().max_parameters


def test_model_rejects_oversized_configuration():
    with pytest.raises(ValueError):
        build_model(ModelConfig(max_parameters=1000), 6, 6, seed=0)


def test_model_outputs_have_the_right_shapes_and_ranges():
    model = build_model(ModelConfig(), 6, 6, seed=0)
    out = model(torch.rand(3, 3, 32, 32))
    assert out["logits"].shape == (3, 6)
    assert out["box"].shape == (3, 4)
    assert out["trajectory"].shape == (3, 12)
    assert (out["box"] >= 0).all() and (out["box"] <= 1).all()


def test_model_contains_no_batchnorm():
    """BatchNorm running stats are client-specific and must not be averaged."""
    model = build_model(ModelConfig(), 6, 6, seed=0)
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
                   for m in model.modules())


def test_local_sgd_step_decreases_the_loss():
    torch.manual_seed(0)
    cfg = ModelConfig()
    model = build_model(cfg, 6, 6, seed=0)
    images = torch.rand(16, 3, 32, 32)
    labels = torch.randint(1, 6, (16,))
    boxes = torch.rand(16, 4).clamp(0.1, 0.9)
    trajectories = torch.randn(16, 12)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    first = None
    for step in range(25):
        optimizer.zero_grad()
        loss, _ = multitask_loss(model(images), labels, boxes, trajectories, cfg)
        loss.backward()
        optimizer.step()
        first = float(loss.detach()) if first is None else first
    assert float(loss.detach()) < first


# ---------------------------------------------------------------- serializer
def test_serializer_round_trip_preserves_values():
    serializer = UpdateSerializer(CommConfig(fp16=True))
    update = OrderedDict(a=torch.randn(32, 3, 3), b=torch.randn(16))
    blob = serializer.pack(update, {"client_id": 1, "n_k": 10, "q_k": 0.5, "round_id": 2})
    restored, meta = serializer.unpack(blob)
    assert meta["client_id"] == 1 and meta["round_id"] == 2
    for key in update:
        assert restored[key].shape == update[key].shape
        assert torch.allclose(restored[key], update[key], atol=1e-2)


def test_serializer_fp32_round_trip_is_exact():
    serializer = UpdateSerializer(CommConfig(fp16=False))
    update = OrderedDict(a=torch.randn(64))
    restored, _ = serializer.unpack(serializer.pack(update, {}))
    assert torch.allclose(restored["a"], update["a"], atol=1e-7)


def test_serializer_detects_corruption():
    serializer = UpdateSerializer(CommConfig())
    blob = serializer.pack(OrderedDict(a=torch.randn(128)), {"client_id": 0})
    corrupted = bytearray(blob)
    corrupted[-3] ^= 0xFF
    with pytest.raises(ChecksumError):
        serializer.unpack(bytes(corrupted))


def test_fp16_halves_the_payload():
    update = OrderedDict(a=torch.randn(20000))
    small = UpdateSerializer(CommConfig(fp16=True)).pack(update, {})
    large = UpdateSerializer(CommConfig(fp16=False)).pack(update, {})
    assert len(small) < 0.6 * len(large)


# --------------------------------------------------------------- compression
def test_topk_keeps_the_largest_coordinates():
    compressor = TopKCompressor(ratio=0.1, error_feedback=False)
    update = OrderedDict(a=torch.arange(100, dtype=torch.float32))
    sparse = compressor.compress(0, update)
    assert int((sparse["a"] != 0).sum()) == 10
    assert sparse["a"][-1] == 99.0


def test_error_feedback_is_lossless_in_aggregate():
    """Nothing is discarded - only delayed - so the running sum converges."""
    compressor = TopKCompressor(ratio=0.2, error_feedback=True)
    update = OrderedDict(a=torch.randn(500))
    transmitted = torch.zeros(500)
    rounds = 40
    for _ in range(rounds):
        transmitted += compressor.compress(0, OrderedDict(a=update["a"].clone()))["a"]
    # The gap between what was sent and the true cumulative update is exactly the
    # residual still held on the vehicle.  Error-feedback theory bounds it by
    # ((1-delta)/delta) * ||u|| for a delta-contraction compressor, so it stays
    # O(1) while the transmitted sum grows with the number of rounds - the
    # relative error therefore decays like 1/rounds instead of accumulating.
    norm = torch.linalg.norm(update["a"])
    gap = torch.linalg.norm(transmitted - update["a"] * rounds)
    assert gap <= ((1 - 0.2) / 0.2) * norm
    assert gap / (rounds * norm) < 0.05


def test_bandwidth_controller_tightens_then_relaxes():
    controller = BandwidthController(9.0, initial_ratio=1.0)
    assert controller.update(30.0) < 1.0
    tight = controller.update(30.0)
    assert controller.update(1.0) > tight


# ----------------------------------------------------------- secure channel
def test_secure_masks_cancel_exactly():
    reference = OrderedDict(a=torch.zeros(20), b=torch.zeros(4, 4))
    masks = SecureAggregationMasks(round_id=3, participants=[1, 4, 9, 11])
    assert verify_masks_cancel(masks, reference)


def test_individual_masked_update_is_hidden():
    reference = OrderedDict(a=torch.ones(64))
    masks = SecureAggregationMasks(round_id=1, participants=[0, 1, 2])
    masked = masks.apply(reference, 0)
    assert state_norm(state_sub(masked, reference)) > 1.0      # unrecognisable alone


def test_channel_accounts_for_bytes_and_drops():
    channel = SecureChannel()
    generator = np.random.default_rng(0)
    delivered = [channel.send_up(b"x" * 1000, 0.0, generator)[0] for _ in range(5)]
    assert not any(delivered) and channel.stats.drops == 5
    channel.send_down(b"y" * 1000, 4)
    assert channel.stats.downlink_bytes > 4000


# -------------------------------------------------------------------- data
def test_dirichlet_partition_produces_skew():
    skewed = dirichlet_class_probs(200, 6, 0.1, np.random.default_rng(0))
    uniform = dirichlet_class_probs(200, 6, 100.0, np.random.default_rng(0))
    assert skewed.max(axis=1).mean() > uniform.max(axis=1).mean()
    assert np.allclose(skewed.sum(axis=1), 1.0)


def test_fleet_splits_and_metadata(fleet, cfg):
    assert len(fleet) == cfg.num_clients
    for vehicle in fleet:
        assert len(vehicle.train) > 0 and len(vehicle.val) > 0 and len(vehicle.test) > 0
        assert 0.0 <= vehicle.profile.quality_signal <= 1.0
        assert 0.0 <= vehicle.profile.link_uptime <= 1.0
        assert vehicle.profile.label_histogram.sum() == pytest.approx(1.0)
    assert fleet.total_samples == sum(v.n_k for v in fleet)
    assert np.isclose(fleet.reference_histogram.sum(), 1.0)


def test_no_raw_data_in_the_transmitted_payload(cfg, fleet):
    """FR-02: the wire payload must contain weights only, never sensor data."""
    model = build_model(cfg.model, cfg.data.num_classes, cfg.data.traj_horizon, seed=0)
    trainer = ClientTrainer(cfg, model)
    update = trainer.train(to_float_state(model.state_dict()), fleet[0], 1,
                           fleet.reference_histogram)
    serializer = UpdateSerializer(cfg.comm)
    blob = serializer.pack(update.delta, update.meta())
    images = fleet[0].train.images.numpy().astype(np.float16).tobytes()
    assert images[:512] not in blob
    assert set(update.meta()) == {"client_id", "round_id", "n_k", "q_k",
                                  "local_epochs", "steps", "train_loss", "update_norm"}


# --------------------------------------------------------------- selection
def _profiles(n=12):
    return [VehicleProfile(i, ["urban", "highway", "rural"][i % 3], "clear", True,
                           50, 0.3 + 0.05 * (i % 10), 1.0, 0.9) for i in range(n)]


def test_selection_respects_connectivity_and_size():
    cfg = SelectionConfig(fraction=0.25, uptime_threshold=0.6, seed_jitter=False,
                          fairness_k=10**6)
    selector = ClientSelector(cfg, 12, seed=0)
    profiles = _profiles()
    result = selector.select(profiles, 1)
    assert len(result.selected) == 3
    for client_id in result.selected:
        assert profiles[client_id].link_uptime >= 0.6


def test_selection_relaxes_when_connectivity_collapses():
    cfg = SelectionConfig(fraction=0.5, uptime_threshold=0.99, seed_jitter=False)
    selector = ClientSelector(cfg, 12, seed=0)
    result = selector.select(_profiles(), 1)
    assert result.relaxed and len(result.selected) == 6


def test_fairness_guard_bounds_starvation():
    cfg = SelectionConfig(fraction=0.25, uptime_threshold=0.0, fairness_k=3,
                          seed_jitter=False)
    selector = ClientSelector(cfg, 12, seed=0)
    for round_id in range(1, 25):
        selector.select(_profiles(), round_id)
    assert selector.coverage()["clients_never_selected"] == 0


def test_trajectory_aware_selection_covers_regions():
    cfg = SelectionConfig(fraction=0.25, uptime_threshold=0.0, trajectory_aware=True,
                          fairness_k=10**6, seed_jitter=False)
    selector = ClientSelector(cfg, 12, seed=0)
    profiles = _profiles()
    selected = selector.select(profiles, 5).selected
    assert len({profiles[i].region for i in selected}) == 3


# ------------------------------------------------------------- integration
def test_adaptive_local_epochs_track_compute(cfg, fleet):
    """FR-10 is off by default (measured net-negative) but must still work."""
    cfg = cfg.merged({"train": {"adaptive_epochs": True}})
    model = build_model(cfg.model, cfg.data.num_classes, cfg.data.traj_horizon, seed=0)
    trainer = ClientTrainer(cfg, model)
    fast, slow = fleet[0], fleet[1]
    fast.profile.compute_capability, slow.profile.compute_capability = 3.0, 0.2
    assert trainer.local_epochs_for(fast) > trainer.local_epochs_for(slow)

    # ...and must be inert when the switch is off, whatever the hardware.
    fixed = ClientTrainer(cfg.merged({"train": {"adaptive_epochs": False}}), model)
    assert fixed.local_epochs_for(fast) == fixed.local_epochs_for(slow)


def test_full_federated_round_updates_the_global_model():
    from fedav.sim.runner import SimRunner
    runner = SimRunner(tiny_config(train={"rounds": 2, "local_epochs": 1}), echo=False)
    before = {k: v.clone() for k, v in runner.global_state.items()}
    result = runner.run()
    moved = max(float((runner.global_state[k] - before[k]).abs().max()) for k in before)
    assert moved > 0
    assert result.rounds_run >= 1
    assert result.history and result.history[0]["clients_reported"] >= 1
    assert result.final_eval.samples > 0
    assert result.privacy["raw_data_egress"] is False


def test_run_is_reproducible_under_a_fixed_seed():
    from fedav.sim.runner import SimRunner
    accuracies = []
    for _ in range(2):
        runner = SimRunner(tiny_config(name="repro", train={"rounds": 2, "local_epochs": 1}),
                           echo=False)
        accuracies.append(runner.run().final_eval.accuracy)
    assert accuracies[0] == pytest.approx(accuracies[1], abs=1e-9)


def test_secure_aggregation_reproduces_the_plain_weighted_mean():
    from fedav.sim.runner import SimRunner
    runner = SimRunner(tiny_config(name="secure", train={"rounds": 1, "local_epochs": 1}),
                       echo=False)
    result = runner.run()
    assert all(record.get("secure_aggregation_verified") for record in result.history
               if record.get("clients_reported"))


def test_dp_run_stays_within_the_privacy_budget():
    from fedav.sim.runner import SimRunner
    cfg = tiny_config(name="dp", train={"rounds": 2, "local_epochs": 1},
                      dp={"enabled": True, "epsilon": 4.0})
    result = SimRunner(cfg, echo=False).run()
    assert result.privacy["epsilon_spent"] <= result.privacy["epsilon_budget"] + 1e-6
    assert result.privacy["noise_multiplier"] > 0
