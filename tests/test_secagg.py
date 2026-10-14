"""Quality-hiding, budget-tiered secure aggregation (fedav/server/secagg.py)."""
import numpy as np
import pytest

from fedav.server import secagg as sa


def _round(n_vehicles=6, full=400, ladder_levels=3, drop=(), seed=5, round_id=3):
    rng = np.random.default_rng(seed)
    ladder = sa.tier_ladder(full, full // 8, ladder_levels)
    keys = {c: sa.keypair(seed, c) for c in range(n_vehicles)}
    pubs = {c: k.public for c, k in keys.items()}
    tiers = {c: int(rng.integers(0, len(ladder))) for c in range(n_vehicles)}
    values = {c: rng.normal(0, 0.01, full) for c in range(n_vehicles)}
    weights = {c: float(rng.uniform(0.2, 2.0)) for c in range(n_vehicles)}
    steps = {c: float(rng.integers(5, 40)) for c in range(n_vehicles)}
    vfp, sfp = sa.FixedPoint(0.5), sa.FixedPoint(4096.0)
    uploads = {}
    for c in range(n_vehicles):
        up = sa.vehicle_encode(c, round_id, keys[c], pubs, tiers, ladder, tiers[c],
                               values[c], weights[c], steps[c], vfp, sfp)
        if c not in drop:
            uploads[c] = sa.parse_upload(up.payload)[3]
    sums = sa.rsu_aggregate(uploads, tiers, keys, round_id, ladder, vfp, sfp)
    return ladder, tiers, values, weights, steps, sums, set(uploads)


def _plaintext(ladder, tiers, values, weights, steps, alive):
    out, start = np.zeros(ladder[-1]), 0
    for t, end in enumerate(ladder):
        cover = [c for c in alive if tiers[c] >= t]
        if cover:
            W = sum(weights[c] for c in cover)
            T = sum(weights[c] * steps[c] for c in cover)
            S = sum(weights[c] * values[c][start:end] for c in cover)
            out[start:end] = (T / W) * S / W
        start = end
    return out


@pytest.mark.parametrize("drop", [(), (2,), (0, 4)])
def test_recovers_the_plaintext_weighted_fednova_mean(drop):
    ladder, tiers, values, weights, steps, sums, alive = _round(drop=drop)
    got = sa.combine_segments(sums, ladder, normalize_steps=True)
    want = _plaintext(ladder, tiers, values, weights, steps, alive)
    # masks cancel exactly in Z_2^32; what remains is fixed-point rounding of the
    # weight sums (2^-13 resolution at a 4096 bound), ~1e-4 relative at most
    assert np.max(np.abs(got - want)) <= 1e-4 * np.max(np.abs(want))
    assert sums.seeds_revealed == len(alive) * len(drop)


def test_rsu_learns_sums_not_individual_weights():
    ladder, tiers, values, weights, steps, sums, alive = _round()
    top = sum(weights[c] for c in alive if tiers[c] >= 0)
    assert sums.W[0] == pytest.approx(top, abs=1e-3)
    # a single masked upload decodes to noise, not to that vehicle's weight
    keys = {c: sa.keypair(5, c) for c in range(6)}
    pubs = {c: k.public for c, k in keys.items()}
    up = sa.vehicle_encode(1, 3, keys[1], pubs, tiers, ladder, tiers[1], values[1],
                           weights[1], steps[1], sa.FixedPoint(0.5), sa.FixedPoint(4096.0))
    vec = sa.parse_upload(up.payload)[3]
    decoded_w = sa.FixedPoint(4096.0).decode(vec[ladder[tiers[1]]:])[0]
    assert abs(decoded_w - weights[1]) > 1.0


def test_corrupted_payload_is_detected():
    keys = {0: sa.keypair(1, 0), 1: sa.keypair(1, 1)}
    up = sa.vehicle_encode(0, 1, keys[0], {1: keys[1].public}, {0: 0, 1: 0}, [10], 0,
                           np.ones(10) * 0.01, 1.0, 3.0, sa.FixedPoint(0.5), sa.FixedPoint(64.0))
    raw = bytearray(up.payload)
    raw[-1] ^= 0xFF
    with pytest.raises(sa.PayloadError):
        sa.parse_upload(bytes(raw))


def test_clipping_excess_returns_to_residual():
    keys = {0: sa.keypair(1, 0)}
    values = np.array([0.1, 2.0, -3.0])
    up = sa.vehicle_encode(0, 1, keys[0], {}, {0: 0}, [3], 0, values, 1.0, 1.0,
                           sa.FixedPoint(1.0), sa.FixedPoint(64.0))
    np.testing.assert_allclose(up.clipped_excess, [0.0, 1.0, -2.0], atol=1e-6)


def test_pair_seeds_agree_and_need_a_private_key():
    a, b, c = sa.keypair(9, 0), sa.keypair(9, 1), sa.keypair(9, 2)
    assert sa.pair_seed(a, b.public, 4) == sa.pair_seed(b, a.public, 4)
    assert sa.pair_seed(a, b.public, 4) != sa.pair_seed(c, b.public, 4)
    assert sa.pair_seed(a, b.public, 4) != sa.pair_seed(a, b.public, 5)


def test_support_is_ordered_by_previous_update_magnitude():
    prev = np.array([0.0, -5.0, 1.0, 3.0, 0.5])
    np.testing.assert_array_equal(sa.shared_support(prev, 5, 3, 0, 2), [1, 3, 2])
    first = sa.shared_support(None, 100, 10, seed=4, round_id=1)
    assert len(set(first.tolist())) == 10


def test_tiers_keep_each_vehicle_inside_its_airtime_share():
    ladder = sa.tier_ladder(100_000, 5_000, 5)
    rates = {0: 2.0, 1: 8.0, 2: 30.0}
    budget = 0.6
    tiers = sa.assign_tiers(rates, ladder, budget)
    assert tiers[0] <= tiers[1] <= tiers[2]
    for c, t in tiers.items():
        if t > 0:
            airtime = (ladder[t] + 2 * (t + 1)) * 4 * 8 / (rates[c] * 1e6)
            assert airtime <= budget / len(rates) + 1e-9
