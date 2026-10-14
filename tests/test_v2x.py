"""V2I radio model: 3GPP path loss, rate, and byte -> airtime/energy mapping."""
import math

import numpy as np
import pytest

from fedav.server.v2x import V2XRadioModel


def test_umi_los_path_loss_matches_tr38901_formula():
    radio = V2XRadioModel()
    # 32.4 + 21 log10(100) + 20 log10(5.9) = 32.4 + 42 + 15.417
    assert radio.path_loss_db(100.0, los=True) == pytest.approx(89.817, abs=1e-3)


def test_nlos_never_beats_los():
    radio = V2XRadioModel()
    for d in (10, 50, 200, 800):
        assert radio.path_loss_db(d, los=False) >= radio.path_loss_db(d, los=True)


def test_los_probability_is_one_near_and_falls_with_distance():
    p = V2XRadioModel.los_probability
    assert p(10.0) == pytest.approx(1.0)
    assert p(50.0) > p(200.0) > p(800.0) > 0


def test_rate_is_capped_and_floored():
    radio = V2XRadioModel()
    assert radio.rate_mbps(80.0) == pytest.approx(radio.max_spectral_eff * radio.bandwidth_mhz)
    assert radio.rate_mbps(-40.0) == radio.min_rate_mbps


def test_airtime_and_energy_scale_linearly_with_bytes():
    radio = V2XRadioModel()
    a = radio.sample(1_000_000, "urban", np.random.default_rng(3))
    b = radio.sample(2_000_000, "urban", np.random.default_rng(3))   # same channel draw
    assert a.rate_mbps == b.rate_mbps
    payload_a = a.airtime_s - radio.access_latency_s
    payload_b = b.airtime_s - radio.access_latency_s
    assert payload_b == pytest.approx(2 * payload_a)
    assert b.energy_j / a.energy_j == pytest.approx(b.airtime_s / a.airtime_s)
    assert a.energy_j == pytest.approx(radio.tx_power_w() / radio.pa_efficiency * a.airtime_s)


def test_rural_links_are_slower_on_average():
    radio = V2XRadioModel()
    g = np.random.default_rng(0)
    urban = np.mean([radio.sample(1, "urban", g).rate_mbps for _ in range(400)])
    rural = np.mean([radio.sample(1, "rural", g).rate_mbps for _ in range(400)])
    assert urban > rural
