"""Standards-based V2I radio model: bytes on the wire -> airtime and energy.

The rest of the simulator counts exact bytes. This module turns those bytes into
the quantities a radio link actually spends - seconds of shared channel and
joules of transmitter energy - so a byte saving can be read as a latency and
energy saving.

Model (one uplink transmission, vehicle -> road-side unit)
----------------------------------------------------------
* **Geometry.** Each transmission samples the vehicle-RSU distance ``d`` from a
  region-dependent range (urban cells are small, rural ones large) and decides
  line-of-sight with the 3GPP TR 38.901 UMi-Street-Canyon LOS probability

      P_LOS(d) = min(18/d, 1) * (1 - exp(-d/36)) + exp(-d/36)

  3GPP TR 37.885 (V2X evaluation methodology) uses the TR 38.901 UMi model for
  vehicle-RSU links.
* **Path loss** (TR 38.901 Table 7.4.1-1, UMi-Street-Canyon, ``f_c`` in GHz):

      PL_LOS  = 32.4 + 21.0 log10(d) + 20.0 log10(f_c)
      PL_NLOS = max(PL_LOS, 22.4 + 35.3 log10(d) + 21.3 log10(f_c) - 0.3 (h_UT - 1.5))

  plus log-normal shadowing (sigma 4 dB LOS, 7.82 dB NLOS).
* **Rate.** SNR = P_tx - PL - (N0 + 10 log10 B + NF); achievable rate
  ``eta * B * log2(1 + SNR)``, capped at the spectral efficiency of the highest
  modulation-and-coding scheme. ``eta`` is the usual implementation gap.
* **Airtime** = TLS-framed bytes * 8 / rate + per-transmission access latency.
  The uplink is one shared channel, so a round's uplink airtime is the *sum* over
  transmitting vehicles - the time the cell is occupied.
* **Energy** = (P_tx / PA efficiency) * airtime: radio energy only, not compute.

Defaults are ITS 5.9 GHz band, 10 MHz channel, 23 dBm transmit power. They are
nominal evaluation-methodology values, not a measurement of any real radio.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

# (min, max) vehicle-RSU distance in metres by region.
REGION_DISTANCE_M: Dict[str, Tuple[float, float]] = {
    "urban": (15.0, 250.0),
    "highway": (30.0, 500.0),
    "rural": (50.0, 800.0),
}


@dataclass(frozen=True)
class LinkSample:
    distance_m: float
    los: bool
    path_loss_db: float
    snr_db: float
    rate_mbps: float
    airtime_s: float
    energy_j: float

    def as_dict(self) -> Dict[str, float]:
        return {"distance_m": round(self.distance_m, 1), "los": self.los,
                "snr_db": round(self.snr_db, 2), "rate_mbps": round(self.rate_mbps, 3),
                "airtime_s": round(self.airtime_s, 5), "energy_j": round(self.energy_j, 5)}


@dataclass
class V2XRadioModel:
    carrier_ghz: float = 5.9
    bandwidth_mhz: float = 10.0
    tx_power_dbm: float = 23.0
    noise_figure_db: float = 9.0
    pa_efficiency: float = 0.25
    implementation_gap: float = 0.75     # fraction of Shannon capacity achieved
    max_spectral_eff: float = 5.55       # bit/s/Hz, 64-QAM high-rate MCS
    min_rate_mbps: float = 0.5           # below this the link is effectively out
    access_latency_s: float = 0.002      # scheduling / channel access per message
    ue_height_m: float = 1.5
    regions: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: dict(REGION_DISTANCE_M))

    # ------------------------------------------------------------- 3GPP pieces
    @staticmethod
    def los_probability(d: float) -> float:
        return min(18.0 / d, 1.0) * (1.0 - math.exp(-d / 36.0)) + math.exp(-d / 36.0)

    def path_loss_db(self, d: float, los: bool) -> float:
        fc = self.carrier_ghz
        pl_los = 32.4 + 21.0 * math.log10(d) + 20.0 * math.log10(fc)
        if los:
            return pl_los
        pl_nlos = (22.4 + 35.3 * math.log10(d) + 21.3 * math.log10(fc)
                   - 0.3 * (self.ue_height_m - 1.5))
        return max(pl_los, pl_nlos)

    def noise_dbm(self) -> float:
        return -174.0 + 10.0 * math.log10(self.bandwidth_mhz * 1e6) + self.noise_figure_db

    def rate_mbps(self, snr_db: float) -> float:
        snr = 10.0 ** (snr_db / 10.0)
        eff = min(self.implementation_gap * math.log2(1.0 + snr), self.max_spectral_eff)
        return max(eff * self.bandwidth_mhz, self.min_rate_mbps)

    def tx_power_w(self) -> float:
        return 10.0 ** ((self.tx_power_dbm - 30.0) / 10.0)

    # ------------------------------------------------------------------ public
    def sample(self, size_bytes: int, region: str, generator: np.random.Generator) -> LinkSample:
        lo, hi = self.regions.get(region, self.regions["urban"])
        d = float(generator.uniform(lo, hi))
        los = bool(generator.random() < self.los_probability(d))
        shadow = float(generator.normal(0.0, 4.0 if los else 7.82))
        pl = self.path_loss_db(d, los) + shadow
        snr_db = self.tx_power_dbm - pl - self.noise_dbm()
        rate = self.rate_mbps(snr_db)
        airtime = self.access_latency_s + size_bytes * 8.0 / (rate * 1e6)
        energy = self.tx_power_w() / self.pa_efficiency * airtime
        return LinkSample(d, los, pl, snr_db, rate, airtime, energy)

    def airtime_s(self, size_bytes: int, rate_mbps: float) -> float:
        """Airtime for a given payload at a known rate (for analytic checks)."""
        return self.access_latency_s + size_bytes * 8.0 / (rate_mbps * 1e6)


__all__ = ["V2XRadioModel", "LinkSample", "REGION_DISTANCE_M"]
