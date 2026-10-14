# Embodiments, technical means and best method

Source text for the "Detailed description" of Form 2. Section 10(4) of the
Patents Act requires a description full enough for a skilled engineer to work the
invention, and the best method known to the applicant; the CRI Guidelines 2025
want the conventional hardware recited as the technical means on which the
technical effect is produced. Reference numerals are those of
[REFERENCE_NUMERALS.md](REFERENCE_NUMERALS.md) and the drawings in `drawings/`.

## 1. Technical problem

A fleet of connected vehicles (100, 102) should improve one shared perception
and trajectory model from the data its sensors record, without that data leaving
the vehicles. Three properties of the vehicle-to-infrastructure (V2I) radio link
(130) make the ordinary federated-averaging remedy fall short:

1. **Uplink airtime is the scarce resource.** All vehicles in a road-side unit's
   coverage cell (210) share one uplink channel; a vehicle at the cell edge or out
   of line of sight transmits the same model update several times slower than one
   near the antenna, so one slow vehicle holds the round open for everyone.
2. **Vehicles' contributions differ in value** (a de-calibrated camera, a vehicle
   that only ever sees one road type), so an aggregator should weight them by a
   quality score. But a quality score computed from a vehicle's sensor health and
   label distribution is itself sensitive information about that vehicle, and a
   plaintext weighted aggregator must receive every score and every update.
3. **Secure aggregation and sparsification conflict.** Pairwise masking hides
   individual updates, but a masked sparse vector is dense, so masking
   per-vehicle top-k updates forfeits the bandwidth saving that sparsification was
   for. Uploads are also lost or corrupted on a mobile link, and an
   error-feedback compressor that has already removed the transmitted part from
   its residual then loses it for good.

## 2. Technical solution (summary)

* The road-side unit (200) predicts each selected vehicle's uplink rate from its
  reported position (3GPP TR 38.901 UMi path loss, as used for V2I links by TR
  37.885) and gives it the longest prefix, from a public ladder of prefix lengths
  (802), that fits an equal share of a per-round uplink airtime budget.
* All vehicles and the road-side unit derive the same ordered coordinate support
  (800) from the previous round's global update, so no indices are sent and every
  vehicle's prefix is a prefix of the same list.
* Each vehicle folds its private aggregation weight `w_k = q_k n_k / n_ref` and
  its local step count `tau_k` into the payload: it sends fixed-point ring
  elements of `w_k u_k` on its prefix (`u_k` = step-normalised update plus
  residual) and, per covered segment (801), of `w_k` and `w_k tau_k`, then adds
  pairwise masks derived by Diffie-Hellman, over only the coordinates both
  vehicles of a pair send.
* The road-side unit sums in the ring, so the masks cancel *exactly*, and obtains
  per segment `S = Σ w u`, `W = Σ w`, `T = Σ w tau`; the update is
  `(T/W) · S/W` — quality-weighted and FedNova-normalised — while no individual
  `q_k`, `n_k`, `tau_k` or update is ever visible to it.
* A failed upload (lost, or CRC-32 mismatch) triggers a negative
  acknowledgement; the vehicle keeps its whole update in the residual (115) for
  the next round, and the survivors reveal their pair seeds with the missing
  vehicle so its masks can be removed.

## 3. Hardware embodiment

### 3.1 On-board unit (110), FIG. 2

| Element | Example implementation |
|---|---|
| Sensors (120) | front camera 1920×1080 at 30 fps; optionally LiDAR and radar |
| Sensor interface (111) | GMSL2 / automotive Ethernet camera inputs |
| Sensor-health / calibration monitor (112) | computes blur, occlusion and exposure statistics per frame and checks the extrinsic-calibration status; outputs the sensor-health signal used in `q_k` |
| Processor + neural accelerator (113) | automotive SoC with a neural accelerator (for example 30–250 INT8 TOPS class); runs local training at night or while parked, or a few minibatches per drive |
| Memory (114) | LPDDR DRAM; holds the model copy, the local training data store (116) and the error-feedback residual (115), one float per model parameter (2.8 MB for the 0.70 M-parameter reference model) |
| Secure element (117) | holds the vehicle's Diffie-Hellman private key; performs the modular exponentiation for each pair seed |
| V2X modem (118) and antenna (119) | C-V2X (PC5 sidelink or Uu) or IEEE 802.11p/bd in the 5.9 GHz ITS band, 10 MHz channel, 23 dBm transmit power |

### 3.2 Road-side unit (200), FIG. 3

| Element | Example implementation |
|---|---|
| V2I transceiver (201) | 5.9 GHz C-V2X / ITS-G5 radio covering a cell of 250–800 m radius |
| Link-rate estimator (202) | converts each vehicle's reported position (cooperative awareness messages) into a predicted rate through the mean path loss of §4.2 |
| Vehicle selector (203) | applies the uptime threshold, regional stratification and starvation guard |
| Tier scheduler (204) | assigns prefix lengths by §4.4 |
| Secure aggregation unit (205) | 32-bit modular adders over the received vectors; seed-reveal handling |
| Processor (208), memory (209) | edge server class; holds the global model (206) and the previous global update from which the support is derived |
| Backhaul interface (207) | fibre or cellular link to the model repository (310) |

## 4. Best method (parameters used for the measured results)

All values are those of `configs/patent_eval.yaml` and
`fedav/sim/baselines.py::proposed_config` unless stated.

### 4.1 Model and data

* Model: multitask CNN, 0.70 M parameters (`fedav/models/perception.py`):
  convolutional stem of 24 channels, 192-dimensional embedding, heads for
  6-class object classification, bounding-box regression and a 6-step
  trajectory; loss = classification + 1.0 × box + 0.5 × trajectory.
* Data characteristics (CRI 2025 asks for characteristics, not the dataset):
  32×32 scene crops, 6 object classes; 50 vehicles; label skew Dirichlet
  α = 0.4; quantity skew log-normal σ = 0.45 around 220 samples; each vehicle
  has a region (urban, highway, rural) and weather (clear, rain, fog, night)
  that change image statistics; each vehicle is de-calibrated with probability
  0.12, with 25–45% label corruption, and rain, fog and night add 5%, 10% and 8%
  label noise. Split 70/15/15 per vehicle.
* Local training: SGD, learning rate 0.01 with cosine decay, momentum 0.9,
  weight decay 5·10⁻⁴, batch 32, 2 local epochs, gradient clip 5.0.

### 4.2 Radio model (`fedav/server/v2x.py`)

* Distance per round uniform in [15, 250] m (urban), [30, 500] m (highway),
  [50, 800] m (rural); LOS probability
  `min(18/d, 1)(1 − e^(−d/36)) + e^(−d/36)`.
* Path loss (TR 38.901 UMi-Street-Canyon, f_c = 5.9 GHz):
  LOS `32.4 + 21 log10 d + 20 log10 f_c`; NLOS
  `max(LOS, 22.4 + 35.3 log10 d + 21.3 log10 f_c)`; shadowing 4 dB (LOS),
  7.82 dB (NLOS).
* Noise `−174 dBm/Hz + 10 log10(10 MHz) + 9 dB`; rate
  `0.75 · B · log2(1 + SNR)`, capped at 5.55 bit/s/Hz, floored at 0.5 Mbit/s;
  2 ms access latency per message; energy = 0.2 W / 0.25 × airtime.
* The road-side unit's prediction uses the mean path loss (no shadowing); the
  actual transmission includes shadowing.

### 4.3 Selection

Fraction C = 0.3 (15 of 50 vehicles); predicted uptime (long-run uptime plus
N(0, 0.08) jitter) ≥ 0.55, relaxed by ×0.7 steps if too few qualify; regional
round-robin by score `0.55·uptime + 0.25·sensor quality + 0.10·log(1+n)/10 +
0.10/(1+times selected)`; starvation guard K = 12 rounds, applied before the
uptime gate.

### 4.4 Support, tiers and encoding (`fedav/server/secagg.py`)

* Top prefix length `L_max = round(r · P)` with P parameters and r the current
  top-k ratio (initial 0.25, adapted ×0.7 / ×1.15 to keep traffic under
  9 MB per round); ladder of 5 lengths geometric from `0.02 · P` to `L_max`.
* Support: coordinates sorted by |previous global update|, ties stable;
  round 1: permutation seeded by (run seed, round).
* Tier: largest t with `(L_t + 2(t+1)) · 4 bytes ≤ (budget / |S|) · rate / 8`;
  budget 12 s per round in the measured runs.
* Weight `w_k = clamp(q_k, 0.05, 1) · n_k / n_ref`, `n_ref = 220 × 0.7`,
  capped at 4.
* Values: ring Z_2^32, signed fixed point with ⌈log2 64⌉ = 6 bits of headroom,
  clipping bound `B_r = 4 · w_max · max |d_{r−1}|` (initial 0.05 · w_max);
  scalars use bound `w_max · τ_cap`.
* Masks: RFC 3526 2048-bit MODP group, generator 2; pair seed = first 8 bytes of
  SHA-256(g^{ab} mod p ‖ round); PCG64 stream of uniform 32-bit words.

### 4.5 Message format (`fedav/server/secagg.py`, FIG. 7)

| Field (numeral) | Bytes | Content |
|---|---|---|
| magic (701) | 4 | `FLSA` |
| vehicle identifier (702) | 4 | uint32 |
| round identifier (703) | 4 | uint32 |
| tier (704) | 4 | uint32 |
| value count (705) | 4 | uint32, `L_tier + 2(tier+1)` |
| CRC-32 (706) | 4 | over the body |
| masked values (707) | 4 · L_tier | ring elements |
| masked segment scalars (708) | 8 · (tier+1) | `[w, w·τ]` per covered segment |

Key set-up adds 256 bytes uplink per vehicle (first participation) and 256 bytes
per selected vehicle on the multicast downlink; each seed reveal adds 32 bytes.
All of this is counted in the reported traffic.

### 4.6 Aggregation

Per segment `d_t = (T_t/W_t) · S_t/W_t`, scattered onto the support; server
momentum β = 0.9 with the convex blend `(1−β) v + β d`; `w ← w + that`.

## 5. Alternative embodiments

* Ring of 2^64 instead of 2^32 when updates need more dynamic range.
* Elliptic-curve Diffie-Hellman (X25519, 32-byte keys) in place of the MODP
  group; same protocol, 224 fewer bytes per key.
* Shamir secret sharing of the pair seeds (Bonawitz et al.) when the road-side
  unit is not trusted to follow the seed-reveal step honestly.
* Tier ladder of other lengths, or assignment by predicted energy instead of
  airtime.
* Co-simulation with a network simulator (ns-3, Veins/SUMO) for mobility and
  MAC-layer contention; not built.

## 6. What the simulator does *not* show

* A single shared uplink with no MAC contention model; airtime is summed, which
  is the occupancy of the channel, not a measured latency.
* Synthetic scenes, not a public driving dataset.
* Private keys are derived from the run seed for reproducibility; a deployment
  generates them in the secure element.
