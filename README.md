# FL-AV — Federated Learning Framework for Collaborative Autonomous-Vehicle Training

A privacy-preserving distributed learning platform for Connected & Autonomous Vehicles (CAVs).
A fleet of `N` vehicles jointly trains one shared perception / trajectory-prediction model
**without a single raw sensor sample leaving any vehicle** — only model deltas (`ΔW`) cross the
V2I link to a Road-Side Unit (RSU) that aggregates them.

The implementation follows *PRD — FL Framework for Collaborative AV Training, v1.0* section by
section. It builds on **FedAvg** and adds the four enhancements the PRD specifies:

| Enhancement | Effect |
|---|---|
| **Quality-aware aggregation** (`q_k`) | off-distribution / sensor-degraded vehicles contribute less |
| **Connectivity- & trajectory-aware selection** | rounds don't stall on unstable V2I links, regional coverage preserved |
| **Adaptive local epochs** (`E_k`) | implemented, but **measured harmful and off by default** — see [section 9](#9-results) |
| **Optional DP + secure aggregation** | formal `(ε, δ)` budget; the RSU only ever sees a *sum* |

---

## 1. Table of contents

- [2. Quick start](#2-quick-start)
- [3. How it works](#3-how-it-works)
- [4. The mathematics](#4-the-mathematics)
- [5. The ML model](#5-the-ml-model)
- [6. Data & Non-IID partitioning](#6-data--non-iid-partitioning)
- [7. Communication & privacy](#7-communication--privacy)
- [8. Configuration reference](#8-configuration-reference)
- [9. Results](#9-results)
- [Mathematical validation](docs/MATHEMATICAL_VALIDATION.md) — 14 figures verifying the equations
- [10. Testing](#10-testing)
- [11. Project layout](#11-project-layout)
- [12. PRD traceability](#12-prd-traceability)
- [13. Limitations & roadmap](#13-limitations--roadmap)
- [14. References](#14-references)

---

## 2. Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# what the configured fleet looks like (model size, clients, skew)
python -m fedav.cli info --config configs/default.yaml

# an 8-round smoke run (~30 seconds on 4 CPU cores)
python -m fedav.cli run --config configs/quick.yaml

# the full PRD experiment: proposed FL, N = 50, C = 0.3, R_max = 60
python -m fedav.cli run --config configs/default.yaml

# all four arms of PRD 12.2 + the PRD 12.3 acceptance check
python -m fedav.cli benchmark --config configs/default.yaml

# ablation: which ingredient is doing the work (PRD 17)
python -m fedav.cli ablation --config configs/quick.yaml

# differential privacy enabled (ε = 8, δ = 1e-5)
python -m fedav.cli run --config configs/dp.yaml
```

### Dashboard

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py          # opens http://localhost:8501
```

| Page | What it shows |
|---|---|
| Overview | headline numbers vs. FedAvg and the PRD 12.3 acceptance checklist |
| Fleet (every vehicle) | all cars in one table (filter by region / weather / calibration, CSV export); accuracy per car, worst to best; sensor quality and label similarity vs. accuracy; a drill-down per car (profile, label mix, accuracy every round, when it took part); who-took-part and per-car accuracy heatmaps over rounds; region and weather breakdowns; each car's accuracy under every benchmark method |
| Benchmark | the four arms round by round: accuracy, uplink MB, loss, ADE, worst-served vehicle |
| Ablation | each enhancement in isolation (`results/ingredients`, `attribution`, `fix`) |
| Launch a run | set N, C, E, rounds, quality/trajectory-aware, DP ε, top-k, Byzantine defence; start it and watch rounds stream in |
| Run explorer | any run under `runs/` or `results/`, with per-round table, summary, config and CSV download |
| Validation | the 14 mathematical-validation figures |

The dashboard only reads what the framework already writes and launches runs through the same
CLI below, so nothing in `fedav/` depends on Streamlit.

Any config field can be overridden from the command line:

```bash
python -m fedav.cli run --config configs/default.yaml \
    --set num_clients=100 selection.fraction=0.2 aggregation.quality_aware=false
```

Every run writes to `runs/<name>/`:

```
runs/proposed_fl/
├── config.json      # exact config, plus a fingerprint hash for reproducibility
├── rounds.jsonl     # one JSON record per round (FR-08)
├── rounds.csv       # the same metrics, spreadsheet-friendly
├── summary.json     # final accuracy, rounds-to-target, MB/round, privacy spend
├── vehicles.json          # every vehicle's profile: region, weather, calibration,
│                          #   shard sizes, V2I uptime, compute, sensor quality, label mix
├── vehicle_rounds.jsonl   # every vehicle, every round: selected / reported / dropped /
│                          #   straggler / rejected, q_k, weight α_k, local loss, and the
│                          #   global model's accuracy on that car's own test split
├── vehicle_summary.json   # per-vehicle participation counts + the reported model's
│                          #   accuracy / mAP / ADE / FDE on each car's own test split
└── best_model.pt    # best global checkpoint
```

The per-vehicle files are simulator-side bookkeeping: nothing in them crosses the simulated
V2I link, and recording them does not change training (runs reproduce bit-for-bit).

### Programmatic use

```python
from fedav.config import FLConfig
from fedav.sim.runner import run_experiment

cfg = FLConfig.from_yaml("configs/default.yaml")
result = run_experiment(cfg)
print(result.summary())
```

---

## 3. How it works

A federated round is the four-stage loop of PRD section 7:

```
        ┌──────────────── broadcast w(r) ────────────────┐
        │                                                ▼
   ┌────┴─────┐   local SGD on private data    ┌───────────────────┐
   │   RSU    │  (camera / LiDAR / radar)      │  AV k = 1 … |S|   │
   │ edge agg │                                │  w_k ← w_k − η∇F_k│
   └────▲─────┘                                └─────────┬─────────┘
        │                                                │  ΔW_k only
        │        TLS V2I: (ΔW_k, n_k, q_k, crc)          │  (no raw data)
        └────────────────────────────────────────────────┘
                     w(r+1) = Σ_k  (q_k n_k / Σ_j q_j n_j) · ΔW_k
```

1. **Select** — `ClientSelector` picks `S = ⌈C·N⌉` vehicles with stable links, stratified over
   regions, with a fairness guard so nobody starves.
2. **Train locally** — `ClientTrainer` runs `E_k` epochs of SGD on the vehicle's private shard.
3. **Transmit** — `TopKCompressor` → `UpdateSerializer` (fp16 + gzip + CRC32) → `SecureChannel`
   (TLS, drops, bandwidth accounting). Optional DP clipping + noise and secure-aggregation masks.
4. **Aggregate** — `FedAggregator` forms the quality-weighted mean, applies server momentum, and
   writes the new global model. Stop when `‖w(r+1) − w(r)‖ ≤ ε` or `r = R_max`.

Faults are handled rather than fatal: dropped uplinks, straggler timeouts, corrupted payloads and
Byzantine updates each remove a vehicle from the round; the best checkpoint is always kept.

---

## 4. The mathematics

> Every equation in this section is checked numerically against the shipped code in
> **[docs/MATHEMATICAL_VALIDATION.md](docs/MATHEMATICAL_VALIDATION.md)** — 14 figures, each
> produced by calling the implementation and comparing it to a closed form or a theoretical
> bound. Regenerate with `python scripts/make_proof_figures.py`.

### 4.1 Objectives (PRD 8.3)

Local objective on vehicle `k` with shard `D_k`, `n_k = |D_k|`:

```
F_k(w) = (1/n_k) Σ_{i ∈ D_k} L(w; x_i, y_i)
```

Global objective over the fleet, `n = Σ_k n_k`:

```
min_w  F(w) = Σ_{k=1..N} (n_k / n) · F_k(w)
```

Local update, `E` epochs of SGD with learning rate `η`:

```
w_k^{t+1} = w_k^t − η ∇F_k(w_k^t)
```

### 4.2 Quality-aware FedAvg (PRD 8.4)

Standard FedAvg uses size weights; this framework replaces them with quality-aware weights:

```
FedAvg     w^{t+1} = Σ_k (n_k / n) · w_k^{t+1}
Proposed   w^{t+1} = Σ_k α_k · w_k^{t+1},   α_k = q_k n_k / Σ_j q_j n_j
```

`α` is always a point on the simplex (`Σ α_k = 1`, `α_k ≥ 0`) — asserted in
`tests/test_math.py::test_aggregation_weights_sum_to_one`.

`q_k = similarity_k × quality_k ∈ [q_min, q_max]`, computed **on the vehicle**:

- `similarity_k = 1 − H(p_k, p_ref)` where `H` is the Hellinger distance
  `H(p,q) = ‖√p − √q‖₂ / √2 ∈ [0,1]` between the vehicle's label histogram and the fleet
  reference. (Jensen–Shannon distance is available as an alternative.) The reference histogram
  is the one piece of fleet-level statistics the scheme needs; it is far less revealing than raw
  data but not free, and in deployment should be collected under secure aggregation — see the
  caveat at the top of `fedav/data/quality.py`.
- `quality_k` is an onboard sensor-health score built from blur, occlusion, rain intensity and the
  calibration flag — never from labels.
- A similarity floor keeps a rare-but-clean region contributing; a vehicle is strongly damped only
  when it is *both* off-distribution *and* degraded (the "clamp `q_k`" mitigation of PRD 18).

Because clients send deltas, the server computes `Σ_k α_k ΔW_k` and sets `w^{t+1} = w^t + Δ`,
which is algebraically identical to averaging the weights whenever every participant started the
round from `w^t`.

**Server momentum** damps the oscillation caused by heterogeneous clients:

```
v^{t+1} = β v^t + Δ^t ,   applied = (1−β) v^{t+1} + β Δ^t ,   w^{t+1} = w^t + applied
```

The convex blend keeps the first-round step identical to plain FedAvg (regression-tested) while
later rounds inherit the smoothed direction.

**FedProx** (optional, PRD 8.8) adds `μ/2 ‖w_k − w‖²` to the local objective when drift is high.

**Byzantine robustness** (PRD 15/16): updates whose norm exceeds 3× the median are rejected, and
the remainder are combined with a coordinate-wise weighted trimmed mean.

### 4.3 Convergence criterion (PRD 8.9, FR-07)

```
stop when ‖w^{t+1} − w^t‖₂ ≤ ε   or   t ≥ R_max
```

### 4.4 Differential privacy (PRD 8.7)

1. **Clip**: `ΔW ← ΔW · min(1, C/‖ΔW‖₂)`, bounding one vehicle's sensitivity to `C`.
2. **Perturb**: add `N(0, σ²I)` with `σ = z·C`.
3. **Calibrate**: `z` is chosen by an **RDP accountant for the sub-sampled Gaussian mechanism**
   (Mironov et al.), because federated training *is* a sub-sampled Gaussian mechanism composed
   `R` times. For integer `α`:

   ```
   ε_RDP(α) = 1/(α−1) · log Σ_{k=0..α} C(α,k) (1−q)^{α−k} q^k exp(k(k−1)/2σ²)
   ```

   composed over rounds and converted to `(ε, δ)`-DP by minimising over `α`. A binary search finds
   the smallest `z` whose `R`-fold composition stays inside the budget. For a single release, the
   **analytic Gaussian mechanism** (Balle & Wang) is also provided — tighter than, and valid where
   the textbook `√(2 ln(1.25/δ))/ε` bound is not (`ε > 1`, and the PRD default is `ε = 8`).
4. Under secure aggregation each vehicle adds only `σ/√|S|`; the independent draws sum to exactly
   `N(0, σ²)` in the aggregate, so the fleet pays the noise once instead of `|S|` times.

### 4.5 Secure aggregation (PRD 8.7)

Every pair `(i, j)` of participants derives a shared seed and expands it to a mask `m_ij`; vehicle
`i` adds `+m_ij` when `j > i` and `−m_ij` when `j < i`. Therefore

```
Σ_k (α_k ΔW_k + mask_k) = Σ_k α_k ΔW_k
```

exactly, while any individual masked update is indistinguishable from noise. The runner **verifies
this identity every round** and logs `secure_aggregation_verified`.

> Secure aggregation and Byzantine-robust aggregation are mutually exclusive by construction: a
> trimmed mean needs per-client values, which secure aggregation is designed to hide. The runner
> takes the robust path when `aggregation.byzantine_robust` is on and says so in the logs.
> Reconciling the two (e.g. via verifiable secret sharing) is a roadmap item.

### 4.6 Communication cost (PRD 8.6)

```
C_total = Σ_{r=1..R} Σ_{k ∈ S} (C_up + C_down)
```

measured from real serialized byte counts, not estimates: TLS record overhead, handshake
amortisation, uplink per vehicle, downlink once per RSU multicast.

---

## 5. The ML model

`fedav/models/perception.py` — a MobileNetV3-style multi-task network, **700,485 parameters**
(PRD 8.1 caps it at ~6M; the cap is enforced at construction and unit-tested).

```
input 3×32×32
  └─ stem conv 3×3  (24 ch)
  └─ 3 × [inverted residual ↓2 + inverted residual]   (32, 64, 128 ch, SE, Hardswish)
  └─ 1×1 neck → 192 ch feature grid 4×4
       ├── centre heatmap 3×3 → softmax over the grid
       │     ├── soft-argmax  ⇒ (cx, cy)          ← localisation
       │     └── attention-pooled embedding       ← object-focused features
       ├── global-average-pooled embedding        ← scene context
       ├── cls head   → 6 classes
       ├── size head  → (w, h)
       └── traj head  → 6 future waypoints (x, y)
```

Two design decisions matter:

- **GroupNorm, never BatchNorm.** Batch statistics are client-specific under Non-IID data and
  averaging them across vehicles is a classic source of FedAvg divergence. Every tensor in the
  state dict is therefore safe to average — unit-tested.
- **Spatial soft-argmax for the box.** Global average pooling destroys the spatial information a
  box centre needs; regressing `(cx, cy)` from a pooled vector gave mAP@0.5 ≈ 0.02. The softmax
  over the feature grid is differentiable and its expectation is sub-cell accurate: in a
  like-for-like comparison during development (identical data, seed and schedule) it raised
  mAP@0.5 from ≈0.02 to ≈0.82.

Loss (PRD 8.1):

```
L = CE(class) + λ_box · [2·SmoothL1(centre) + SmoothL1(size)] + λ_traj · SmoothL1(trajectory)
```

Box and trajectory terms are masked on background samples — there is no object to localise, and
regressing one would inject noise into the shared representation.

---

## 6. Data & Non-IID partitioning

PRD 11.1 names KITTI, nuScenes and CARLA. KITTI/nuScenes cannot be redistributed and CARLA needs a
GPU simulator, so the repo ships a **CARLA-style procedural generator**
(`fedav/data/scenes.py`) that reproduces the properties the framework has to cope with, and
`fedav/data/adapters.py` documents the (small) surface you implement to swap in the real datasets:
four aligned arrays — `images [N,3,S,S]`, `labels [N]`, `boxes [N,4]`, `trajectories [N,2T]`.

Every cue the labels depend on is actually drawn into the image, so the task is genuinely learnable
rather than noise-fitting:

| Label | Visual cue |
|---|---|
| class | object shape, aspect ratio and (jittered, deliberately tight) colour |
| box | the rendered rectangle |
| trajectory | motion-streak length/direction (speed, heading) + lane-marking slant (yaw rate) |

Three skew axes are combined (PRD 11.2):

| Axis | Mechanism |
|---|---|
| **Label skew** | `p_k ~ Dirichlet(α·1)` over the class simplex (`α = 0.4` by default) |
| **Feature skew** | region ∈ {urban, highway, rural} × weather ∈ {clear, rain, fog, night} changes brightness, contrast, blur, noise, fog blend, rain streaks and occlusion |
| **Quantity skew** | `n_k` log-normal (`σ = 0.45`) around the configured mean |

A ~12% minority of vehicles are **de-calibrated**: extra blur, brightness drift and 25–45% label
corruption. These are exactly the vehicles `q_k` is meant to discount, so the quality-aware
weighting is being tested against a real effect, not a synthetic one.

Splits are 70 / 15 / 15 per vehicle (PRD 11.3), plus an **IID global test set** spanning every
region × weather combination so all arms are scored on the same unbiased distribution.

---

## 7. Communication & privacy

| Lever | Effect |
|---|---|
| deltas instead of weights | the wire carries `ΔW`, which sparsifies and compresses far better |
| float16 | 2× |
| gzip | a further 1.1–2× (and it crushes sparsified zeros nearly for free) |
| top-k + error feedback | the main lever — transmit the `k` largest coordinates, **keep the remainder in an on-vehicle residual** that is added back next round, so nothing is dropped, only delayed |
| RSU multicast | the global model is broadcast once into the cell instead of unicast per vehicle |
| bandwidth controller | closed loop that raises sparsity when a round overruns `comm.mb_budget_per_round` and relaxes it when there is headroom |

Privacy properties, by mode:

| Mode | Raw data egress | Individual update visible to RSU | Formal guarantee |
|---|---|---|---|
| default | none | yes (norm-filterable) | — |
| `dp.secure_aggregation` | none | **no** (sum only) | — |
| `dp.enabled` | none | no | `(ε, δ)`-DP, accounted per round |

---

## 8. Configuration reference

All defaults are PRD section 10.

| Group | Key | Default | Meaning |
|---|---|---|---|
| — | `num_clients` | 50 | `N`, fleet size |
| — | `seed` | 1337 | everything is seeded from here |
| `train` | `rounds` | 60 | `R_max` |
| `train` | `local_epochs` | 5 | `E` (per-vehicle `E_k` when adaptive) |
| `train` | `adaptive_epochs` | **false** | FR-10, scale `E_k` to onboard compute; off by default, measured net-negative (section 9) |
| `train` | `lr` | 0.01 | `η`, cosine-decayed |
| `train` | `momentum` | 0.9 | client SGD `β` |
| `train` | `batch_size` | 32 | `B` |
| `train` | `fedprox_mu` | 0.0 | `μ`, raise when drift is high |
| `train` | `tolerance` | 1e-3 | `ε` for `‖Δw‖` |
| `train` | `target_accuracy` | 0.90 | PRD 12.3 acceptance bar |
| `selection` | `fraction` | 0.3 | `C` |
| `selection` | `uptime_threshold` | 0.55 | minimum V2I link uptime |
| `selection` | `trajectory_aware` | true | stratify over regions |
| `selection` | `fairness_k` | 12 | every vehicle within `K` rounds |
| `selection` | `round_timeout_s` | 45 | straggler cut-off |
| `aggregation` | `quality_aware` | true | `q_k` weighting |
| `aggregation` | `normalize_local_steps` | true | FedNova; required whenever `E_k` varies |
| `aggregation` | `server_momentum` | 0.9 | `β` on the aggregate |
| `aggregation` | `byzantine_robust` | false | trimmed mean + norm filter |
| `aggregation` | `q_min` / `q_max` | 0.05 / 1.0 | clamp on `q_k` |
| `dp` | `enabled` | false | clip + Gaussian noise |
| `dp` | `clip_norm` | 1.0 | `C` |
| `dp` | `epsilon` / `delta` | 8.0 / 1e-5 | privacy budget |
| `dp` | `secure_aggregation` | true | masked sum |
| `comm` | `fp16` / `gzip_level` | true / 6 | wire codec |
| `comm` | `topk_ratio` | 0.25 | sparsified uplink |
| `comm` | `mb_budget_per_round` | 9.0 | NFR budget the controller defends |
| `data` | `dirichlet_alpha` | 0.4 | label skew |
| `data` | `quantity_skew` | 0.45 | log-normal `σ` on `n_k` |

---

## 9. Results

Measured on this repository — fleet `N = 50`, `C = 0.3`, `E = 5`, `R_max = 60`, Dirichlet `α = 0.4`, seed `1337`. All four arms share one fleet and one seed.

| Metric | Centralized | Local-only | Std. FedAvg | Proposed FL |
|---|---|---|---|---|
| Model accuracy, best round (%) | 89.89 | 25.19 | 83.00 | 84.67 |
| Model accuracy, last-10-round mean (%) | 88.21 | 24.70 | 80.09 | 82.94 |
| mAP@0.5 | 0.5443 | 0.0050 | 0.0892 | 0.1004 |
| Trajectory ADE | 0.1877 | 0.3234 | 0.1966 | 0.2017 |
| Rounds to target | — | — | — | — |
| Rounds run | 60 | 75 | 60 | 60 |
| Comm. cost (MB/round) | 0.000 | 0.000 | 74.561 | 8.722 |
| Total comm. (MB) | 90.2 | 0.0 | 4473.7 | 523.3 |
| Client acc. std (Non-IID) | 0.1264 | 0.0687 | 0.1304 | 0.1369 |
| Worst-client accuracy | 0.5091 | 0.1689 | 0.4634 | 0.3265 |
| Wall clock (s) | 1144 | 270 | 1957 | 3795 |

**Convergence curve** — first round reaching each accuracy (federated arms; lower is faster):

| Threshold | Centralized | Local-only | Std. FedAvg | Proposed FL |
|---|---|---|---|---|
| 60% | 3 | — | 18 | 16 |
| 70% | 4 | — | 21 | 24 |
| 75% | 5 | — | 28 | 28 |
| 80% | 7 | — | 28 | 28 |

### Acceptance (PRD 12.3): **FAIL**

| Check | Requirement | Observed | Result |
|---|---|---|---|
| `accuracy_target` | accuracy >= 90.00% | observed=0.8467 | **FAIL** |
| `convergence_speedup` | rounds-to-target cut by >= 25% vs standard FedAvg | proposed_rounds=None, fedavg_rounds=None, reduction=None | **FAIL** |
| `communication_saving` | MB/round cut by >= 30% vs standard FedAvg | proposed_mb=8.7221, fedavg_mb=74.561, reduction=0.883 | **PASS** |
| `communication_budget` | MB/round <= 9.0 | observed=8.7221 | **PASS** |
| `non_iid_robustness` | cross-client accuracy dispersion <= standard FedAvg | proposed_std=0.1369, fedavg_std=0.13038 | **FAIL** |
| `convergence_speedup_relative` | rounds to 95% of the centralized ceiling, cut by >= 25% | threshold_accuracy=0.8539, proposed_rounds=None, fedavg_rounds=None, reduction=None | **FAIL** |
| `accuracy_vs_ceiling` | proposed accuracy >= 95% of the centralized ceiling | centralized=0.8989, proposed=0.8467, ratio=0.9419 | **FAIL** |
| `privacy` | no raw sensor data egress in any federated arm | observed=True | **PASS** |

### Reading these results

**Collaboration is worth 59 accuracy points.** Vehicles training alone on their own Non-IID shards
reach 25.2%. Federated, the same fleet reaches **84.7%** against a centralized ceiling of 89.9% —
recovering 94% of what pooling all the data would give, while **no raw sensor data ever leaves a
vehicle**. That is the system's reason to exist, and it works.

**The proposed framework wins on accuracy and communication, and loses on the tail:**

| | Std. FedAvg | Proposed | |
|---|---|---|---|
| Accuracy, best round | 83.00% | **84.67%** | +1.7 pts |
| Accuracy, last-10 mean | 80.09% | **82.94%** | +2.9 pts |
| Rounds to 83% | 38 | **34** | 11% faster |
| Rounds to 75% / 80% | 28 / 28 | 28 / 28 | tie |
| Cross-client std (Non-IID) | **0.1304** | 0.1369 | **worse** |
| Worst-served vehicle | **46.3%** | 32.7% | **13.6 pts worse** |
| MB/round | 74.56 | **8.72** | **88% less** |

**Three PRD 12.3 criteria still fail. Stated plainly:**

1. **`accuracy_target` (≥ 90%) — a scale artefact, not an algorithm failure.** The centralized
   upper bound on this synthetic data is 89.9%, so *no* federated arm can clear an absolute 90%
   bar. The bar is calibrated to the PRD's data volume, not this generator's.
2. **`convergence_speedup` (≥ 25% fewer rounds) — not reproduced.** The two arms tie at the
   75% and 80% thresholds (28 rounds each); the proposed arm is 11% faster to 83%. Nothing
   like the 25–30% the PRD predicts.
3. **`non_iid_robustness` — fails.** Cross-client dispersion is 0.1369 against FedAvg's
   0.1304, and the worst-served vehicle is 13.6 points worse (32.7% vs 46.3%).
4. **`accuracy_vs_ceiling` (≥ 95% of centralized)** — 94.2%, just under.

`communication_saving`, `communication_budget` and `privacy` pass.

**The result that cuts against the framework, not buried.** The proposed arm raises the fleet
*average* while making the fleet *tail* worse: cross-client dispersion is higher than FedAvg's
(0.1369 vs 0.1304) and the worst-served vehicle drops 13.6 points (32.7% vs 46.3%). PRD 8.4
specifies quality-aware weighting as the Non-IID robustness mechanism, and at fleet level it
measurably is not — the isolated ablation shows it *does* cut dispersion by 31% on its own
([below](#what-the-ablation-found--including-one-prd-requirement-that-is-actively-harmful)), so
the regression comes from combining it with connectivity-aware selection, which by design sees
poorly-connected vehicles less often. For a safety-relevant system the tail is arguably what
matters more than the mean. `aggregation.q_min` and `selection.fairness_k` are the two levers,
and neither has been tuned against a tail objective here.

### What the ablation found — including one PRD requirement that is actively harmful

The headline numbers above come from a configuration that **deviates from the PRD in one place**,
on measured evidence. Each mechanism was run in isolation at `R = 30` on one fleet and seed, with
transport held constant (`results/ingredients/`):

| Ingredient, isolated | Accuracy | vs FedAvg control | Cross-client std |
|---|---|---|---|
| FedAvg control | 70.8% | — | 0.154 |
| Connectivity/trajectory selection (FR-04) | **79.0%** | **+8.2** | 0.124 |
| Server momentum (8.8) | 75.0% | +4.2 | 0.157 |
| Quality-aware weighting (8.4) | 72.8% | +2.0 | **0.106** (−31%) |
| **Adaptive local epochs (FR-10)** | 58.6% | **−12.2** | 0.193 |

Three of the four mechanisms help. Quality-aware weighting cuts cross-client dispersion by 31% —
precisely the Non-IID robustness it is specified to deliver, and the clearest confirmation of a
PRD hypothesis in this study.

**Adaptive local epochs is harmful.** The cause is *objective inconsistency*: scaling `E` to
onboard compute makes `τ_k` differ across vehicles, so a vehicle that ran 8 epochs contributes a
longer vector than one that ran 1, and a plain weighted mean over-weights fast ECUs regardless of
`α_k` — converging to the wrong point. The fix is FedNova-style normalised averaging,

```
Δ = (Σ_k α_k τ_k) · Σ_k α_k (ΔW_k / τ_k)
```

now implemented in `FedAggregator` and on by default (inert when step counts match). It recovers
**about half** the loss — adaptive epochs alone goes −12.2 → −6.2, and the combined algorithmic arm
−9.2 → −2.6 (`results/fix/`). Half, not all: heterogeneous `τ_k` costs something beyond objective
inconsistency at this scale.

So FR-10 — a *"Could"* priority in the PRD — remains fully implemented, configurable and tested,
but **ships off by default**, because a default that measurably costs 6 accuracy points is the
wrong default. `aggregation.normalize_local_steps` stays on, so whenever `E_k` does vary the sound
aggregation is used. This is the one place the implementation knowingly departs from the PRD, and
the evidence for it is committed.

**Caveats that bound all of the above.** One seed, one synthetic dataset, ~7,700 training samples.
Differences of a point or two sit inside run-to-run noise and should not be read as rankings; the
88% communication reduction, the +59-point collaboration gain and the −12.2-point adaptive-epochs
regression are large enough to be robust. mAP@0.5 is low for every federated arm (0.08–0.09 against
0.54 centralized): localisation is the first thing Non-IID averaging degrades, and it is the most
useful signal here for future work.

---

## 10. Testing

```bash
PYTHONPATH=. python -m pytest tests/ -q
```

Coverage maps onto PRD section 17:

| PRD test | Where |
|---|---|
| local SGD step | `test_local_sgd_step_decreases_the_loss` |
| aggregation math, `Σ α = 1` | `test_aggregation_weights_sum_to_one`, `test_fedavg_matches_closed_form`, `test_quality_aware_aggregation_matches_closed_form` |
| serializer round-trip | `test_serializer_round_trip_preserves_values`, `test_serializer_detects_corruption` |
| full federated round | `test_full_federated_round_updates_the_global_model` |
| ablation | `fedav/sim/ablation.py` (`python -m fedav.cli ablation`) |
| fixed-seed regression | `test_run_is_reproducible_under_a_fixed_seed` |
| scale test | `--set num_clients=100` |

Plus tests the PRD implies rather than lists: DP clipping bounds, `ε` composition inside budget,
secure-mask cancellation, error-feedback losslessness, fairness-guard starvation bounds,
connectivity relaxation under collapse, no-BatchNorm, the parameter cap, and an explicit
**"no raw data on the wire"** assertion for FR-02.

---

## 11. Project layout

```
fedav/
├── config.py                 # dataclass configs, YAML loading, fingerprinting
├── cli.py                    # run / benchmark / ablation / info
├── data/
│   ├── scenes.py             # procedural AV scene generator (class + box + trajectory)
│   ├── federated.py          # Dirichlet / feature / quantity skew, per-vehicle splits
│   ├── quality.py            # q_k: Hellinger similarity × sensor quality
│   └── adapters.py           # how to plug in KITTI / nuScenes / CARLA
├── models/perception.py      # MobileNetV3-style multi-task net, GroupNorm, soft-argmax head
├── client/
│   ├── trainer.py            # ClientTrainer: local SGD, FedProx, adaptive E_k, DP
│   ├── compression.py        # top-k + error feedback, bandwidth controller
│   └── serializer.py         # fp16 + gzip + CRC32 wire format
├── server/
│   ├── selection.py          # ClientSelector: connectivity, trajectory, fairness
│   ├── aggregator.py         # FedAggregator: quality-aware FedAvg, momentum, trimmed mean
│   └── secure.py             # pairwise masks + TLS V2I channel simulation
├── privacy/dp.py             # clipping, analytic Gaussian, RDP accountant
├── metrics/
│   ├── evaluation.py         # accuracy, mAP@0.5, ADE/FDE, per-client dispersion
│   ├── logger.py             # JSONL + CSV + summary, MLflow/W&B sink hook
│   └── fleet.py              # per-vehicle profiles, per-round records, final scorecard
├── sim/
│   ├── runner.py             # SimRunner: the PRD section 9 loop
│   ├── baselines.py          # centralized / local-only / standard FedAvg presets
│   ├── benchmark.py          # 4-arm comparison + PRD 12.3 acceptance check
│   └── ablation.py           # per-ingredient study
└── utils/                    # seeding, parameter-vector algebra

dashboard/
├── app.py                    # Streamlit dashboard: streamlit run dashboard/app.py
└── data.py                   # result loaders + run launcher, Streamlit-free and unit-tested
```

---

## 12. PRD traceability

| ID | Requirement | Implementation |
|---|---|---|
| FR-01 | local training for `E` epochs per round | `client/trainer.py::ClientTrainer.train` |
| FR-02 | transmit updates only, never raw data | `client/serializer.py`; asserted in `test_no_raw_data_in_the_transmitted_payload` |
| FR-03 | FedAvg with quality-aware weights | `server/aggregator.py`, `data/quality.py` |
| FR-04 | select connected, trajectory-viable clients | `server/selection.py::ClientSelector` |
| FR-05 | secure aggregation + DP noise | `server/secure.py`, `privacy/dp.py` |
| FR-06 | broadcast the new global model | `sim/runner.py::SimRunner._broadcast` |
| FR-07 | stop on `‖Δw‖ ≤ ε` or `R_max` | `server/aggregator.py::has_converged` |
| FR-08 | log per-round metrics | `metrics/logger.py` → `rounds.jsonl` / `rounds.csv` |
| FR-09 | compare against baselines | `sim/baselines.py`, `sim/benchmark.py` |
| FR-10 | scale `E` to onboard compute | `ClientTrainer.local_epochs_for` — implemented and tested, **default off**: measured −12.2 accuracy points (−6.2 with normalised averaging). The one knowing deviation from the PRD; evidence in `results/ingredients/` |
| 8.8 | objective consistency under varying `E_k` | `FedAggregator.combine` (FedNova normalisation) |
| 8.5 | fairness guard, min coverage per `K` rounds | `ClientSelector`, `coverage()` |
| 8.9 | convergence criterion | `has_converged` |
| 12.1 | accuracy, rounds, MB/round, robustness | `metrics/evaluation.py`, `RunResult.summary` |
| 12.3 | acceptance thresholds | `sim/benchmark.py::Benchmark.acceptance` |
| 15 | reproducibility, fault tolerance, security | seeding + fingerprint, drop/straggler/CRC handling, TLS + trimmed mean |
| 16 | edge cases | drop, straggler, poisoning, extreme skew, low connectivity, non-convergence |

---

## 13. Limitations & roadmap

Stated plainly, because a simulation that overstates itself is worse than no simulation:

1. **Data is synthetic.** The generator reproduces the *structure* of the problem (multi-task
   labels, three skew axes, a real quality axis) but not the appearance statistics of real
   sensors. Absolute accuracies are therefore not comparable to published KITTI/nuScenes numbers;
   the *relative* ordering of the arms is what this repo measures. `data/adapters.py` is the
   swap-in point.
2. **Absolute MB/round depends on model size and fleet size**, neither of which the PRD's
   comparison table pins down. The benchmark reports the measured *reduction versus standard
   FedAvg* (the PRD's ~30% criterion) and separately checks the `≤ 9 MB/round` budget for the
   shipped configuration.
3. **V2I is simulated**, not measured against a real radio stack — bandwidth, latency and drops are
   parametric (PRD 18 lists this risk).
4. **Secure aggregation has no dropout-recovery protocol** (no Shamir shares); masks are re-derived
   over the set that actually reported. It also cannot be combined with trimmed-mean robustness.
5. **One knowing deviation from the PRD.** FR-10 (adaptive local epochs) ships disabled because
   it measurably hurts — see section 9. It is implemented, configurable and tested, so enabling it
   is a one-line config change; the default reflects the evidence, not the spec.
6. **Averaged robustness is not tail robustness.** The proposed arm raises mean accuracy but
   widens cross-client dispersion (0.1369 vs 0.1304) and leaves the worst-served vehicle 13.6
   points worse off than standard FedAvg. For a safety-relevant
   system that trade may be unacceptable; `aggregation.q_min` bounds it.
7. **Simulation only.** `v1` is explicitly not an on-vehicle deployment (PRD 3.2).
8. **Roadmap** (PRD 20): certified DP budgets, blockchain contribution auditing, 5G/V2X
   latency-aware scheduling, verifiable secret sharing to combine secure and robust aggregation.

---

## 14. References

1. *Recent Advances in Federated Learning for Connected Autonomous Vehicles: Addressing Privacy,
   Performance, and Scalability Challenges*, IEEE, 2025.
2. *Federated Learning for Connected and Automated Vehicles: A Survey of Existing Approaches and
   Challenges*, IEEE Transactions on Intelligent Vehicles, 2023.
3. H. B. McMahan, E. Moore, D. Ramage, S. Hampson, B. A. y Arcas,
   *Communication-Efficient Learning of Deep Networks from Decentralized Data*, AISTATS 2017.
4. T. Li et al., *Federated Optimization in Heterogeneous Networks* (FedProx), MLSys 2020.
5. I. Mironov, K. Talwar, L. Zhang, *Rényi Differential Privacy of the Sampled Gaussian Mechanism*,
   2019.
6. B. Balle, Y.-X. Wang, *Improving the Gaussian Mechanism for Differential Privacy: Analytical
   Calibration and Optimal Denoising*, ICML 2018.
7. K. Bonawitz et al., *Practical Secure Aggregation for Privacy-Preserving Machine Learning*,
   CCS 2017.
8. D. Yin et al., *Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates*,
   ICML 2018.
9. S. U. Stich et al., *Sparsified SGD with Memory* (error feedback), NeurIPS 2018.
