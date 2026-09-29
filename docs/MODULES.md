# Module guide

What each part of the repository is for. The project simulates **federated learning for
autonomous vehicles (AVs)**: a fleet trains one shared model, and only model updates
(`ΔW`) — never raw sensor data — travel to a Road-Side Unit (RSU) that aggregates them.

## 1. Prototypes (start here)

| Folder | Purpose |
|---|---|
| `python_prototype/` | Small numpy-only demo of the core idea. Compares centralized training, plain FedAvg and the proposed method (connectivity-aware selection + quality-aware aggregation) on synthetic non-IID vehicle data. Good for understanding the algorithm in ~150 lines. |
| `matlab_prototype/` | The same demo in base MATLAB, for MATLAB users and quick experiments. Produces the same CSV output and an accuracy plot. |

## 2. Full framework — `fedav/`

| Module | Purpose |
|---|---|
| `config.py` | Typed configuration for the whole experiment (data, model, training, selection, aggregation, privacy, communication). Loaded from `configs/*.yaml`. |
| `cli.py` | Command-line entry point: `info`, `run`, `benchmark`, `ablation`. |
| **`client/`** | Everything that runs *on the vehicle*. |
| `client/trainer.py` | Local SGD on the vehicle's private data; returns the weight delta, sample count and quality inputs. |
| `client/compression.py` | Top-k sparsification with error feedback, plus a controller that keeps upload size inside the MB-per-round budget. |
| `client/serializer.py` | Wire format for updates (fp16 + gzip + CRC32 integrity check). |
| **`server/`** | Everything that runs *on the RSU*. |
| `server/selection.py` | Picks which vehicles participate each round, favouring stable V2I links and regional coverage, with a fairness guard. |
| `server/aggregator.py` | Quality-weighted FedAvg, server momentum, optional Byzantine-robust trimmed mean, and the convergence stop test. |
| `server/secure.py` | Secure aggregation (pairwise masks so the RSU sees only a sum) and a simulated encrypted V2I channel with drops, latency and bandwidth accounting. |
| **`data/`** | Synthetic data and fleet construction. |
| `data/scenes.py` | Procedural driving-scene generator (image, object box, future trajectory). |
| `data/federated.py` | Builds the fleet with non-IID splits: Dirichlet label skew, log-normal quantity skew, per-vehicle region/weather shift. |
| `data/quality.py` | Computes each vehicle's quality score `q_k` and the aggregation weights `q_k·n_k / Σ q_j·n_j`. |
| `data/adapters.py` | Hook for plugging in real AV datasets instead of synthetic ones. |
| **`models/`** | |
| `models/perception.py` | Lightweight multi-task network (classification + box + trajectory, ≤ ~6M parameters) and its loss. |
| **`privacy/`** | |
| `privacy/dp.py` | Differential privacy: update clipping, Gaussian noise, and an accountant that tracks the `(ε, δ)` budget spent. |
| **`metrics/`** | |
| `metrics/evaluation.py` | Accuracy, IoU, average precision, and per-vehicle robustness across non-IID shards. |
| `metrics/logger.py` | Writes per-round telemetry to JSONL/CSV. |
| **`sim/`** | Experiment orchestration. |
| `sim/runner.py` | Runs one federated experiment end to end (select → train → transmit → aggregate → evaluate). |
| `sim/baselines.py` | The comparison arms: centralized (upper bound), local-only (lower bound), standard FedAvg, proposed. |
| `sim/benchmark.py` | Runs all arms and checks the acceptance criteria. |
| `sim/ablation.py` | Measures which ingredient (quality weighting, selection, momentum, …) contributes what. |
| **`utils/`** | |
| `utils/seeding.py` | Deterministic seeding for reproducible runs. |
| `utils/tensors.py` | Parameter-vector helpers (flatten, add, scale, norm, weighted average, trimmed mean) shared by client, server and privacy code. |

## 3. Supporting directories

| Path | Purpose |
|---|---|
| `configs/` | Ready-made experiment configs: `quick.yaml` (smoke run), `default.yaml` (full experiment), `dp.yaml` (privacy on). |
| `scripts/` | `run_experiments.py` (full suite), `make_proof_figures.py` and `make_validation_doc.py` (regenerate the math-validation document), `update_readme_results.py` (paste results into the README). |
| `tests/` | Unit tests for the mathematics and an end-to-end pipeline test. |
| `results/`, `docs/` | Committed benchmark evidence, the mathematical-validation document and its figures. |
