# FL-AV benchmark report

Config fingerprint: `47d140ca4677`  
Fleet: N=50, C=0.3, E=5, R_max=60, Dirichlet alpha=0.4

## Results

| Metric | Centralized | Local-only | Std. FedAvg | Proposed FL |
|---|---|---|---|---|
| Model accuracy, best round (%) | 89.89 | 25.19 | 81.50 | 82.17 |
| Model accuracy, last-10-round mean (%) | 88.21 | 24.70 | 79.53 | 81.33 |
| mAP@0.5 | 0.5443 | 0.0050 | 0.0892 | 0.0960 |
| Trajectory ADE | 0.1877 | 0.3234 | 0.2062 | 0.1983 |
| Rounds to target (90%) | — | — | — | — |
| Rounds to 95% of ceiling | 11 | — | — | — |
| Rounds run | 60 | 75 | 60 | 60 |
| Comm. cost (MB/round) | 0.000 | 0.000 | 74.564 | 8.709 |
| Total comm. (MB) | 90.2 | 0.0 | 4473.8 | 522.6 |
| Client acc. std (Non-IID) | 0.1264 | 0.0687 | 0.1379 | 0.1328 |
| Worst-client accuracy | 0.5091 | 0.1689 | 0.3265 | 0.4082 |
| Raw data egress | yes | no | no | no |

## Acceptance (PRD 12.3)

**FAIL**

- `accuracy_target` — accuracy >= 90.00% → **FAIL** ({"observed": 0.8217})
- `convergence_speedup` — rounds-to-target cut by >= 25% vs standard FedAvg → **FAIL** ({"proposed_rounds": null, "fedavg_rounds": null, "reduction": null})
- `communication_saving` — MB/round cut by >= 30% vs standard FedAvg → **PASS** ({"proposed_mb": 8.7094, "fedavg_mb": 74.564, "reduction": 0.8832})
- `communication_budget` — MB/round <= 9.0 → **PASS** ({"observed": 8.7094})
- `non_iid_robustness` — cross-client accuracy dispersion <= standard FedAvg → **PASS** ({"proposed_std": 0.13285, "fedavg_std": 0.13794})
- `convergence_speedup_relative` — rounds to 95% of the centralized ceiling, cut by >= 25% → **FAIL** ({"threshold_accuracy": 0.8539, "proposed_rounds": null, "fedavg_rounds": null, "reduction": null})
- `accuracy_vs_ceiling` — proposed accuracy >= 95% of the centralized ceiling → **FAIL** ({"centralized": 0.8989, "proposed": 0.8217, "ratio": 0.9141})
- `privacy` — no raw sensor data egress in any federated arm → **PASS** ({"observed": true})
