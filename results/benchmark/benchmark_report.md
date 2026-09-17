# FL-AV benchmark report

Config fingerprint: `47d140ca4677`  
Fleet: N=50, C=0.3, E=5, R_max=60, Dirichlet alpha=0.4

## Results

| Metric | Centralized | Local-only | Std. FedAvg | Proposed FL |
|---|---|---|---|---|
| Model accuracy, best round (%) | 89.89 | 25.19 | 83.00 | 84.67 |
| Model accuracy, last-10-round mean (%) | 88.21 | 24.70 | 80.09 | 82.94 |
| mAP@0.5 | 0.5443 | 0.0050 | 0.0892 | 0.1004 |
| Trajectory ADE | 0.1877 | 0.3234 | 0.1966 | 0.2017 |
| Rounds to target (90%) | — | — | — | — |
| Rounds to 95% of ceiling | 11 | — | — | — |
| Rounds run | 60 | 75 | 60 | 60 |
| Comm. cost (MB/round) | 0.000 | 0.000 | 74.561 | 8.722 |
| Total comm. (MB) | 90.2 | 0.0 | 4473.7 | 523.3 |
| Client acc. std (Non-IID) | 0.1264 | 0.0687 | 0.1304 | 0.1369 |
| Worst-client accuracy | 0.5091 | 0.1689 | 0.4634 | 0.3265 |
| Raw data egress | yes | no | no | no |

## Acceptance (PRD 12.3)

**FAIL**

- `accuracy_target` — accuracy >= 90.00% → **FAIL** ({"observed": 0.8467})
- `convergence_speedup` — rounds-to-target cut by >= 25% vs standard FedAvg → **FAIL** ({"proposed_rounds": null, "fedavg_rounds": null, "reduction": null})
- `communication_saving` — MB/round cut by >= 30% vs standard FedAvg → **PASS** ({"proposed_mb": 8.7221, "fedavg_mb": 74.561, "reduction": 0.883})
- `communication_budget` — MB/round <= 9.0 → **PASS** ({"observed": 8.7221})
- `non_iid_robustness` — cross-client accuracy dispersion <= standard FedAvg → **FAIL** ({"proposed_std": 0.1369, "fedavg_std": 0.13038})
- `convergence_speedup_relative` — rounds to 95% of the centralized ceiling, cut by >= 25% → **FAIL** ({"threshold_accuracy": 0.8539, "proposed_rounds": null, "fedavg_rounds": null, "reduction": null})
- `accuracy_vs_ceiling` — proposed accuracy >= 95% of the centralized ceiling → **FAIL** ({"centralized": 0.8989, "proposed": 0.8467, "ratio": 0.9419})
- `privacy` — no raw sensor data egress in any federated arm → **PASS** ({"observed": true})
