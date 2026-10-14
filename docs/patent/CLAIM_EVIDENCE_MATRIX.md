# Claim → code → test → evidence

Every claim element in [CLAIMS_DRAFT.md](CLAIMS_DRAFT.md), where it is implemented,
the test that pins its behaviour, and the measured result an examiner can be
shown. `scripts/check_claim_matrix.py` (run by `tests/test_claim_matrix.py`)
fails if any referenced file, function or test disappears, so the matrix cannot
silently drift from the code.

| Claim | Element | Code | Test | Evidence |
|---|---|---|---|---|
| 1(a), 7 | vehicle selection: uptime gate, regional stratification, starvation guard before the gate | `fedav/server/selection.py::ClientSelector.select` | `tests/test_patent_mechanisms.py::test_fairness_guard_reaches_weak_link_vehicles` | `results/patent/effects.md` (vehicles reporting per round) |
| 1(b) | rate prediction from position (mean 3GPP path loss, no shadowing) | `fedav/server/v2x.py::V2XRadioModel.predicted_rate_mbps` | `tests/test_v2x.py::test_umi_los_path_loss_matches_tr38901_formula` | — |
| 1(b), 5 | airtime-equalising prefix-length assignment; geometric ladder | `fedav/server/secagg.py::assign_tiers`, `fedav/server/secagg.py::tier_ladder` | `tests/test_secagg.py::test_tiers_keep_each_vehicle_inside_its_airtime_share`, `tests/test_patent_mechanisms.py::test_airtime_budget_tiers_bound_round_airtime` | `results/patent/effects.md` (uplink airtime per round, worst round) |
| 5 | closed-loop adjustment of the top prefix length on measured MB | `fedav/client/compression.py::BandwidthController.update` | `tests/test_pipeline.py` | `results/patent/runs/*/seed_*/rounds.csv` (`topk_ratio`) |
| 1(c), 1(d), 2(e), 9 | ordered shared support from the previous global update; seeded permutation in round 1 | `fedav/server/secagg.py::shared_support` | `tests/test_secagg.py::test_support_is_ordered_by_previous_update_magnitude` | — |
| 1(d), 2(f), 2(g) | fixed-point ring encoding with per-segment weight and weight-step scalars | `fedav/server/secagg.py::FixedPoint`, `fedav/server/secagg.py::vehicle_encode` | `tests/test_secagg.py::test_clipping_excess_returns_to_residual` | `results/patent/runs/proposed_*/seed_*/rounds.jsonl` (`secagg_clip_frac`) |
| 1(e), 1(f) | ring summation and per-segment FedNova ratio; RSU sees only sums | `fedav/server/secagg.py::rsu_aggregate`, `fedav/server/secagg.py::combine_segments` | `tests/test_secagg.py::test_recovers_the_plaintext_weighted_fednova_mean`, `tests/test_secagg.py::test_rsu_learns_sums_not_individual_weights` | `rounds.jsonl` `secagg_rel_error` (≤ 1e-3 every round) |
| 1(g) | global update with server momentum | `fedav/server/aggregator.py::FedAggregator.apply` | `tests/test_pipeline.py` | — |
| 2(h) | Diffie-Hellman pair seeds (RFC 3526 group 14), masks only over common coordinates | `fedav/server/secagg.py::pair_seed`, `fedav/server/secagg.py::keypair` | `tests/test_secagg.py::test_pair_seeds_agree_and_need_a_private_key` | — |
| 2(i) | masked message with CRC-32 | `fedav/server/secagg.py::parse_upload` | `tests/test_secagg.py::test_corrupted_payload_is_detected` | FIG. 7 |
| 2(j) | NACK-driven residual retention | `fedav/sim/runner.py::SimRunner._secure_round`, `fedav/client/compression.py::TopKCompressor.restore` | `tests/test_patent_mechanisms.py::test_nack_keeps_the_failed_upload_in_the_residual` | `results/patent/effects.md` (K1+K2 vs FedAvg + same transport) |
| 4 | dropout recovery by seed reveal | `fedav/server/secagg.py::rsu_aggregate` | `tests/test_secagg.py::test_recovers_the_plaintext_weighted_fednova_mean`, `tests/test_patent_mechanisms.py::test_secure_round_matches_plaintext_even_with_failed_uploads` | `rounds.jsonl` `secagg_seeds_revealed` |
| 6 | weight cap and clipping bound from the previous aggregate | `fedav/sim/runner.py::SimRunner._weight`, `fedav/sim/runner.py::SimRunner._secure_round` | `tests/test_patent_mechanisms.py::test_secure_round_matches_plaintext_even_with_failed_uploads` | `rounds.jsonl` `secagg_value_bound` |
| 2(c), 8 | quality score: sensor health × (1 − Hellinger distance) | `fedav/data/quality.py::compute_quality_weight`, `fedav/data/quality.py::hellinger_distance` | `tests/test_math.py` | `docs/figures/10_quality_metric.png` |
| 10 | update clipping and Gaussian noise; RDP accounting | `fedav/privacy/dp.py::clip_state`, `fedav/privacy/dp.py::add_gaussian_noise` | `tests/test_math.py` | `docs/figures/04_dp_epsilon_composition.png` |
| 3 | V2I radio link cost: airtime and energy per transmission | `fedav/server/v2x.py::V2XRadioModel.transmit` | `tests/test_v2x.py::test_airtime_and_energy_scale_linearly_with_bytes` | `results/patent/effects.md` (airtime, energy) |
