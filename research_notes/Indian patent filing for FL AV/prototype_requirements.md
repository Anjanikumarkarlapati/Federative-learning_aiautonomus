# Prototype and evidence requirements for an Indian patent application on the FL-AV framework

*Research notes, 14 Oct 2026. Not legal advice. Several primary sources (ipindia.gov.in, intepat.com, bananaip.com) did not resolve from this environment ("getaddrinfo ENOTFOUND"), so the Indian legal points below come from search-result summaries of law-firm and official pages; the final CRI Guidelines 2025 PDF and the Manual of Patent Office Practice and Procedure (MPPP) text were not read. Repository facts come from reading the repository directly (commit `2cc7004`).*

## Q1. What do Indian practice and practitioners recommend as prototype / experimental evidence for AI / CRI applications? Is a working model required?

### Takeaway
No working model or physical prototype is required to file in India; what is required is a specification that enables the invention and discloses the best method (s.10(4)), and, for a computer-related invention, a stated and preferably measured technical effect to survive s.3(k). The Controller can demand a model, sample or experimental data later (s.10(3)), and refusals have followed when none was produced, so the evidence should exist before filing and its key numbers should be in the specification.

### Cited Findings
- A prototype is not a legal requirement for filing; "descriptive completeness rather than physical construction" is what matters; a provisional specification need only describe the invention, while the complete specification must describe it fully including the best method — [Intepat, When to file](https://www.intepat.com/blog/what-is-the-right-stage-for-filing-a-patent); [Maheshwari & Co](https://www.maheshwariandco.com/blog/provisional-vs-complete-patent/) (search summaries).
- Section 10(3) lets the Controller require a model or sample "if the application needs supplementing", before the application is found in order for grant; this is typically raised when the examiner doubts the invention works — [Mondaq, Models and samples in Indian patent applications](https://webiis08.mondaq.com/india/patent/1136566/explaining-inventions-better-models-and-samples-in-indian-patent-applications).
- Example: in IPA 1451/MUM/2013 the Controller asked for experimental data or a working model; none was produced and the application was refused — same [Mondaq article](https://webiis08.mondaq.com/india/patent/1136566/explaining-inventions-better-models-and-samples-in-indian-patent-applications).
- Leaving "significant unknowns" in how the invention works or is reproduced creates an insufficiency risk if you file too early — [Intepat, When to file](https://www.intepat.com/blog/what-is-the-right-stage-for-filing-a-patent).
- CRI Guidelines 2025 were finalised 29 July 2025 (drafts March and June 2025) with a dedicated AI/ML/DL chapter — [IIPRD](https://www.iiprd.com/tag/clinical-research-innovation-2025/); [EU IP Helpdesk](https://intellectual-property-helpdesk.ec.europa.eu/news-events/news/indian-patent-office-guidance-examining-ai-inventions-cri-guidelines-2025-2025-08-05_en); [IAM](https://www.iam-media.com/article/indian-patent-office-releases-highly-anticipated-examination-guidelines-computer-related-inventions).
- Practitioner reading of the guidelines' AI examples: the specification is expected to disclose **model architecture, training-dataset characteristics, the pre-processing pipeline, training parameters, and validation results that demonstrate the claimed technical effect**; a specification saying what the model does without how is "unlikely to clear examination on the sufficiency ground" — search summary of [Intepat AI/IP article](https://www.intepat.com/blog/intellectual-property/artificial-intelligence-ipr/) (secondary).
- Commentary on the *draft* says AI applications should disclose training data sources, model architecture (e.g. layers) and learning algorithms, and notes the guidelines do not say whether high-level functional disclosure suffices — [Conventus Law](https://conventuslaw.com/report/india-draft-guidelines-for-examination-of-computer-related-inventions-2025/); [Khurana & Khurana](https://old.khuranaandkhurana.com/?p=15628). BananaIP argued that dataset *characteristics* (type, distribution, relevance) should be accepted instead of full datasets — [BananaIP comments](https://www.bananaip.com/intellepedia/?p=128941). Whether the final text kept the draft's data-source wording is unverified.
- *Caleb Suresh Motupalli v. Controller* (Madras HC, 29 Jan 2025, CMA (PT) No. 2 of 2024, 2025:MHC:293): refusal upheld; the specification had no working examples showing how its "black-box modernization" achieved the result; s.3(k), s.10(4) enablement and s.10(5) clarity were all in play — [LiveLaw](https://www.livelaw.in/amp/tech-law/madras-high-court-upholds-rejection-of-ai-human-integration-patent-claim-283918); [BananaIP](https://bananaip.com/?p=115259); [Online Law Connect](https://onlinelawconnect.com/casecommentory/highcourt/Caleb-Suresh-Motupalli-v-Controller-of-Patents-CMA-(PT)-No-2-of-2024-2025-MHC-293.php).
- Indian examiners "invariably" insist that the specification discloses a technical effect, technical application and technical result; applicants in AI and software typically answer with comparative experimental data or benchmarks (e.g. measurable speed-up or memory reduction versus existing methods); one commentary says the effect "must be measurable and tangible" — [SS Rana, technical effect](https://ssrana.in/articles/the-significance-of-technical-effect-in-determining-the-patentability-of-computer-related-inventions-cris/); [Mondaq / Conventus, 3Ts of CRIs](https://www.mondaq.com/india/patent/1730010/3ts-of-computer-related-inventions-and-beyond-analysis-of-section-3k-of-the-patents-act-1970); [Intepat software/AI patents](https://www.intepat.com/ip-services/patents/software-and-ai-patents).
- Guidance-level examples of acceptable technical effects in the 2025 guidelines (secondary): higher speed, reduced memory access time, better control, enhanced security, **reduced latency**; novel hardware is not a prerequisite; if any claim form falls in the exclusion the subject matter fails — summarised in prior notes `research_notes/Patent ready research paper 2026/ai_patentability.md` citing [Intepat](https://www.intepat.com/blog/cri-guidelines-2025-patent-india-3k) and [Fox Mandal](https://foxmandal.in/patenting-ai-in-india-patent-office-revises-its-cri-guidelines/). Sources disagree on whether a *measurable* effect is demanded.
- Case law anchors: *Ferid Allani* (Delhi HC 2019) and *Microsoft Technology Licensing* (technical effect / technical contribution improving system functionality); *Malikie/BlackBerry* (Delhi HC 2024) lost after FER reply because the contribution was software instructions without technical effect — [SS Rana](https://ssrana.in/articles/demystifying-section-3k-of-the-indian-patents-act/); [IP India-hosted Microsoft order (Madras IPD, 2024)](https://ipindia.gov.in/storage/uploads/media/12.%20Microsoft%20Technology%20Licensing%20LLC%20vs%20Assistant%20Controller%20of%20Patents%20and%20Designs%2C%20(T)%20CMA%20(PT)%20No.49%20of%202023%2C%2003072024.pdf); [Mondaq 3Ts](https://www.mondaq.com/india/patent/1730010/3ts-of-computer-related-inventions-and-beyond-analysis-of-section-3k-of-the-patents-act-1970).
- Recommended FER-response structure for 3(k): technical problem, technical solution, technical effect, inventive step, focused on the actual technical contribution — [Intepat software/AI patents](https://www.intepat.com/ip-services/patents/software-and-ai-patents); [taxaj](https://www.taxaj.com/learn/?p=942).

### Inferences
- For this project the "prototype" an Indian examiner can use is: (a) a working example in the specification with architecture, data description, hyper-parameters and a results table; (b) comparative numbers against a conventional baseline (standard FedAvg) for a *technical* quantity (uplink bytes, airtime/latency, energy, robustness to faulty sensors), and (c) the ability to produce more data on request (the repository and its reproducible runs). The simulation is acceptable as a working example if it is described as one, but the effect must be shown on quantities a network engineer would recognise as technical, not just classification accuracy.
- Because the guidelines reportedly ask for training-data characteristics rather than the data itself, the synthetic generator should be described by its parameters (classes, image size, Dirichlet α = 0.4, region × weather skew, ~12% de-calibrated vehicles with 25-45% label corruption) and, if possible, supplemented with one public real dataset run.
- Evidence must be generated before the complete specification is filed: post-filing data can support an argument in an FER reply, but cannot add matter that is not in the specification (general principle; not verified for India in this session).

### Gaps
- Final CRI Guidelines 2025 text (AI chapter, sufficiency paragraph) and MPPP sections on s.10(4) "best method" and on drawings were not read (domain unreachable). Whether post-filed experimental data is admissible in an Indian FER reply, and with what weight, was not found.
- No Indian decision specific to federated learning or V2X ML was found.

## Q2. Which technical-effect metrics matter most for this invention, and what baselines, ablations and statistical rigour are expected?

### Takeaway
The strongest 3(k) argument is a network/system effect: uplink bytes per round, V2I airtime / round latency at a given link rate, energy per vehicle, and robustness of the aggregate under faulty-sensor and dropped-link conditions; accuracy should be shown only as "not degraded". Each must be compared with standard FedAvg under the same seeds, with multiple seeds and confidence intervals; the repository currently has one seed for every comparison.

### Cited Findings
- Comparative benchmarks showing measurable improvement in speed or memory versus existing methods are the typical evidence in AI/software prosecutions — [SS Rana](https://ssrana.in/articles/the-significance-of-technical-effect-in-determining-the-patentability-of-computer-related-inventions-cris/).
- "Reduced latency" and "enhanced security" are listed examples of technical effect in the CRI Guidelines 2025 (secondary) — [Intepat](https://www.intepat.com/blog/cri-guidelines-2025-patent-india-3k) via prior notes.
- Reproducibility guidance: ML models should be benchmarked with multiple random seeds so variance can be reported, and seeds published — [Reproducibility in ML-based research, arXiv 2406.14325](https://arxiv.org/pdf/2406.14325).
- FL papers commonly use 3 to 9 seeds and report 95% bootstrap CIs over seeds; a two-sided Wilcoxon signed-rank test cannot reach p < 0.05 with n = 5 seeds — [arXiv 2609.23463](https://arxiv.org/pdf/2609.23463) (search summary).
- Repository facts (read directly): headline benchmark at one seed (1337), N = 50, C = 0.3, E = 5, R = 60 (`results/benchmark/benchmark_report.md`): Proposed 82.17% vs FedAvg 81.50% best-round accuracy; 8.709 vs 74.564 MB/round (−88.3%); worst vehicle 40.8% vs 32.7%; client std 0.1328 vs 0.1379; acceptance 4/8 pass. README §9 itself says the accuracy and tail differences are "within noise" and that a previous run had the worst vehicle 13.6 points *below* FedAvg.
- Repository facts: the ablation (`results/ingredients/`, R = 30, one seed) attributes +8.2 points to connectivity/trajectory selection, −31% cross-client std to quality weighting, −12.2 points to adaptive epochs; README notes these predate two fixes and "should be re-measured".
- Repository facts: `SecureChannel._transfer` in `fedav/server/secure.py` computes a simulated airtime (`base_latency_s` 35 ms + bytes·8 / 12 Mbit/s uplink), but `fedav/sim/runner.py` line 204 discards it (`delivered, _, _ = self.channel.send_up(...)`), so **no V2I latency/airtime is logged**. `round_seconds` in `rounds.csv` is host CPU wall-time (~58 s/round proposed, 3,497 s total; FedAvg 21,750 s total) and is not a valid latency metric.
- Repository facts: no energy measurement anywhere in `fedav/` (grep for energy/joule/RAPL found nothing).
- CodeCarbon reads CPU energy through Intel RAPL on Linux (accumulated microjoules), falls back to a TDP-based estimate when RAPL is unavailable, and RAPL usually needs read permission on `/sys/class/powercap` — [CodeCarbon docs, RAPL](https://docs.codecarbon.io/3.2/how-to/enable-rapl/); [CodeCarbon power estimation](https://docs.codecarbon.io/latest/explanation/power-estimation/).

### Inferences
- Metric priority for the specification and an FER reply (value to 3(k) argument, then cost):
  1. **Uplink bytes per round and to a target accuracy** (already measured from serialized bytes; extend to "bytes to reach X% accuracy" and per-vehicle bytes; add multi-seed CIs). Strongest, cheapest.
  2. **Simulated V2I airtime and round completion time** (log the seconds `send_up` already returns; report per-round max over selected vehicles = round latency, and the fraction of rounds finishing within `round_timeout_s` = 45 s). Cheap; maps directly to "reduced latency".
  3. **Robustness to faulty sensors / poisoned updates**: accuracy and worst-vehicle accuracy as the fraction of de-calibrated vehicles rises (0, 12, 25, 40%), proposed vs FedAvg; and Byzantine norm filter on/off with injected scaled updates (`docs/figures/09_byzantine_robustness.png` exists as a math check, not an end-to-end sweep). Medium cost; it is the most AV-specific effect.
  4. **Robustness to link loss**: accuracy vs link uptime distribution / drop rate, proposed selection vs random selection. Medium cost.
  5. **Per-vehicle compute/energy**: on-vehicle training seconds per round (already partly recorded as `train_seconds`) and energy via CodeCarbon or a joules = power × time estimate; plus radio energy = transmit power × airtime. Cheap if estimated, uncertain if measured in a container (RAPL may be unavailable).
  6. **Privacy**: (ε, δ) at DP settings with accuracy cost (configs/dp.yaml exists; no result committed), and the exact-cancellation residual of secure aggregation (already in `docs/figures/06_secure_aggregation_exactness.png`).
  7. **Convergence**: rounds to 80%/81% (34/37 vs 38/51 at one seed) — only claim if multi-seed CIs hold.
- Baselines needed: standard FedAvg (exists), FedAvg with the *same* transport (fp16 + top-k + error feedback) to separate the algorithmic effect from compression (the `results/attribution/` runs partly do this at one seed), and FedProx (implemented via `fedprox_mu`, never benchmarked). Ablation of each claimed feature on/off, re-run after the fixes.
- Statistical rigour that is feasible on CPU: 5 seeds × {FedAvg, Proposed} at a reduced setting (e.g. N = 50, R = 30, E = 2), mean ± 95% CI and paired differences; keep the full 60-round run at one seed as the "best mode" illustration. Note that at full scale FedAvg took ~6 h host time per seed in the committed run, so full-scale multi-seed is not an "hours" task.

### Gaps
- No Indian source states a minimum number of seeds or a statistical test; the multi-seed recommendation is general ML practice.
- Why the FedAvg arm took ~6× the host time of the proposed arm was not investigated; it should not be presented as a speed effect without a controlled measurement.

## Q3. Would a real or public AV dataset, or a realistic V2X channel model / network simulator, materially strengthen the evidence? What is cheaply feasible?

### Takeaway
Yes, moderately: one run on real driving images and a standards-based V2I path-loss model would blunt the "synthetic toy" objection and make the latency claim technical. Cheapest options are a 3GPP TR 37.885 path-loss + Shannon-rate model inside `secure.py` (pure Python, under an hour) and a small real-image classification shard (IDD crops, GTSRB, or nuScenes-mini), noting licence limits; ns-3 / Veins / SUMO co-simulation is not feasible in hours.

### Cited Findings
- 3GPP TR 37.885 Table 6.2.1-1 (V2X): highway LOS PL = 32.4 + 20 log10(d3D) + 20 log10(fc) dB, σ_SF = 3 dB; urban LOS PL = 38.77 + 16.7 log10(d3D) + 18.2 log10(fc), σ = 3 dB; urban NLOS PL = 36.85 + 30 log10(d3D) + 18.9 log10(fc), σ = 4 dB; fc in GHz, d in m; NLOSv adds vehicle-blockage loss — [iTecSpec transcription of TR 37.885 §6.2.1](https://www.itecspec.com/3gpp/37.885/s/6.2.1); implemented in [ns-3 ThreeGppV2vPropagationLossModel](https://www.nsnam.org/docs/release/3.42/doxygen/d8/de1/three-gpp-v2v-propagation-loss-model_8cc_source.html). A 2026 arXiv paper uses the highway LOS form at 5.9 GHz with d clamped ≥ 1 m and 3 dB shadowing — [arXiv 2605.25431](https://arxiv.org/pdf/2605.25431). (Coefficients should be checked against the official 3GPP PDF.)
- India Driving Dataset (IIIT Hyderabad, Intel, UCSD, 2018): IDD Detection ~22.8 GB, ~46.6k-47k images, 40 classes; licence terms not found in search — [INSAAN / IIIT](https://insaan.iiit.ac.in/); [Dataset Ninja](https://datasetninja.com/idd-detection); [IDD-3D paper](https://arxiv.org/pdf/2210.12878).
- nuScenes is reported as CC BY-NC-SA 4.0 with account sign-up; nuScenes-mini is 10 scenes, 404 samples, ~4.17 GB — [arXiv 2203.16258](https://arxiv.org/pdf/2203.16258); [nuscenes-devkit](https://simple-repository.app.cern.ch/project/nuscenes-devkit). KITTI is reported (second-hand) as CC BY-NC-SA 3.0 — [HF KITTI-QA card](https://huggingface.co/datasets/heesang1/KITTI-QA).
- GTSRB: 43 traffic-sign classes, 39,209 train / 12,630 test images, available as `torchvision.datasets.GTSRB`; Armory lists it as CC0 (another listing says "Other") — [Armory dataset licensing](https://armory.readthedocs.io/en/latest/dataset_licensing/); [torchvision gtsrb.py mirror](https://dagshub.com/jinensetpal/vision/src/main/torchvision/datasets/gtsrb.py); [hyper.ai](https://hyper.ai/en/datasets/16402).
- Repository facts: `fedav/data/adapters.py` already defines `from_arrays(images, labels, boxes, trajectories)` and written notes for KITTI / nuScenes / CARLA; nothing is wired up. README §6 and §13 state the data is synthetic and V2I is "parametric". Channel is fixed 12 Mbit/s up / 40 Mbit/s down, 35 ms base latency, Bernoulli drop on `link_uptime` (`fedav/server/secure.py`).
- Repository facts: the model input is 3×32×32 and the full-scale proposed arm takes ~58 s/round on 4 CPU cores, so real images must be cropped/resized to 32×32 to stay within CPU hours.

### Inferences
- **Channel model (high value, low effort):** replace the fixed rate with per-vehicle distance to RSU → TR 37.885 path loss (5.9 GHz, highway/urban by region) + log-normal shadowing → SNR → Shannon rate with a 10/20 MHz channel and an efficiency factor → airtime; drop when SNR below threshold. This turns the 88% byte saving into a seconds/round and a "rounds completing within deadline" number, which is the "reduced latency" technical effect named in the CRI guidelines. Cite it as a 3GPP-standard model, which reads as technical to an examiner.
- **Real data (medium value, medium effort):** IDD is the most persuasive for an Indian filing (Indian roads, unstructured traffic) but is 22.8 GB and its licence is unconfirmed; a practical route is object crops from a subset of IDD Detection (classification + box, trajectory head off with λ_traj = 0), partitioned by city/sequence for natural non-IID. GTSRB is the cheapest real-image fallback (small, CC0 per Armory, torchvision loader) but is traffic-sign classification, not scene perception. nuScenes-mini is small but non-commercial licensed and only 404 samples. Non-commercial dataset licences may matter if the patent is to be commercialised; using data for experiments described in a specification is a question for counsel.
- **Network simulators:** ns-3 (with its 3GPP V2V model) or Veins/SUMO would be more realistic but are a multi-day integration; mention as the next embodiment rather than a pre-filing requirement. SUMO-generated mobility traces could drive the distance-to-RSU variable later.

### Gaps
- IDD licence/terms of use not confirmed; GTSRB licence sources conflict (CC0 vs "Other").
- Whether an Indian examiner gives extra weight to real vs synthetic data was not found in any source; this is an inference.

## Q4. What figures should the specification include, and in what style under Indian rules?

### Takeaway
Rule 15 requires A4 sheets with ≥ 4 cm top/left and ≥ 3 cm bottom/right margins, no dimensions, sequential numbering, and the applicant's name (and sheet count, signature) on each sheet; practice is black-and-white line drawings with reference numerals that the description cites. For this invention: system architecture, vehicle on-board unit and RSU hardware block diagrams, one flowchart per independent claim (vehicle method, RSU method), a round sequence diagram, the update-packet format, the quality-score computation, the selection logic, and B/W results graphs.

### Cited Findings
- Patents Rules 2003, Rule 15: drawings on standard A4 sheets with clear margin of at least 4 cm top and left and 3 cm bottom and right; scale large enough to show the invention; dimensions not marked on drawings; sheets sequentially or systematically numbered and bearing the applicant's name in the top-left corner; at least one copy prepared neatly on durable paper — [IP India Rule 15](https://ipindia.gov.in/writereaddata/portal/ev/rules/pr15.html); [IP India Rule 15 (new site)](https://www.ipindia.gov.in/acts/patent-rules-2003/rule-15) (search summary; page could not be opened).
- Drawings supplied under s.10(2) are treated as part of the specification unless the Controller directs otherwise — [Indian Kanoon, s.10](https://future.indiankanoon.org/doc/1418267).
- Each element should carry a reference numeral explained in the description (Indian practitioner commentary, not rule text) — [Kanakkupillai, patent drawing](https://www.kanakkupillai.com/patent-drawing); [iPleaders](https://blog.ipleaders.in/patent-drawing-importance/). Black-ink / solid-black-line rule is US 37 CFR 1.84, not found in the Indian text — [37 CFR 1.84](https://chanrobles.com/usa/uslaws/cfr/title37/37-1.0.1.1.1.2.74.42.php).
- Method patents commonly include flowcharts of the method steps — [Brown & Michaels](https://bpmlegal.com/howtopat3.html).
- Prior project notes recommend drawings: system block diagram, server and vehicle flowcharts, round sequence diagram, privacy data-flow, model architecture, results — `background/PATENT_READINESS.md` §4.

### Inferences
- Recommended figure set (FIG numbering as it would appear):
  - FIG. 1 System overview: vehicles 100a-n, sensors 102, OBU 104, V2I link 110, RSU 120 with aggregator 122, selector 124, model store 126, optional cloud 130.
  - FIG. 2 Vehicle on-board unit hardware block diagram: sensor interface, processor/accelerator, memory holding private shard and residual buffer, sensor-health monitor, quality-score module, compressor/serializer, V2X modem (PC5/Uu), secure element for mask keys. **This is the hardware embodiment the repo lacks** and is what moves claims away from "computer programme per se".
  - FIG. 3 RSU hardware block diagram: V2X transceiver, multicast scheduler, link-quality estimator, aggregation processor, storage, backhaul.
  - FIG. 4 Vehicle-side method flowchart (claim 1 family): receive global model → compute sensor-health term → compute label-histogram Hellinger similarity → q_k with floor/clamp → local training → delta → top-k + error feedback → fp16/gzip/CRC → mask → transmit.
  - FIG. 5 RSU-side method flowchart (claim 2 family): link/uptime filter with geometric relaxation → region stratification → fairness promotion → broadcast → collect with deadline → CRC/norm filter → α_k = q_k n_k / Σ q_j n_j → FedNova-normalised aggregate → server momentum → convergence test → multicast.
  - FIG. 6 Message sequence diagram of one round (vehicles, RSU, timers, drop/straggler paths).
  - FIG. 7 Update packet format: header (vehicle id, round, n_k, q_k, τ_k, sparsity k), index/value payload, CRC32, TLS framing; byte sizes from `fedav/client/serializer.py`.
  - FIG. 8 Quality-score computation block diagram; FIG. 9 neural-network architecture (from README §5).
  - FIG. 10-14 Results graphs in B/W (line styles/markers, not colour): uplink MB vs rounds; accuracy vs cumulative MB; round latency CDF under 3GPP channel; worst-vehicle accuracy vs fraction of faulty sensors; ablation bars with CI whiskers.
- Existing `docs/figures/*.png` are colour matplotlib plots made for validation; they are useful source data but need redrawing in B/W with numerals; the ASCII diagram in README §3 is the seed for FIG. 1/6.

### Gaps
- MPPP / Rule 15 text on colour drawings, photographs, and whether graphs count as drawings was not read; confirm with an agent. Exact wording of the remaining sub-rules of Rule 15 (sheet count, signature) not retrieved.

## Q5. Repository audit against the recommended outputs, and a prioritised CPU-feasible build list

### Takeaway
The repository is strong on reproducible code, exact byte accounting, mathematical verification and an honest single-seed benchmark; it is missing every patent-specific artefact (drawings, hardware embodiment, claim-to-code map, disclosure record) and the statistical and system-level evidence (multi-seed CIs, latency/airtime, energy, faulty-sensor sweep, real data, standards channel model) that a 3(k) technical-effect argument needs.

### Cited Findings
- Inventorship and dated records: everywhere that has ruled requires a natural-person inventor; documentation should show which human conceived each claim limitation; keep dated records — prior notes `research_notes/Patent ready research paper 2026/ai_patentability.md` citing [EU IP Helpdesk on Indian DABUS decision (27 Jul 2026)](https://intellectual-property-helpdesk.ec.europa.eu/news-events/news/indias-patent-office-rejects-ai-inventor-landmark-dabus-decision-2026-07-27_en) and [Holland & Knight](https://www.hklaw.com/en/insights/publications/2026/02/the-human-element-uspto-clarifies-inventorship).
- Lab notebook practice: a notebook is prime evidence of who conceived and when; entries should be dated and witnessed promptly by a non-inventor; computer files can be altered at will, so where no bound notebook exists a signed invention-disclosure form is the best available evidence — [FindLaw](https://corporate.findlaw.com/intellectual-property/ignore-u-s-research-record-keeping-principles-at-the-peril-of.html); [Oregon State](https://sites.science.oregonstate.edu/~gablek/proper_laboratory_notebook.htm); [Patent Baristas](https://patentbaristas.com/?p=343). (US-focused; no WIPO page found.)
- `background/PATENT_READINESS.md` (repo) records that every commit is authored by "Claude", the PRD the design follows is not in the repository, and no LICENSE file exists.

### Audit table (repository facts, read directly)

| # | Recommended output | Status | Where / what is missing |
|---|---|---|---|
| A1 | Working, reproducible implementation | **Exists** | `fedav/`, CLI `python -m fedav.cli run/benchmark/ablation`, `config.json` fingerprint, `tests/test_pipeline.py::test_run_is_reproducible_under_a_fixed_seed` |
| A2 | Best-mode configuration disclosed | **Exists** | `configs/default.yaml`, `results/benchmark/proposed_fl/config.json`, README §8 |
| A3 | Model architecture, loss, hyper-parameters (sufficiency) | **Exists** | README §5, `fedav/models/perception.py` (700,485 params) |
| A4 | Training-data characteristics | **Exists (synthetic)** | README §6, `fedav/data/scenes.py`, `federated.py` |
| A5 | Baseline comparison table | **Partial** | `results/benchmark/benchmark_report.md` — one seed; no FedProx; transport-matched FedAvg only in `results/attribution/` (one seed, R=30) |
| A6 | Ablation of each claimed feature | **Partial** | `results/ingredients/`, `results/fix/` — one seed, predates two fixes |
| A7 | Multiple seeds, mean ± CI, paired test | **Missing** | single seed 1337 everywhere; no `--seeds` sweep in `fedav/cli.py` or `scripts/run_experiments.py` |
| A8 | Uplink bytes measured | **Exists** | `mb_round`, `mb_cumulative` in `rounds.csv`; `docs/figures/08_communication_stack.png` |
| A9 | V2I airtime / round latency | **Missing (computed then discarded)** | `secure.py::_transfer` returns seconds; `runner.py:204` drops them; no deadline-hit metric |
| A10 | Standards-based channel model (3GPP / ns-3) | **Missing** | fixed 12/40 Mbit/s, 35 ms, Bernoulli drops |
| A11 | Energy per vehicle (compute + radio) | **Missing** | only host `round_seconds`, `train_seconds` |
| A12 | Faulty-sensor robustness sweep | **Partial** | de-calibration fixed at ~12%; per-vehicle data in `vehicle_summary.json`; no sweep over fault rate |
| A13 | Byzantine / poisoning robustness end-to-end | **Partial** | `docs/figures/09_byzantine_robustness.png` (unit-level); no end-to-end run committed |
| A14 | DP privacy–utility result | **Partial** | `configs/dp.yaml`, accountant figures 04/05; no committed DP run result |
| A15 | Secure aggregation exactness | **Exists** | `docs/figures/06_secure_aggregation_exactness.png`, `residuals.json` |
| A16 | Real / public AV dataset run | **Missing** | `fedav/data/adapters.py` has notes only |
| A17 | Hardware embodiment (OBU, RSU, V2X modem, secure element) | **Missing** | no hardware description anywhere; README §13 "simulation only" |
| A18 | Patent drawings (B/W, numerals, Rule 15 format) | **Missing** | only colour validation plots in `docs/figures/` and ASCII art in README §3 |
| A19 | Message / packet format specification | **Partial** | in code (`fedav/client/serializer.py`) but not as a documented field layout |
| A20 | Claim-to-code / claim-to-evidence traceability | **Missing** | README §12 maps PRD requirements to code, not claims |
| A21 | Invention-disclosure record, dated conception log, named human inventor | **Missing** | git history (all commits authored "Claude") is the only dated record; PRD absent |
| A22 | Demo / dashboard for examiner hearing | **Exists** | `dashboard/app.py` (Streamlit, 7 pages) |

### Prioritised build list (value to s.10(4) / s.3(k) ÷ CPU effort)

| Rank | Output | Supports | Effort (CPU, 4 cores) | Notes |
|---|---|---|---|---|
| 1 | **Invention disclosure record** (`docs/patent/INVENTION_DISCLOSURE.md`): human inventor(s), dated conception of each candidate feature, problem → solution → effect, prior art known; signed/dated PDF witnessed by a non-inventor | inventorship, priority | 1-2 h, no compute | Highest value; cannot be generated later |
| 2 | **Log V2I airtime and round latency**; replace fixed rate with TR 37.885 path loss + shadowing + Shannon rate (5.9 GHz) per region; report seconds/round, deadline-hit rate | 3(k) "reduced latency" | ~1-2 h code + reuse runs | Pure Python in `secure.py`/`runner.py`; config switch |
| 3 | **Multi-seed comparison**: 5 seeds × {FedAvg, transport-matched FedAvg, Proposed} at N=50, R=30, E=2; mean ± 95% CI, paired differences for MB/round, airtime, accuracy, worst-vehicle | technical effect credibility | ~5-10 h background (est. from ~58 s/round at E=5; FedAvg slower) | Add `--seeds` loop + aggregation script; quick config for smoke |
| 4 | **Patent drawings FIG. 1-9** as B/W SVG/PDF with reference numerals (matplotlib/graphviz or hand SVG), plus reference-numeral list | sufficiency, drawing requirement | 3-4 h | Includes OBU/RSU hardware diagrams |
| 5 | **Hardware embodiment section** (`docs/patent/EMBODIMENTS.md`): OBU, RSU, V2X modem (PC5/Uu), secure element, timing budget; map each module to code | 3(k) (not program per se), s.10(4) | 2 h writing | |
| 6 | **Claim-to-code-to-evidence matrix** | prosecution, FER replies | 1-2 h | each claim element → file/function → test → result file |
| 7 | **Faulty-sensor sweep**: de-calibrated fraction 0/12/25/40%, proposed vs FedAvg, 3 seeds, reduced R | AV-specific technical effect (robustness) | ~4-8 h background | uses existing knobs in data generator |
| 8 | **B/W results graphs** FIG. 10-14 from rank 2/3/7 outputs, with CI bands | technical effect | 1-2 h | line styles, no colour |
| 9 | **Packet format table** (field, bytes, encoding) from `serializer.py` | sufficiency | 1 h | |
| 10 | **Energy estimate**: CodeCarbon/RAPL if readable, else per-vehicle compute J = P_TDP × train_seconds and radio J = P_tx × airtime | 3(k) energy | 1-2 h | label as estimate unless RAPL works |
| 11 | **Re-run ablation** after fixes, 3 seeds | best method, inventive step | ~4-6 h background | |
| 12 | **Real-data run**: GTSRB (cheap) or an IDD-Detection crop subset via `adapters.from_arrays`, λ_traj = 0, proposed vs FedAvg, 1-3 seeds | rebuts "synthetic only" | 3-6 h incl. download; IDD 22.8 GB is heavy | check licences |
| 13 | DP run (ε = 8) and end-to-end Byzantine run committed with results | privacy/security effect | 2-4 h background | |
| 14 | ns-3 / Veins / SUMO co-simulation | realism | days | defer; describe as alternative embodiment |

### Inferences
- Ranks 1, 2, 4, 5 and 6 need almost no compute and close the biggest gaps an Indian examiner would probe (program-per-se, black-box sufficiency, missing hardware). Ranks 3 and 7 are the runs that turn the 88% byte saving and the tail/robustness story into statistically defensible technical effects; they fit "hours" only at reduced rounds/epochs, run in background.
- The accuracy gain (+0.7 pt) should not be claimed as the technical effect; the communication saving, latency under a standard channel, and robustness to degraded sensors are the defensible effects. Claims should be drafted as vehicle-side and RSU-side single-actor methods plus a system, each reciting hardware (OBU, V2X transceiver, RSU processor).

### Gaps
- Run-time estimates for ranks 3, 7, 11 are extrapolated from the committed `round_seconds`; not benchmarked at the reduced configuration.
- Whether a git history authored by an AI agent is acceptable as a conception record in India is untested; treat the separate signed disclosure as necessary.
