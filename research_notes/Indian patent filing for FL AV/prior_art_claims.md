# Prior-art map: federated learning for connected and autonomous vehicles (for an Indian patent application on the fedav framework)

*Working notes, not a legal search opinion. **Method limit:** patents.google.com, patents.justia.com, USPTO ppubs/Gazette and IP India pages could not be opened from this environment (DNS failure or proxy 403). Everything about patents below comes from web-search result snippets and Google Patents / USPTO titles shown in those results. **No claim text was read directly.** Where a snippet quoted claim language, that is said explicitly. A registered patent agent must pull full claims and legal status (Google Patents, Espacenet, Patentscope, IP India InPASS; CPC G06N 3/098, G06N 20/00, H04W 4/40, H04W 4/44, H04L 9/008, G08G 1/0967) before any claim is drafted.*

Features of the project referred to below (from README.md and `fedav/`): (a) quality-aware aggregation weight q_k = sensor quality x calibration x label-distribution (Hellinger) similarity, floored (`fedav/data/quality.py`, `fedav/server/aggregator.py`); (b) RSU client selection with link-uptime threshold, regional stratification and a fairness/starvation guard (`fedav/server/selection.py`); (c) FedNova normalised averaging + server momentum; (d) top-k + error feedback, adaptive sparsity under a per-round MB budget, fp16+gzip+CRC payloads, multicast delta broadcast (`fedav/client/compression.py`, `serializer.py`); (e) DP clipping/noise with RDP accounting and pairwise-mask secure aggregation (`fedav/privacy/dp.py`, `fedav/server/secure.py`); (f) per-vehicle telemetry and fault handling (dropped links, stragglers, corrupt payloads, Byzantine trimmed mean).

## Q1. Closest patents (US, EP, CN, IN, WO) per feature

### Takeaway
Vehicular FL itself is well patented, most visibly by Toyota (three or more US grants), Qualcomm (sidelink FL) and MERL (communication/computation-aware vehicular FL). The closest single patent to this project is **Toyota US 12,258,045 B2**, which uses sensor-quality metadata to choose which vehicles train. No patent found in this search combines sensor-quality-and-label-similarity aggregation weights with link-uptime selection, budgeted sparsification and secure aggregation. Because claims were not read, absence of evidence is weak.

### Cited Findings

**Vehicular FL with vehicle selection / sensor metadata (features a, b)**
- **US 12,258,045 B2**, "Systems and methods for vehicular-network-assisted federated machine learning", Toyota Motor Engineering & Manufacturing North America; appl. 17/168,406 filed 5 Feb 2021; granted 25 Mar 2025; inventors Ucar, Higuchi, Wang, Krijestorac, Altintas; 16 claims. Pre-grant publication US 2022/0250656 A1. — [Google Patents](https://patents.google.com/patent/US12258045); [USPTO Gazette](https://patentsgazette.uspto.gov/week12/OG/html/1532-4/US12258045-20250325.html); [US20220250656A1](https://patents.google.com/patent/US20220250656A1/en)
  - Per search snippets: connected vehicles in a vehicular micro cloud exchange **metadata about their sensors**; based on it, vehicles are **elected to join a training phase**; they train locally and submit for aggregation. A snippet quoted a dependent claim: "wherein the metadata includes information regarding at least one of availability of particular types of sensors, capabilities of specific available sensors, quality of the specific available sensors". — [USPTO PDF listing](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12258045)
  - One search summary said the description also gives different vehicles' models **different influence in aggregation according to sensor-data quality**, and steers vehicles to locations giving better training data. That summary was not confirmed against the claims. — [US20220250656A1](https://patents.google.com/patent/US20220250656A1/en)
- **US 12,061,904 B2** (pub. US 2023/0205520 A1, appl. 18/068,094), "Information processing method, information processing apparatus and server apparatus", Toyota Motor Corp; inventors Lei Zhong, Jing Ma, Xiao Shao, Ryokichi Onishi. Drawings describe FL between a server and in-vehicle apparatuses and in-vehicle apparatuses "autonomously and distributedly selecting local models". — [Google Patents](https://patents.google.com/patent/US12061904)
- **US 12,337,867 B2** / US 2024/0174254 A1, "Systems and methods for communication-aware federated learning" (Toyota family per Justia inventor listings that include Krijestorac; assignee not confirmed). Content not retrieved; title indicates communication-aware selection/aggregation and it must be read as a primary reference against feature (b). — [Google Patents](https://patents.google.com/patent/US12337867); [Justia](https://patents.justia.com/patent/20240174254)
- **US 2024/0265296 A1**, "Systems and methods for user-edge association based on vehicle heterogeneity for reducing the heterogeneity in hierarchical federated learning networks": vehicles associated to RSU/edge servers for hierarchical FL. Assignee not shown in results. — [Google Patents](https://patents.google.com/patent/US20240265296A1)
- Other Toyota-area publications visible in results (titles only): US 2024/0104427 "FL with heterogeneous clients via data-free knowledge distillation"; US 2024/0256892 "FL training for a neural network associated with autonomous vehicles"; US 2024/0127105 A1 "contribution-aware federated learning" (vision-based lane centering). — [Justia 20240104427](https://patents.justia.com/patent/20240104427); [Justia 20240256892](https://patents.justia.com/patent/20240256892); [US20240127105A1](https://patents.google.com/patent/US20240127105A1/en)
- **US 12,369,186 B2** / **EP 4374590**, "Communication and computation aware distributed machine learning for vehicular networks", Mitsubishi Electric Research Laboratories (MERL), appl. 17/652,145; inventors Guo, Pervej, Kim, Parsons, Orlik, Di Cairano, Menner, Berntorp. Learning servers select global models, RSUs relay, servers distribute models to **selected on-road vehicles** and aggregate. Whether selection uses link duration/route was not confirmed. — [Google Patents](https://patents.google.com/patent/US12369186); [EPO](https://data.epo.org/publication-server/rest/v1.2/patents/EP4374590NWA1/document.html)
- **US 2019/0215378 A1**, "Predicting vehicle dwell time to optimize V2I communication": ML predicts how long a vehicle will stay on a future road segment from driving history and road features. Not FL, but it is a ready secondary reference for "predict link uptime from trajectory". — [Google Patents](https://patents.google.com/patent/US20190215378A1/en)

**Qualcomm sidelink FL (features b, d context)**
- **US 12,333,391 B2**, "Sidelink-supported federated learning for training a machine learning component", Qualcomm, filed Jan 2021; claims a first UE receiving a request for local-update assistance. — [Google Patents](https://patents.google.com/patent/US12333391); [Gazette](https://patentsgazette.uspto.gov/week24/OG/html/1535-3/US12333391-20250617.html)
- **US 12,376,142 B2**, "Methods for enhanced sidelink communications with clustered or peer-to-peer federated learning" (Qualcomm per search summary): UE sends model update plus a measurement, receives a cluster-aggregated update, then signals a sidelink parameter in SCI output by the model. — [Justia](https://patents.justia.com/patent/12376142)

**Data-quality-weighted aggregation, non-vehicle (feature a)**
- **US 2022/0414464 A1**, "Method and server for federated machine learning": computes a **data quality index per data source** and uses it to **weight training updates** when building the global model; weights may be computed on the source or on the server. Assignee not shown. — [Google Patents](https://patents.google.com/patent/US20220414464A1/en)
- Earlier memo (same project) recorded healthcare filings: US 12,230,012 (client selection under inconsistent image quality) and US 11,934,555 (privacy-preserving data curation for FL); claims not read. — `research_notes/Patent ready research paper 2026/ai_patentability.md`

**Compression / communication-efficient FL (feature d)**
- Google family **US 10,657,461** "Communication efficient federated learning" (McMahan, Bacon, Konecny, Yu), continuations US 11,763,197 and later grant, plus EP counterpart: structured and sketched updates (sub-sampling, quantisation, random rotations). — recorded in `ai_patentability.md` of this project
- A USPTO grant titled "Gradient accumulation for federated learning" (US 12,035,286) addresses information loss from local-update compression; assignee not shown. — [USPTO PDF](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12035286)
- No patent was found that claims **top-k with error feedback in a vehicular FL system under a per-round RSU byte budget**; searches returned only papers. — [search results incl. Sahu et al. NeurIPS 2021](https://mcanini.github.io/papers/rethink-gs.neurips21.pdf)

**CN filings (features a, e)**
- **CN 120546993 B**, "A federated learning data sharing method for Internet of Vehicles" (Qilu Institute of Technology, filed 2025 per one summary): weights local models by **parameter similarity to the global model** and adds **Laplace noise**. — [Google Patents](https://patents.google.com/patent/CN120546993B/en)
- US national phase of PCT/CN2022/110197 (filed 4 Aug 2022), multi-agent federated reinforcement learning vehicle-road cooperative control at intersections (US 11,862,016). — [USPTO PDF](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11862016)

**Indian filings**
- No Indian application on FL for vehicles was found. Indian FL filings surfaced were generic: 202321072013 (FL system with data privacy and security, Patent Office Journal 47/2023) and 202023107184 (listed by MMCOE). — [MMCOE list](https://mmcoe.edu.in/wp-content/uploads/2026/01/2023-24-Published-Patents.pdf)
- A vendor (PatSnap) report says Huawei has a comprehensive FL framework for AV ecosystems; marketing source, no numbers. — [PatSnap](https://eureka.patsnap.com/report-federated-learning-in-autonomous-vehicle-data-collaboration)

### Inferences
- Toyota (US 12,258,045; US 12,337,867; US 12,061,904; US 2022/0250656) is the assignee to beat. Any claim on "selecting vehicles for FL based on sensor quality" is very likely anticipated; a claim on "weighting aggregation by sensor quality" may be anticipated by the 12,258,045 description even if not claimed (description is prior art for novelty).
- US 2022/0414464 A1 makes a generic "data-quality-index weighted aggregation" claim obvious; the project's distinguishing content must be the specific composition (label-free sensor health x calibration residual x Hellinger similarity of label histogram to a fleet reference, floored), its on-vehicle computation, and its interaction with normalised averaging and secure aggregation.
- No Ford, GM, Honda, Hyundai, Bosch, Denso, Nvidia, Baidu, Tata Elxsi or KPIT vehicular-FL patent surfaced; that is a search-coverage gap, not evidence of absence.

### Gaps
- Claim text, priority, family and legal status of every patent above (direct pages blocked).
- Assignee-filtered searches for Ford, GM, Honda, Hyundai, Samsung, LG, Intel, Huawei, Bosch, Denso, Nvidia, Baidu, Tata Elxsi, KPIT, IITs/IISc: not completed; the general web search does not index these well.
- CNIPA full-text search (Baidu, Huawei, Chinese universities file heavily on 车联网 联邦学习): not done.
- IP India InPASS search: not done (portal not reachable here).

## Q2. Key academic prior art per feature

### Takeaway
Every individual technique in (a)-(f) is published, most of it 2016-2021; vehicular papers 2020-2026 already combine pairs such as mobility-aware selection with resource allocation, or quality weighting with RSU aggregation. Novelty must come from a specific combination and its measured technical effect.

### Cited Findings
**Foundations (standard citations; links are canonical arXiv/ePrint pages, not re-fetched this session)**
- FedAvg: McMahan et al. 2017, arXiv 1602.05629 — [arXiv](https://arxiv.org/abs/1602.05629)
- Konecny et al. 2016, structured/sketched updates — [arXiv](https://arxiv.org/abs/1610.05492)
- FedProx: Li et al. 2020 — [arXiv](https://arxiv.org/abs/1812.06127)
- FedNova: Wang et al. NeurIPS 2020 (normalised averaging for heterogeneous local steps) — [arXiv](https://arxiv.org/abs/2007.07481)
- Server momentum (FedAvgM): Hsu, Qi, Brown 2019 — [arXiv](https://arxiv.org/abs/1909.06335); adaptive server optimisers (FedOpt): Reddi et al. 2021 — [arXiv](https://arxiv.org/abs/2003.00295)
- Secure aggregation with pairwise masks and dropout recovery: Bonawitz et al. CCS 2017 — [IACR ePrint 2017/281](https://eprint.iacr.org/2017/281)
- DP-FedAvg (user-level DP, clipping, weighted estimator with capped weights): McMahan et al. ICLR 2018 — [arXiv](https://arxiv.org/abs/1710.06963); Rényi DP accounting: Mironov 2017 — [arXiv](https://arxiv.org/abs/1702.07476)
- Deep Gradient Compression (top-k + local accumulation/momentum correction): Lin et al. ICLR 2018 — [arXiv](https://arxiv.org/abs/1712.01887); error feedback theory: Karimireddy et al. 2019 — [arXiv](https://arxiv.org/abs/1901.09847)
- Top-k is communication-optimal per iteration for fixed budget; adaptive schemes target a whole-training budget: Sahu et al. NeurIPS 2021 — [PDF](https://mcanini.github.io/papers/rethink-gs.neurips21.pdf)
- Byzantine-robust coordinate-wise trimmed mean/median: Yin et al. ICML 2018 — [arXiv](https://arxiv.org/abs/1803.01498)
- Communication-efficient FL survey (top-k + EF keeps accuracy at 99-99.9 % sparsity) — [arXiv 2208.01200](https://arxiv.org/pdf/2208.01200)

**Vehicular FL surveys**
- FL for connected and automated vehicles surveys — [arXiv 2303.10677](https://arxiv.org/pdf/2303.10677); [arXiv 2308.10407](https://arxiv.org/pdf/2308.10407)
- Privacy-preserving ML in IoV (2025) — [arXiv 2503.01089](https://arxiv.org/pdf/2503.01089)
- FL in vehicular edge computing (selects "fine" local models by evaluating image quality and computation capability) — [IEEE Access AAM](https://ieeexplore.ieee.org/ielaam/6287639/8948470/8964354-aam.pdf)

**Mobility / connectivity-aware selection (feature b)**
- C-V2X Mode 4 vehicle selection: estimates each vehicle's **stay time in RSU range**; RSU selects using reported status — [arXiv 2401.07224](https://arxiv.org/pdf/2401.07224)
- Selection rule T_stay > T_train + T_trans with stay time from position and speed, and joint vehicle selection under intermittent participation (vehicles leave RSU coverage before upload) — [ResearchGate](https://www.researchgate.net/publication/376783354_Vehicle_Selection_and_Resource_Allocation_for_Federated_Learning-Assisted_Vehicular_Network)
- FedSSC: joint client selection and resource management modelling vehicle-RSU links — [Computer Networks 2023](https://www.sciencedirect.com/science/article/abs/pii/S1389128623005455)
- VR-VFL: joint rate and client selection under imperfect CSI — [arXiv 2602.03711](https://arxiv.org/html/2602.03711)
- Multi-objective distributed client selection in FL-IoV, scoring sample count, throughput, compute and **data quality**; handover via X2 — [arXiv 2401.03159](https://arxiv.org/pdf/2401.03159); [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11243953/)
- V2X-boosted FL with contextual client selection — [arXiv 2305.11654](https://arxiv.org/pdf/2305.11654)
- DRL vehicle selection for asynchronous FL in VEC — [arXiv 2304.02832](https://arxiv.org/abs/2304.02832)
- Multi-server VEC FL with application-layer handover so updates are not lost when a vehicle leaves coverage — [Future Internet 17(7):315](https://doi.org/10.3390/fi17070315)
- Mobility-aware decentralised FL (leader selection, local iterations) — [arXiv 2503.06443](https://arxiv.org/pdf/2503.06443); multi-task variant — [arXiv 2503.06468](https://arxiv.org/pdf/2503.06468)
- FedCLF participant selection in heterogeneous IoV — [arXiv 2509.25233](https://arxiv.org/pdf/2509.25233); Sense4FL notes earlier selection ignores mobility — [arXiv 2503.17697](https://arxiv.org/pdf/2503.17697)

**Quality-weighted aggregation in vehicles (feature a)**
- FLSimCo: RSU aggregates vehicle models with weights based on **image blur level** — [arXiv 2408.00256](https://arxiv.org/html/2408.00256); [Springer](https://link.springer.com/article/10.1007/s44285-024-00020-5)
- FedPAW personalised aggregation weights for vehicle speed prediction — [arXiv 2412.01281](https://arxiv.org/pdf/2412.01281)
- Secure cluster-based hierarchical vehicular FL weighting clusters by reliability scores — [arXiv 2505.01186](https://arxiv.org/pdf/2505.01186)
- Aggregation weighting via generalisation-bound estimation (general FL) — [arXiv 2311.05936](https://arxiv.org/pdf/2311.05936)

**Compression in vehicular FL (feature d)**
- Sparse federated training of object detection in IoV — [arXiv 2309.03569](https://arxiv.org/pdf/2309.03569)

**Privacy / secure aggregation in vehicular FL (feature e)**
- Secure aggregation for privacy-preserving FL in vehicular networks (ACM 2024) — [ACM DL](https://dl.acm.org/doi/10.1145/3657644)
- Laplace noise during RSU-assisted aggregation (Sci. Rep. 2025) — [Nature](https://www.nature.com/articles/s41598-025-14341-0.pdf)
- LDP + blockchain edge intelligence with RSU aggregators — [arXiv 2208.12755](https://arxiv.org/pdf/2208.12755)
- Empirical analysis of secure FL for AV applications (2025) — [arXiv 2509.20223](https://arxiv.org/pdf/2509.20223)
- Janus (ICML 2025): dual-server multi-round verifiable secure aggregation, constant per-client cost — [PMLR](https://proceedings.mlr.press/v267/pu25a.html)

### Inferences
- (b) is the most crowded academic area: "select vehicles whose predicted RSU dwell time covers train + upload time" is published (2401.07224 and the ResearchGate paper). Only the layered mechanics (geometric threshold relaxation, region stratification, starvation promotion) can carry novelty.
- (c) and (e) taken alone are textbook; claim them only as dependent claims or as part of an interaction (e.g., how q_k is applied inside masks or clipped for DP).

### Gaps
- Full texts of most vehicular papers were not read; claims about what they omit are from snippets.
- A search summary described a framework "Janus" combining adaptive top-k + EF, homomorphic encryption and compressed downlink broadcast for AVs; the follow-up search found only the unrelated ICML Janus paper. Treat as unverified.

## Q3. Which combinations of (a)-(f) appear together in one reference?

### Takeaway
Pairs are common; triples are rare; no reference found combines (a) multi-factor quality weights with (b) link-uptime RSU selection and (d) budgeted sparsification, let alone with (e) and (f). The closest multi-feature items are Toyota US 12,258,045 (a-ish + b-ish), the multi-objective IoV selector (a-ish inside b), and FLAA-V2V (a-ish + b-ish + e).

### Cited Findings
| Reference | a quality weight | b link/mobility selection | c FedNova/momentum | d compression | e DP/SecAgg | f faults/Byzantine |
|---|---|---|---|---|---|---|
| Toyota US 12,258,045 ([GP](https://patents.google.com/patent/US12258045)) | sensor-quality metadata (selection; weighting only per one summary) | metadata-based election in micro cloud (not link uptime) | no evidence | no evidence | no evidence | no evidence |
| MERL US 12,369,186 ([GP](https://patents.google.com/patent/US12369186)) | ? | communication/computation-aware vehicle selection | ? | ? | ? | ? |
| US 2022/0414464 A1 ([GP](https://patents.google.com/patent/US20220414464A1/en)) | data-quality-index weights | no | no | no | no | no |
| CN 120546993 B ([GP](https://patents.google.com/patent/CN120546993B/en)) | similarity-to-global weights | no | no | no | Laplace noise | no |
| Multi-objective IoV selection ([arXiv](https://arxiv.org/pdf/2401.03159)) | data quality as a selection score | throughput + handover | no | no | no | no |
| FLSimCo ([arXiv](https://arxiv.org/html/2408.00256)) | blur-based RSU weights | mobility-aware | no | no | no | no |
| C-V2X Mode 4 selection ([arXiv](https://arxiv.org/pdf/2401.07224)) | no | RSU dwell time | no | no | no | no |
| FLAA-V2V preprint ([RG](https://www.researchgate.net/publication/396354516_Privacy-Preserving_Federated_Learning_with_Adaptive_Model_Aggregation_for_Efficient_Vehicle-to-Vehicle_V2V_Communication_in_Intelligent_Transportation_Systems)) | RSU aggregation weighted by data-quality metrics | QoS-aware client selection | no | no | LDP + lightweight HE | no |
| Secure hierarchical vehicular FL ([arXiv](https://arxiv.org/pdf/2505.01186)) | reliability-weighted clusters | clustering | ? | ? | secure aggregation | trust weighting |
| DGC / Sahu 2021 | no | no | momentum correction | top-k + EF, budget | no | no |
| Bonawitz 2017 + DP-FedAvg | no | no | no | no | yes (weights capped in DP-FedAvg) | dropout recovery |

("?" = not determinable from snippets.)

### Inferences
- **Appear together already:** (a)+(b) in a loose sense (data quality as a selection criterion; sensor-quality metadata driving election); (a)+(e) (quality-weighted RSU aggregation with LDP/HE in FLAA-V2V; similarity weights + Laplace noise in CN 120546993 B); (b)+resource allocation; (d)+(c) in DGC; (e)+(f dropouts) in Bonawitz.
- **Not found together:** (a) as a *multiplicative on-vehicle score including calibration and label-histogram Hellinger similarity* combined with (c) FedNova normalisation; (b) link-uptime selection + (d) a per-round byte budget allocated across the selected set; (d) error-feedback residual handling coupled to (f) CRC-failure/straggler outcomes; (a) quality weights applied *inside* (e) pairwise-masked secure aggregation (so the server never sees individual q_k) with DP sensitivity bounded by a q_k cap; (d) sync-state-aware delta-vs-full multicast downlink at an RSU.
- These "not found" results rest on snippet-level searching and need a professional search before relying on them.

### Gaps
- Full claims of Toyota US 12,337,867 ("communication-aware FL") could change the (b) row materially.
- Whether FLAA-V2V is peer-reviewed and its publication date (preprint; claims unverified).

## Q4. Candidate independent-claim concepts (ranked by likely novelty / inventive step)

### Takeaway
The strongest positions are mechanism-level interactions that solve a technical problem the single-feature prior art creates: (1) privacy-preserving quality weighting inside secure aggregation, (2) a link-uptime-aware byte-budget allocator coupled to error feedback, and (3) a composite on-vehicle quality score feeding normalised averaging. All are rated moderate at best; each needs a measured technical effect in the specification and should be drafted as server-side, vehicle-side and system claims. Pure selection or pure compression claims are weak.

### Cited Findings (basis for the distinctions)
- Sensor-quality metadata already drives vehicle election (Toyota) — [US12258045](https://patents.google.com/patent/US12258045)
- Data-quality-index weighting already claimed generically — [US20220414464A1](https://patents.google.com/patent/US20220414464A1/en)
- Dwell-time-based RSU selection already published — [arXiv 2401.07224](https://arxiv.org/pdf/2401.07224); [US20190215378A1](https://patents.google.com/patent/US20190215378A1/en)
- Fixed-budget top-k and adaptive budget across training published — [Sahu 2021](https://mcanini.github.io/papers/rethink-gs.neurips21.pdf)
- Pairwise masks and capped-weight DP estimators published — [Bonawitz 2017](https://eprint.iacr.org/2017/281); [McMahan 2018](https://arxiv.org/abs/1710.06963)
- India: CRI claims need a technical effect beyond the algorithm (s.3(k), CRI Guidelines 2025) — recorded in `background/PATENT_READINESS.md`

### Inferences (candidate claims)

**K1 (best): Quality-weighted secure aggregation without revealing per-vehicle quality.** Vehicle computes q_k on-board (label-free sensor health x calibration residual x Hellinger similarity of its label/class histogram to an RSU-broadcast fleet reference, floored at q_min and capped at q_max); multiplies its clipped, normalised update by q_k and appends q_k as an extra coordinate; both are pairwise-masked; the RSU recovers sum(q_k Δ_k) and sum(q_k) only, divides, and adds DP noise whose sensitivity is set from C*q_max/sum-lower-bound. Technical effect: data-quality weighting and DP/SecAgg at once without the server learning which vehicle is degraded (a privacy leak in Toyota-style metadata exchange). Closest art: FLAA-V2V (quality weights + HE), Bonawitz 2017, DP-FedAvg capped weights, Toyota US 12,258,045. Risk: an examiner may combine DP-FedAvg (weighted, capped) + Bonawitz + a quality score as obvious; also **the project does not implement this yet** (README audit says normalised averaging is not on the secure path), so it must be built and measured before filing to be enabled and credible.

**K2: Link-uptime-aware per-round byte-budget allocation with persistent error feedback.** RSU predicts each candidate's remaining link uptime from trajectory, admits vehicles with uptime >= train + upload time (relaxing the threshold geometrically if fewer than m qualify), then splits a per-round MB budget across admitted vehicles in proportion to predicted uplink capacity x q_k; each vehicle picks k to fit its share, keeps the untransmitted residual, and on CRC failure or upload timeout the RSU signals "not received" so the vehicle keeps the full residual for the next round. Technical effect: bounded per-round airtime with no lost gradient mass under link drops. Closest art: arXiv 2401.07224 (dwell time), Sahu 2021 (budgeted top-k), DGC, MERL US 12,369,186, Toyota US 12,337,867 (unread). Risk: moderate obviousness; each element is known; the "residual retained on failed delivery" coupling is the most specific hook. Partly implemented (adaptive sparsity under budget exists; uptime-proportional allocation and NACK-driven residual retention need checking in `compression.py`/`selection.py`).

**K3: Composite on-vehicle quality score feeding normalised (FedNova-style) averaging.** Weight = q_k x n_k / tau_k with q_k as in K1, where the label-distribution term uses only a histogram (no labels or data leave the vehicle) and the sensor term uses self-diagnosed calibration residuals. Technical effect (needs multi-seed evidence): lower cross-client dispersion and robustness to degraded sensors (earlier audit: -31 % dispersion at one seed). Closest art: US 2022/0414464 A1, Toyota US 12,258,045, FLSimCo, FedNova. Risk: high obviousness ("multiply known weights"); strengthen with a specific calibration-residual measurement and the floor/cap rule, or merge into K1 as dependent claims.

**K4: Regionally stratified selection with geometric threshold relaxation and starvation promotion.** Closest art: multi-objective IoV selection (2401.03159), FedSSC, fairness-aware selection literature. Risk: medium-high; earlier audit found the starvation guard does not deliver its stated bound, so fix first. Best as dependent claims under K2.

**K5: Sync-state-aware multicast downlink.** RSU tracks each vehicle's last-acknowledged global version; vehicles in sync receive a multicast compressed delta (fp16+gzip+CRC), out-of-sync vehicles a unicast full model or a cumulative delta. Closest art: Google US 10,657,461 family (compressed updates), bidirectional-compression papers (not searched here), Qualcomm sidelink FL. Novelty unknown (no vehicular reference found), but the evidence for the downlink saving is not isolated in the results; needs its own measurement.

**K6 (weak): Exact pairwise-mask cancellation via norm-matched float64 arithmetic.** Narrow implementation detail; masks are simulated, not cryptographic. Keep only as a dependent claim.

**Not worth independent claims:** FedNova + server momentum alone; DP with RDP accounting alone; trimmed mean alone; fp16+gzip alone — all textbook (see Q2).

### Gaps
- Whether K1 is already disclosed in a 2024-2026 paper on "weighted secure aggregation" or "private weighted FedAvg" was not searched specifically; do this first.
- Toyota US 12,337,867 and MERL US 12,369,186 claims must be read against K2.
- No assessment of whether any K-concept is already disclosed by this project's own public material (repository, README, preprints); see `background/PATENT_READINESS.md` §1 on disclosure timing. If the repository is public, Indian novelty may already be compromised for features described in it.
