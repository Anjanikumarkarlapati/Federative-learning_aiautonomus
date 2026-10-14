# Patent readiness memo for the FL-AV paper

*Working notes, not legal advice. Every legal and fee statement below comes from web-search summaries of law-firm and official pages, because the primary sources (USPTO, EPO, IP India, WIPO) could not be opened from this environment. Confirm each point with a registered patent attorney or agent before relying on it. Detailed notes with links are in `research_notes/Patent ready research paper 2026/`.*

## 1. The one rule that matters: file before you publish

* Making the repository, README, figures, a preprint (arXiv v1), a conference submission visible on OpenReview, a poster, a talk or a shared Claude session link public is a **public disclosure** that starts the novelty clock.
* **US:** one-year grace period for the inventor's own disclosures (35 USC 102(b)(1)).
* **EPO and most of Europe:** effectively no grace period; a disclosure before the filing date can destroy novelty.
* **India:** the grace rules are narrow and conditional (Patents Act s.31: certain exhibitions and papers of a "learned society", 12 months, claimed with Form 31 under the 2024 Rules). Do not rely on them for a preprint or ordinary journal paper.
* **China:** 6-month grace for four listed situations; journals, theses and online posts are not listed. **Japan:** 12 months, with a statement at filing.
* **Safe sequence:** (1) file a provisional (US) or provisional specification (India) that is written like a full specification; (2) only then post the preprint, submit to a venue, or open the repository; (3) file the complete application or PCT within 12 months of the first filing.
* A provisional supports only the claims its text enables, so write it from this paper plus the claims below, not from a summary.

## 2. Current status of this project

| Item | Finding (repository audit, 8 Oct 2026) |
|---|---|
| Repository visibility | **Private**, created 16 Sep 2026; no forks, stars or releases; no public-visibility event found. Earlier visibility history could not be proven from here. |
| Earliest content | First commit 16 Sep 2026 already contains the full architecture and README. |
| Possible indirect disclosure | Commit trailers contain claude.ai session links. If those sessions were ever shared publicly, they would disclose the content. Check their sharing status. |
| Authorship of the code | Every commit is authored by "Claude" with model co-author trailers. The repository owner account differs from the commit email's user. The requirements document the design follows is not in the repository. |
| Licence | No LICENSE file. Decide before any release. |

**Inventorship needs a human answer.** The USPTO (Nov 2025 guidance), EPO, UK, China and an Indian Controller decision all require a natural-person inventor, and the USPTO treats AI as a tool: inventorship turns on who conceived the claimed invention. Someone must be able to say, in their own words, what they conceived (the requirements, the choices, the measured decisions such as disabling adaptive epochs), and that should be documented now. Inventorship (conception, 35 USC 116) and paper authorship (ICMJE-style contribution) are different lists and may legitimately differ. Ownership (employer, university, funder) should be settled before filing.

## 3. What is realistically patentable here

Almost every ingredient is published prior art: FedAvg, FedProx, server momentum (Hsu 2019), FedNova (Wang 2020), top-$k$ with error feedback, Bonawitz-style secure aggregation, DP-FedAvg with Rényi accounting, and link-aware client selection for vehicles. A patent on any one of them would fail. The audit's candidate **combination or implementation claims**, each rated low to medium strength:

| # | Candidate | Technical problem and effect to argue | Main risk |
|---|---|---|---|
| C1 | On-vehicle quality score: Hellinger similarity to a fleet reference histogram, floored, multiplied by a label-free sensor-health term, used as an aggregation weight | Down-weights off-distribution and degraded vehicles without sharing labels; measured −31 % cross-client dispersion in isolation | Obviousness over quality-weighted aggregation literature; effect measured at one seed |
| C2 | Layered selector with geometric threshold relaxation, starvation promotion and region stratification | Avoids empty rounds under poor connectivity while keeping regional coverage | The starvation guard does not currently deliver its stated bound (audit finding 4) |
| C3 | Sync-state-aware delta or full-model multicast downlink | Saves downlink bytes while keeping out-of-sync vehicles consistent | Needs its own evidence; downlink savings are not isolated in the results |
| C5 | Norm-matched float64 mask arithmetic for secure aggregation | Makes the pairwise-mask cancellation exact at scale (error 1.3e-7 at 64 participants against 0.18 for float32) | Narrow; the secure aggregation is simulated and the masks are not cryptographic |

**Not patentable or not worth claiming:** adaptive local epochs (a measured negative result); the transport stack taken as a whole (its saving is mostly fp16 and multicast, which are well known).

**Eligibility and how to draft.** Under US § 101, the EPO's technical-character test, and India's s.3(k) with the July 2025 CRI Guidelines, "machine learning applied to data" is not enough; claims need a specific technical improvement, for example communication cost, verifiable cancellation accuracy, or robustness under link loss. Federated learning splits the method between server and vehicle, so draft **single-actor claim families** (server-side method, vehicle-side method, system) to avoid divided-infringement problems. Describe architecture, training procedure and a working example in enough detail to satisfy enablement; no jurisdiction requires publishing the training data itself.

**The more promising directions are not implemented yet** (from the audit): selection-propensity-corrected quality weighting, a tail-aware controller for the quality floor, verifiable quality scores, and secure collection of the reference histogram. They address the real weaknesses this study found, and implementing and measuring them would make a stronger application than the current components.

**Before drafting claims, fix the defects** listed in Section 5.8 of the paper. Claims written from the README would over-claim: the starvation guard, normalised averaging on the secure path and the privacy properties do not behave as the README states. One prior-art item needs counsel's attention: a Toyota US patent on vehicular federated learning with sensor-based selection (US 12,258,045), whose claims were not read. A proper prior-art search (Google Patents, Espacenet; CPC G06N 3/098, G06N 20/00, H04L 9/008, plus vehicle classes) has not been done.

## 4. Suggested order of work

1. **Now:** keep the repository private; do not post a preprint or share session links. Record who conceived what.
2. **Weeks 0 to 2:** fix the audit defects, regenerate all arms from one revision with at least ten seeds, add transport-matched and stronger federated baselines. This produces the evidence a claim of technical effect needs, and the paper's headline comparisons.
3. **Weeks 2 to 4:** prior-art search with an attorney; decide the claim families; file a provisional (US, about $65, $130 or $325 by entity size as reported) or an Indian provisional specification with drawings (system block diagram, server and vehicle flowcharts, round sequence diagram, privacy data-flow, model architecture, results).
4. **Only after the filing receipt:** post the preprint, submit to a venue, make the code public with a licence.
5. **Within 12 months:** complete specification or PCT. Reported national-phase deadlines: 30 months (US, Japan, China) and 31 months (EPO, India).

**If an inventor resides in India:** s.39 requires filing in India first (and waiting about six weeks) or obtaining Form 25 permission before filing abroad; the Indian provisional must be followed by the complete specification within 12 months with no extension. Expedited examination (Form 18A) is available to startups, small entities and applicants who named India as the search authority. Reported fee tiers conflict between sources; check the official schedule.

## 5. Costs as reported (unverified, subject to change)

* **USPTO** (schedule effective 19 Jan 2025, last revised Aug 2026): provisional $325 / $130 / $65 (large / small / micro); basic utility filing, search and examination $2,000 / $800 / $400. The USPTO's fee-setting authority was reported to be extended only to about December 2026.
* **EPO:** fees rose about 5 % on 1 Apr 2026; 30 % reduction for micro-enterprises, universities, non-profits and natural persons.
* **PCT:** international filing fee reported as CHF 1,330 plus per-sheet charges (the USPTO route lists $1,667 from Jan 2026); the 90 % reduction is for natural persons only.
* **India:** reported e-filing fees for the lowest tier: Form 1 INR 1,600, Form 9 INR 2,500, Form 18 INR 4,000; tiers for small entities conflict across sources.
* Attorney fees are not covered by any retrieved source.

## 6. What to tell venues and journals

No ML or medical-AI venue found restricts patenting or requires patent disclosure at submission. Journal competing-interests forms (Nature Portfolio, ICMJE-style forms) ask about planned, pending or issued patents, and the paper's *Competing interests* declaration has a placeholder for it. ICLR submissions are public on OpenReview during review, which counts as disclosure, so file first. Several venues now require disclosure of significant AI assistance in writing or coding; the paper has a placeholder.
