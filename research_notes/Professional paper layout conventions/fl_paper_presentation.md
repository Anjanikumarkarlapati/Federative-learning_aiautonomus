# Presentation conventions in well-regarded federated-learning (FL) and connected-autonomous-vehicle (CAV) ML papers

**Access caveat (read first).** The research sandbox blocked every direct paper host (arxiv.org, ar5iv, proceedings.mlr.press, proceedings.mlsys.org, openreview.net, ieeexplore.ieee.org, semanticscholar.org: DNS failure in WebFetch, HTTP 403 CONNECT in curl). Only WebSearch worked, and it returns summaries, not PDFs. So **no paper PDF was read.** Nothing below about page counts, algorithm-box layout, table typography, caption wording, or conclusion length is verified from the papers themselves. "Cited Findings" holds only what search results confirmed (venue, authors, abstract content, pages where listed). Everything else is under "Recalled (NOT verified)" and must be treated as background knowledge to be checked against the PDFs before being presented as fact. No literal captions or contribution bullets could be quoted verbatim from the papers; the few verbatim quotes are from abstracts/intro snippets returned by search.

## Q1. Section order, section titles, abstract content (length, numbers)

### Takeaway
Verified: abstracts in this field are one-paragraph problem -> gap -> method -> evidence arcs, and several carry concrete numbers (FedAvg "10-100x" fewer rounds, FedProx "22%" accuracy gain, secure-aggregation "1.73x" expansion). Section order is, from recall, near-universal: Introduction, Related Work (or Background), Method/Algorithm, Theory, Experiments, Conclusion.

### Cited Findings
- FedAvg (McMahan et al., AISTATS 2017, PMLR 54:1273-1282) states in its intro/abstract that it evaluates "five different model architectures and four datasets", and that FedAvg is "robust to unbalanced and non-IID data distributions" and cuts communication rounds "by 10-100x as compared to synchronized stochastic gradient descent". Abstract therefore includes quantified claims. — [WebSearch summary of arXiv 1602.05629](https://arxiv.org/html/1602.05629v4); [PMLR page](https://proceedings.mlr.press/v54/mcmahan17a.html)
- FedProx (Li et al.) abstract reports "22% average improvement in absolute test accuracy" in the most heterogeneous settings (an earlier arXiv version said 18.8%, so the number changed across versions; a lesson: pin numbers to the final version). Abstract also states the framing: "FedProx can be viewed as a generalization and re-parametrization of FedAvg". — [arXiv 1812.06127](https://arxiv.org/abs/1812.06127) (via search summary)
- SCAFFOLD (Karimireddy et al., ICML 2020, PMLR 119:5132-5143): abstract moves problem (FedAvg suffers "client drift" under non-iid data) -> tight FedAvg convergence characterisation -> fix (control variates) -> claim ("significantly fewer communication rounds and is not affected by data heterogeneity or client sampling"). Title changed from "...for On-Device Federated Learning" (v1) to "...for Federated Learning". — [arXiv 1910.06378](https://arxiv.org/abs/1910.06378v3); [ICML page](https://icml.cc/virtual/2020/poster/53058)
- FedNova (Wang, Liu, Liang, Joshi, Poor; NeurIPS 2020): abstract = problem (different local step counts -> "stationary point of a mismatched objective"), general analysis covering FedAvg/FedProx, proposed fix (normalized averaging). Names the phenomenon in the title ("Objective Inconsistency"). — [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2020/hash/564127c03caab942e503ee6f810f54fd-Abstract.html)
- Secure aggregation (Bonawitz et al., ACM CCS 2017): abstract states problem, protocol properties, security model (honest-but-curious and malicious servers, arbitrary user dropout) and a concrete number: "1.73x communication expansion" for 2^10 users, 2^20-dim vectors, 16-bit inputs. — [Google Research](https://research.google/pubs/pub47246/); [ePrint 2017/281](https://eprint.iacr.org/2017/281)
- Sattler et al. (IEEE TNNLS vol. 31 no. 9, pp. 3400-3413, 2020): abstract = problem (communication overhead), gap (existing compression is upstream-only or iid-only), method (Sparse Ternary Compression: top-k sparsification + downstream compression + ternarization + Golomb encoding), results (four tasks, beats FedAvg in non-iid, small batch, low participation; "Pareto-superior" even with iid data), closing recommendation (frequent low-bitwidth communication). Journal text differs from arXiv preprint wording. — [ar5iv / arXiv 1903.02891](https://ar5iv.arxiv.org/html/1903.02891); [Korea Univ. record](https://scholar.korea.ac.kr/handle/2021.sw.korea/53658)
- Kairouz et al., "Advances and Open Problems in Federated Learning", Foundations and Trends in Machine Learning vol. 14 no. 1-2, pp. 1-210 (published 23 Jun 2021), 59 authors in total (Kairouz plus 58); abstract is a definition of FL, privacy/data-minimisation framing, then a statement that it surveys progress and "open problems". — [Now Publishers](https://nowpublishers.com/article/Details/MAL-083); [arXiv 1912.04977](https://arxiv.org/abs/1912.04977)
- Peyton Jones' advice (talk "How to write a great research paper"): tell a story (problem, why interesting, why unsolved, your idea, evidence, comparison); one clear idea; introduction roughly one page. — [Microsoft Research talk page](https://www.microsoft.com/en-us/research/?p=186470) and [notes via talks.cam](https://talks.cam.ac.uk/talk/index/25220/)

### Inferences
- A defensible abstract template, consistent with every verified abstract above: (1) setting/problem, (2) specific gap in prior work, (3) "We propose X, which ..." (one sentence), (4) theory claim if any, (5) experimental scope, (6) one or two headline numbers with the baseline named, (7) optional implication. Roughly 5-8 sentences, ~150-250 words. Every verified abstract names the method and includes a quantified or precisely scoped claim; none are purely descriptive.
- Titles name either the problem ("Objective Inconsistency", "Heterogeneous Networks", "Non-IID Data") or the property delivered ("Communication-Efficient", "Robust", "Practical"). Coined names (FedAvg, FedProx, SCAFFOLD, FedNova, STC) are the norm for method papers.

### Recalled (NOT verified, from background knowledge)
- Typical section order: 1 Introduction; 2 Related Work (FedAvg puts it as "Related Work" after the method in Sec. 4 or near the end; FedProx puts Related Work as Sec. 2 and Preliminaries as Sec. 3); 3 Method; 4 Convergence analysis; 5 Experiments; 6 Conclusion; References; Appendix. ICML/NeurIPS-style papers push proofs to the appendix. FedAvg's actual order (recalled): 1 Introduction, 2 The FederatedAveraging Algorithm, 3 Experimental Results, 4 Conclusions and Future Work, then Acknowledgements/References/Appendix.
- Page counts (recalled): FedAvg 10 pages + appendix (AISTATS limit 8 + refs, camera-ready PMLR 1273-1282 = 10 pp.); FedProx (MLSys) ~ 12-13 pp. main + large appendix; SCAFFOLD ~ 12 pp. main in ICML + appendix to ~ 50 pp. on arXiv; FedNova 12-ish pp. main + appendix; Sattler TNNLS 14 pp. (pp. 3400-3413 = 14 pages, which is derivable from the verified page range); Bonawitz CCS ~ 17 pp.; Kairouz 210 pp. (verified).

## Q2. How contributions are phrased and placed

### Takeaway
Contributions sit at the end of the introduction, after the problem and gap. In verified summaries the wording follows "we propose / we provide / we evaluate" with the claim quantified; ML-conference papers often use prose rather than numbered bullets, IEEE journals usually use numbered or bulleted lists.

### Cited Findings
- FedProx's intro-style framing sentences, quoted by search summary: "A key insight we have in developing FedProx is that an interplay exists between systems and statistical heterogeneity in federated learning", and "While this re-parameterization makes only minor modifications to the method itself, these modifications have important ramifications both in theory and in practice." — [arXiv 1812.06127 via search](https://arxiv.org/pdf/1812.06127)
- FedProx contribution pattern as summarised: diagnose the interplay between systems and statistical heterogeneity; add a proximal term; give convergence guarantees "while adhering to device-level systems constraints by allowing each participating device to perform a variable amount of work"; evaluate on "a suite of synthetic and real-world federated datasets". — same source
- FedAvg's intro states its contributions in three parts per the search summary: (1) identification of the problem and naming of FL, (2) the FederatedAveraging algorithm, (3) an "extensive empirical evaluation, considering five different model architectures and four datasets". The search could not retrieve the exact numbered list text. — [arXiv HTML v4](https://arxiv.org/html/1602.05629v4)
- Peyton Jones: "nail your contributions", state them explicitly and make claims refutable/checkable; put related work at the end. — [talk notes](https://talks.cam.ac.uk/talk/index/25220/)

### Inferences
- Do not invent a list of vague benefits. Each contribution should be (a) one sentence, (b) begin with a strong verb (propose, derive, prove, show, release, evaluate), (c) point to a section or theorem, (d) include scope or number where honest.
- Lead the contributions list with the idea, then theory, then empirics; last bullet often "code/data released".

### Gaps
- Exact contribution-bullet wording of Li/Karimireddy/Wang/Sattler/Bonawitz could not be retrieved; no literal bullet lists to quote. Recall only: Sattler (TNNLS) uses a short list of 3 numbered contributions near the end of the intro (not verified).
- No vehicular-FL paper text could be accessed (see Q4).

## Q3. Algorithms, notation tables, theorems, equations

### Takeaway
Not verifiable in this session. Everything below is recalled convention, flagged as such.

### Cited Findings
- (Only indirect.) SCAFFOLD's abstract says it gives "a tight convergence characterization" and FedNova's gives "a general theoretical framework", i.e., theory is presented as a headline result and both are theory-led papers. — [ICML page](https://icml.cc/virtual/2020/poster/53058); [NeurIPS page](https://proceedings.neurips.cc/paper/2020/hash/564127c03caab942e503ee6f810f54fd-Abstract.html)

### Inferences
- Given that all of the verified papers are named-method papers, a boxed "Algorithm 1" for the proposed method is expected; an unnamed method with no algorithm box would read as underspecified.

### Recalled (NOT verified)
- Algorithm box (LaTeX algorithm/algorithmic, caption "Algorithm 1 FederatedAveraging" or "Algorithm 1 FedProx (Proposed Framework)"): title line, then `Input:`/`Output:` or "Server executes:" and "ClientUpdate(k, w):" procedures, numbered lines, `for each round t = 1, 2, ... do`, `for each client k in S_t in parallel do`, comment markers `// ...`. FedAvg prints hyperparameters in the header (K clients, B minibatch size, E epochs, eta learning rate). SCAFFOLD, FedNova have side-by-side or colour-highlighted lines marking the difference from FedAvg.
- Notation: ML-conference papers usually define symbols inline at first use or in a preliminaries section ("Problem formulation: min_w F(w) = sum_k p_k F_k(w)"). A full symbol table is common in IEEE journals and long theory appendices (SCAFFOLD/FedNova appendices include them). Kairouz et al. includes a notation/glossary-style section; exact form unverified.
- Equations: numbered (1), (2), ... only when referenced; sentence punctuation is kept after displayed equations (comma/period); "where ..." clauses define symbols right after the equation. Theorems: "Theorem 1 (Convergence of FedProx)." italicised statement, assumptions numbered ("Assumption 1 (B-local dissimilarity)"), proofs in appendix with "Proof sketch." in main text.

### Gaps
- No algorithm box, theorem box, or notation table could be inspected.

## Q4. Results presentation: tables, plots, captions; CAV/vehicular exemplars

### Takeaway
Cannot confirm table/plot/caption typography from the papers. For vehicular FL, the search surfaced TIV/TITS items but none could be read; no vehicular paper is verified beyond bibliographic metadata.

### Cited Findings
- Sattler et al. evaluate on "four learning tasks" and report scenarios along explicit axes (non-iid-ness, batch size, number of clients, participation rate), and use the framing "Pareto-superior", implying accuracy-vs-communication-cost trade-off plots rather than only accuracy tables. — [ar5iv 1903.02891](https://ar5iv.arxiv.org/html/1903.02891) (abstract-level)
- Bonawitz et al. present a quantified overhead table/claim ("1.73x communication expansion" at specified scale) with parameters stated (users, vector dimension, input bit-width). — [Google Research](https://research.google/pubs/pub47246/)
- Vehicular-FL items found (metadata only; text NOT read): "Federated Learning for Connected and Automated Vehicles: A Survey of Existing Approaches and Challenges", IEEE Trans. Intelligent Vehicles vol. 9 no. 1, pp. 119-137 (2024) (covers centralized and decentralized FL frameworks, data sources and security techniques, application-specific base models and datasets, open challenges) — [Princeton record](https://collaborate.princeton.edu/en/publications/federated-learning-for-connected-and-automated-vehicles-a-survey-/); "Federated Intelligence for Intelligent Vehicles", IEEE TIV vol. 9 no. 5, pp. 4835-4839 (2024), a short letter — [Unpaywall/IEEE record](https://unpaywall.org/10.1109%2FTIV.2024.3415410); a TIV 2024 paper on privacy-preserving data aggregation for the Internet of Vehicles with FL, DOI 10.1109/TIV.2024.3411313 — [OpenAlex](https://api.openalex.org/works/doi:10.1109%2FTIV.2024.3411313); another candidate DOI 10.1109/TITS.2025.3546088 (title not retrieved) — [OpenAlex](https://api.openalex.org/works/doi:10.1109%2FTITS.2025.3546088). Note "Federated Learning on the Road: Autonomous Controller Design for CAVs" (arXiv 2102.03401) turned out to be IEEE Trans. Wireless Communications 21(12), 2022, not TIV/TITS — [Oulu repo](https://urn.fi/URN:NBN:fi-fe202301183455).

### Inferences
- The vehicular exemplar requirement (>=2 papers read) is NOT met. The caller should obtain TIV/TITS PDFs another way (e.g. user supplies PDFs or opens network access) before relying on any claim about IEEE vehicular style.

### Recalled (NOT verified)
- Tables: best result bold, second best underlined; "mean ± std over 3-5 seeds"; header carries units and direction arrows (Acc. (%) ↑, Loss ↓, Rounds-to-target ↓, Comm. (MB) ↓); consistent decimals (one or two); methods in rows, datasets/heterogeneity levels in column groups with `\toprule/\midrule/\bottomrule` (booktabs); proposed method last row. FedAvg uses "rounds to reach target accuracy" tables with a speed-up column; FedProx/SCAFFOLD/FedNova plot training loss and test accuracy vs communication rounds.
- Plots: x-axis = communication rounds, y-axis = test accuracy or training loss (often log), shaded error band or multiple seeds, same colour/marker per method in every figure, legend inside top-left/bottom-right or shared above the grid of subplots; multi-panel (a)(b)(c) with sub-captions.
- Captions: self-contained, below figures and above tables, a short noun-phrase or declarative first sentence (some venues use bold lead-in: "**Figure 2. Test accuracy vs. rounds.** ..."), then settings ("E=20, 10% client participation, 3 seeds, shaded = ±1 std"). IEEE style: "Fig. 3." / "TABLE II" (small caps, roman numeral, caption above table).
- IEEE vehicular papers: two-column IEEEtran, 10-14 pp. (TIV regular), abstract <=250 words single paragraph + "Index Terms", numbered contributions, system-model figure at the start of Section III, symbol table ("TABLE I: Summary of Notations"), numbered references in order of appearance in brackets [1].

### Gaps
- No table or caption literal quotes available for any exemplar.

## Q5. Limitations, negative results, related work, conclusion, references; reviewer hallmarks

### Takeaway
Verified: major venues explicitly require claims to match evidence, limitations to be stated, and error bars/seeds to be disclosed. Per-paper handling of limitations in the exemplars could not be verified.

### Cited Findings
- NeurIPS paper checklist: the first item asks that the abstract and introduction "accurately reflect the paper's contributions and scope" and that claims "match theoretical and experimental results, and reflect how much the results can be expected to generalize"; a "No/N/A" there "will not be perceived well by the reviewers". Reviewers use the checklist as one factor. Answering "no" to other items (e.g., error bars too expensive) is acceptable with justification. Items cover limitations, assumptions and proofs, reproducibility details, error bars/statistical significance, compute resources. — [NeurIPS 2021 checklist](https://neurips.cc/Conferences/2021/PaperInformation/PaperChecklist); [NeurIPS 2022 checklist](https://neurips.cc/Conferences/2022/PaperInformation/PaperChecklist)
- Third-party summary of the error-bar item: state number of runs/seeds and whether ± is std or standard error (non-official). — [checklist summary](https://cdn.jsdelivr.net/npm/claude-code-skills@0.5.0/data/skills/ml-paper-writing/references/checklists.md) (unofficial; low reliability)
- The NeurIPS 2019 reproducibility programme and Pineau's reproducibility checklist underpin these requirements. — [arXiv 2003.12206](https://arxiv.org/pdf/2003.12206); [Pineau checklist](https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf)
- Peyton Jones: put related work at the end (after the reader understands your idea) rather than after the intro; put readers first; tell a story. — [talk notes](https://talks.cam.ac.uk/talk/index/25220/)
- FedProx abstract itself acknowledges that a degenerate setting (proximal term set to 0) and dropping stragglers both "can adversely impact convergence behavior": an example of the paper stating when its own baseline/ablation fails. — [arXiv 1812.06127 via search](https://arxiv.org/pdf/1812.06127)
- Sattler et al. state the regime where their method wins (non-iid, small batch, many clients, low participation) and a regime where it still holds (iid), i.e. claims scoped by condition. — [ar5iv 1903.02891](https://ar5iv.arxiv.org/html/1903.02891)
- Kairouz et al. is explicitly organised around "open problems" (title), i.e. future work is the survey's main product. — [Now Publishers](https://nowpublishers.com/article/Details/MAL-083)

### Inferences
- Hallmarks of a professional manuscript supported by the sources: scoped, quantified claims; named baselines; seeds/error bars disclosed; stated assumptions; limitations stated plainly; reproducibility details (hyperparameters, compute, code). Amateur signals (inferred as the inverse): unquantified superlatives, accuracy numbers without variance or baseline, symbols that change meaning between sections, related work that is a list of citations without positioning against your method.
- Wording tip for calibrated claims: "improves ... by X points on A and B under Dirichlet(0.1) partitioning" rather than "significantly outperforms all methods".

### Recalled (NOT verified)
- Related work: ML conferences use a short ~0.5-1 page paragraph-per-theme (local-SGD / FL optimisation, non-iid, communication compression, privacy), each ending with a sentence on how this work differs; IEEE journals use a longer subsectioned Related Work (often with a comparison table of prior approaches with check-marks).
- Conclusions: ML conference papers 1 short paragraph (FedAvg's "Conclusions and Future Work" is a few sentences listing next steps such as combining with differential privacy and secure multi-party computation); IEEE journals 1-2 paragraphs + a "Future work" sentence list. Negative results/limitations are usually in a sentence or a paragraph in experiments or the appendix, rarely a dedicated section before the NeurIPS checklist era.
- References: ML conferences: author-year or numeric (natbib), arXiv ids included; IEEE: numbered, abbreviated journal names, "et al." after six authors; FTML monograph (Kairouz) uses author-year with a large bibliography.

### Gaps
- No exemplar's actual limitations/negative-result paragraph was read. ICML/IEEE Transactions reviewer guideline pages were not retrieved (only NeurIPS checklist). Simon Peyton Jones' slides and Jennifer Widom's advice were not directly read (only third-party notes of the talk). Nature/IEEE style guides not retrieved.

## Recommended next step for the caller
Re-run this task with network access to arxiv.org / ar5iv / PMLR / IEEE Xplore (or have the user drop the PDFs into the repo). The "Recalled" blocks above are a hypothesis set; verify each before encoding it as a rule in the rewrite.
