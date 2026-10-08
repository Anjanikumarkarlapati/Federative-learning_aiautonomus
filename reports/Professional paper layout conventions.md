# Match IEEE Transactions layout and FL rhetoric

A professionally published federated-learning (FL) or intelligent-vehicles paper looks like an IEEE Transactions article built with the stock `IEEEtran` journal class. That means US Letter, 10 pt Times on 12 pt leading, two 21 pc (3.5 in) columns separated by a 1 pc gutter, a 24 pt non-bold title, a bold 9 pt "*Abstract*—" paragraph followed by "*Index Terms*—", centred small-caps Roman-numeral section heads, 8 pt captions ("Fig. 1." below figures, "TABLE I" above tables), and 8 pt numbered references from `IEEEtran.bst`. **The installed class file (IEEEtran.cls v1.8b, 2015/08/26) confirms all of these values**, and the class file is the authoritative source for its own defaults. It also corrects two of the research notes' recollections: in journal mode a one-line figure caption is set flush left, not centred, and the bottom margin is close to 0.5 in, not 1 in. The content conventions are a separate layer that the class cannot enforce. Strong FL papers name one method, state a problem-gap-method-evidence arc in the abstract with one or two headline numbers against a named baseline, list contributions at the end of the introduction, and scope every claim by the regime where it holds. Venue checklists now require that claims match the evidence and that limitations, seeds and error bars are disclosed. The main caveat is about sources. No paper PDF and no official IEEE style page could be opened during research, so most fine-grained typographic rules (footnote wording, reference punctuation, table typography, algorithm layout) are recalled conventions and are marked unverified below. No TIV or TITS vehicular paper was read beyond its bibliographic metadata.

## The class file fixes the page geometry: 43 pc of text in two 21 pc columns

The draft (`paper/ieee/paper.tex`) already loads `\documentclass[journal]{IEEEtran}`, so the geometry comes from the class rather than from anything the authors set. Reading the installed class settles the numbers the research notes could only recall. In 10 pt journal mode, **`\textwidth` is 43 pc and `\columnsep` is 1 pc**. The class comments this as "2 x 21pc + 1pc = 43pc", which gives two 3.5 in columns. **`\textheight` is 58 pc (696 pt), quantised to exactly 58 lines per column**, and the default top margin is 58 pt. The text block is centred horizontally, which leaves side margins of about 0.67 in, and the arithmetic leaves roughly 0.5 in at the bottom of the page ([IEEEtran.cls v1.8b, lines 1700-1740](file:///usr/share/texlive/texmf-dist/tex/latex/ieeetran/IEEEtran.cls)). The class defines its size table explicitly: normal size 10/12 pt, `\small` 9/10 pt, `\footnotesize` 8/9 pt, `\scriptsize` 7/8 pt and `\Huge` 24/28 pt ([IEEEtran.cls, 10 pt size table](file:///usr/share/texlive/texmf-dist/tex/latex/ieeetran/IEEEtran.cls)). The class header documents the mode options: `journal`, `conference` and `technote`, the review modes `peerreview` and `peerreviewca`, and 9/10/11/12 pt sizes, with 10 pt, letterpaper and two columns as defaults. It also provides a `captionsoff` option for journals that want captions removed and figures moved to separate pages at first submission ([IEEEtran.cls header, as mirrored](https://ece.iisc.ac.in/~spcom/2018/img/IEEEtran.cls)).

These dimensions match IEEE's own artwork guidance, which sets **figure widths at 3.5 in (88.9 mm, 21 pc) for one column and 7.16 in (182 mm, 43 pc) for two columns**, a maximum depth of about 9 in to leave room for the caption, and resolution above 300 dpi for colour or greyscale and above 600 dpi for line art ([Proceedings of the IEEE figure guidelines](https://proceedingsoftheieee.ieee.org/resources/guidelines-for-figures-and-tables/)). The practical consequence is that figures should be drawn at final size, about 3.5 x 2.4 in or 7.16 x 2.8 in, with 8 pt text so that nothing is rescaled. IEEE does not publish an official minimum font size inside figures. The commonly quoted floor of about 6-7 pt is unverified. Matplotlib's default PDF output embeds Type 3 fonts, which some publishers reject, so set `pdf.fonttype = 42` and check the result with `pdffonts` ([Matplotlib fonts documentation](https://matplotlib.org/stable/users/explain/text/fonts.html)). The SciencePlots `ieee` style presets IEEE column widths and font sizes ([SciencePlots README](https://github.com/garrettj403/SciencePlots)).

| Element | Value in IEEEtran journal mode (10 pt) | Status |
|---|---|---|
| Paper, columns | US Letter; two 21 pc columns, 1 pc gutter, 43 pc text width | Verified in class |
| Text height | 58 pc, 58 lines per column; top margin 58 pt; bottom margin about 0.5 in (derived) | Verified in class; bottom margin derived from class values, so measure a compiled page |
| Body | 10/12 pt, paragraph indent 1 em, no space between paragraphs | Verified in class |
| Title | `\Huge` 24/28 pt, not bold | Verified in class |
| Abstract, Index Terms | `\small` 9 pt, bold body; italic lead-in "Abstract—" / "Index Terms—" | Verified in class |
| Captions | `\footnotesize` 8 pt | Verified in class |
| References | `\footnotesize` 8 pt, numbered list, `IEEEtran.bst` | Verified in class; .bst installed |
| Figure widths, dpi | 3.5 in / 7.16 in; >300 dpi colour, >600 dpi line art | Verified (IEEE guideline) |
| Running heads, first-page footnote wording, page limits | See below | Unverified (recalled) |

## Front matter runs title, author footnotes, abstract, Index Terms in a fixed order

The class confirms the typography of the opening block, and the notes supply recalled conventions for its content. The **title is set in `\Huge` 24 pt and is not bold** ([IEEEtran.cls `\maketitle`](file:///usr/share/texlive/texmf-dist/tex/latex/ieeetran/IEEEtran.cls)). In Transactions style, membership grades follow the author names ("First A. Author, *Member, IEEE*"), and affiliations go in first-page footnotes rather than under the names. Conference mode puts affiliations under the names instead. The recalled footnote sequence is as follows: a "Manuscript received …; revised …; accepted …" line, a funding line with a "(Corresponding author: …)" tag, an affiliation and e-mail line, and a final "Digital Object Identifier 10.1109/…" line. Even pages carry the journal name, volume, number and date in the running head, and odd pages carry the author running title, which is what `\markboth` produces. These wordings are unverified, and the publisher fills in most of them at production, so a submission only needs correct `\thanks{}` content and a sensible `\markboth`.

The abstract environment in two-column journal mode prints the word *Abstract* in italic, then an em dash, then the whole paragraph in **bold 9 pt**. `IEEEkeywords` prints *Index Terms*— in the same way ([IEEEtran.cls `\abstract`, `\IEEEkeywords`](file:///usr/share/texlive/texmf-dist/tex/latex/ieeetran/IEEEtran.cls)). The recalled content rules are one paragraph of about 150-250 words, with no citations, displayed equations or figure references, abbreviations defined at first use, and Index Terms drawn from the IEEE taxonomy in alphabetical order. These rules are unverified, and the exact word limit varies by journal. The first word of the Introduction uses the drop cap `\IEEEPARstart`, and the class implements this as a two-line drop with the rest of the word in small caps.

The content of the abstract is better sourced than its format. Every FL exemplar the research could confirm follows a problem-gap-method-evidence arc with a quantified claim. FedAvg reports a "10-100x" reduction in communication rounds over synchronous SGD across "five different model architectures and four datasets" ([McMahan et al., arXiv 1602.05629](https://arxiv.org/html/1602.05629v4); [PMLR](https://proceedings.mlr.press/v54/mcmahan17a.html)). FedProx reports a "22% average improvement in absolute test accuracy" in the most heterogeneous settings. An earlier version said 18.8%, which shows why abstract numbers must be pinned to the final run ([Li et al., arXiv 1812.06127](https://arxiv.org/abs/1812.06127)). Secure aggregation states a "1.73x communication expansion" with the scale specified: 2^10 users, 2^20-dimensional vectors and 16-bit inputs ([Bonawitz et al., Google Research](https://research.google/pubs/pub47246/)). Sattler et al.'s TNNLS abstract runs from problem through gap, method and results to a closing recommendation ([arXiv 1903.02891](https://ar5iv.arxiv.org/html/1903.02891)). A defensible template has five to eight sentences: setting, specific gap, "We propose X", theory claim if any, experimental scope, one or two headline numbers against a named baseline, and the implication.

## Headings, equations, floats and references follow IEEE numbering conventions

**Headings are verified in the class.** Level 1 is centred small caps with a Roman numeral and a full stop ("I. Introduction"). Level 2 is italic and flush left, labelled "A.". Level 3 is italic, run in and indented by the paragraph indent, labelled "1)". Level 4 is run in and labelled "a)". Cross-references to a subsection print as "I-A". Appendices default to "Appendix A", and the `romanappendices` option switches to "Appendix I". The references heading is unnumbered, and the list is set in 8 pt ([IEEEtran.cls, sectioning and `\thebibliography`](file:///usr/share/texlive/texmf-dist/tex/latex/ieeetran/IEEEtran.cls)). The unnumbered singular "Acknowledgment" heading before the references, and the biography blocks after them (`IEEEbiography` with a 1 x 1.25 in photo, or `IEEEbiographynophoto`), are recalled practice and unverified.

**Equation numbers are verified as "(1)" at the right margin.** The remaining equation rules are recalled conventions. Displayed equations are punctuated as part of the sentence and followed by a "where …" clause that defines the symbols. Variables are italic, vectors bold, and operators upright (`\operatorname`). Cross-references are written "(1)", with "Equation (1)" only at the start of a sentence. Long expressions break at an operator with `split` or `multline`. IEEE journal papers commonly include a "Summary of Notations" table near the system model, whereas ML-conference papers define symbols inline. Theorems are labelled in the form "Theorem 1 (Convergence)", assumptions are numbered, and full proofs go in an appendix with a proof sketch in the body.

**Float captions are verified in the class.** A table caption is centred above the table in 8 pt: "TABLE I" on one line, with the title in small caps on the next. A figure caption sits below the figure: "Fig. 1." followed by two non-breaking spaces and the text in 8 pt. A caption longer than a line wraps as a justified paragraph. **A one-line figure caption is left-aligned in journal mode and centred only in conference mode**, which contradicts the notes' recollection. The class names are `\figurename` = "Fig." and `\tablename` = "TABLE" ([IEEEtran.cls `\@makecaption`](file:///usr/share/texlive/texmf-dist/tex/latex/ieeetran/IEEEtran.cls)). The table typography conventions are recalled, not verified:

- Horizontal rules only (booktabs top, mid and bottom), with no vertical rules.
- Units in the column headers, such as "Acc. (%)", with direction arrows where helpful.
- Numbers right-aligned or decimal-aligned, with a consistent number of decimals.
- Best result in bold and second-best underlined.
- The proposed method in the last row.
- `table*` or `figure*` for spanning floats, which LaTeX places only at the top of a page.

Floats should be placed with `[!t]`, cited in the text before they appear, and kept in numerical order. Sub-panels are labelled "(a)", "(b)" and cited as "Fig. 2(a)". Captions should stand on their own: say what is plotted, the setting, the number of seeds, and whether the error band is SD, SE, CI or min-max ([CASRAI figure-legend guidance](https://casrai.org/guides/figure-legend-example); [NeurIPS 2022 checklist on error bars](https://neurips.cc/Conferences/2022/PaperInformation/PaperChecklist)). Use a colour-blind-safe categorical palette such as Okabe-Ito, with marker or line-style redundancy so plots survive greyscale printing ([viridisLite manual](https://cran.r-project.org/web/packages/viridisLite/refman/viridisLite.html)).

**Algorithms** follow a convention that is recalled, not verified. The float is ruled and titled "Algorithm 1: Name". It has `Input:`/`Output:` lines, numbered lines, server and client procedures ("Server executes:" and "ClientUpdate(k, w)" in the FedAvg mould), the hyperparameters named in the header, and fits in a single column where possible. For named-method papers like every verified FL exemplar, a missing algorithm box reads as an underspecified method.

**Citations and references.** The citation rules are recalled but stable. Citations are numbered in brackets in order of first appearance. `cite` compresses runs to "[2]-[4]", and the bracket follows a space and precedes punctuation. A citation is not used as a grammatical noun, although "in [5]" is acceptable. `IEEEtran.bst` is installed alongside the class ([local bst directory](file:///usr/share/texlive/texmf-dist/bibtex/bst/ieeetran/)). The recalled reference format has these parts:

- Initials before the surname.
- More than six authors collapsed to "et al.".
- The title in quotation marks and in sentence case.
- The journal name abbreviated and italic.
- "vol. X, no. Y, pp. a-b, Mon. Year".
- Conference entries as "in *Proc.* …".
- arXiv entries as "2019, arXiv:1912.04977".
- An optional DOI.

The bibliographic facts of the anchor references are verified. FedAvg appears at AISTATS 2017, PMLR 54:1273-1282 ([PMLR](https://proceedings.mlr.press/v54/mcmahan17a.html)). Kairouz et al. appears in *Found. Trends Mach. Learn.* vol. 14, no. 1-2, pp. 1-210, 2021 ([Now Publishers](https://nowpublishers.com/article/Details/MAL-083)). Sattler et al. appears in *IEEE TNNLS* vol. 31, no. 9, pp. 3400-3413, 2020 ([Korea Univ. record](https://scholar.korea.ac.kr/handle/2021.sw.korea/53658)). Two vehicular references are TIV vol. 9, no. 1, pp. 119-137, 2024 ([Princeton record](https://collaborate.princeton.edu/en/publications/federated-learning-for-connected-and-automated-vehicles-a-survey-/)) and TIV vol. 9, no. 5, pp. 4835-4839 ([IEEE record](https://unpaywall.org/10.1109%2FTIV.2024.3415410)). "Federated Learning on the Road" (arXiv 2102.03401) was published in *IEEE Trans. Wireless Commun.*, not TIV ([Oulu repository](https://urn.fi/URN:NBN:fi-fe202301183455)).

The recalled hallmarks of professional typesetting are as follows:

- No widows or orphans.
- Balanced last-page columns, via `balance` or `\IEEEtriggeratref`.
- A non-breaking space between a number and its unit ("10~ms").
- No overfull boxes from URLs.
- Vector figures with embedded fonts.
- One spelling variant used consistently throughout.

## FL papers present one named idea with scoped, quantified claims

The content conventions are where a draft most visibly reads as professional or amateur, and here the sourcing is stronger because venue checklists state the rules. The NeurIPS checklist's first item requires that the abstract and introduction "accurately reflect the paper's contributions and scope" and that claims "match theoretical and experimental results, and reflect how much the results can be expected to generalize". Other checklist items cover limitations, assumptions and proofs, reproducibility details, error bars and compute. Reviewers are told to reward honest "No/N/A" answers that come with a justification ([NeurIPS 2021 checklist](https://neurips.cc/Conferences/2021/PaperInformation/PaperChecklist); [NeurIPS 2022 checklist](https://neurips.cc/Conferences/2022/PaperInformation/PaperChecklist)). The error-bar guidance asks authors to say which factors vary (seed, split, initialisation), whether the bar shows SD or SE, how it was computed, and to prefer 2-sigma when normality is unverified ([NeurIPS 2022 checklist](https://neurips.cc/Conferences/2022/PaperInformation/PaperChecklist)). These checklists are NeurIPS rules, but IEEE reviewers in this field apply the same expectations. Seed practice in published FL benchmarks ranges from **3 runs (NIID-Bench) to 5 seeds (FLamby)** ([NIID-Bench README](https://github.com/Xtra-Computing/NIID-Bench); [FLamby README](https://github.com/owkin/FLamby)).

**Contributions** sit at the end of the introduction, after the problem and the gap. Simon Peyton Jones's advice is to "nail your contributions" as explicit, refutable claims ([talk notes](https://talks.cam.ac.uk/talk/index/25220/)). FedAvg's introduction states three contributions: the problem framing, the algorithm, and an extensive empirical evaluation ([arXiv 1602.05629](https://arxiv.org/html/1602.05629v4)). IEEE journals typically use a numbered list, whereas ML conferences often use prose; this distinction is recalled. Write each item as one sentence that starts with a strong verb (propose, derive, show, release), points to a section or table, and carries its scope or number. Order the items as idea, then theory, then empirics, then the artefact release.

**Calibrated claims** are scoped by condition rather than inflated. Sattler et al. state where their method wins (non-IID data, small batches, many clients, low participation) and where it still holds (IID data) ([arXiv 1903.02891](https://ar5iv.arxiv.org/html/1903.02891)). FedProx says openly that a zero proximal term and dropping stragglers "can adversely impact convergence behavior" ([arXiv 1812.06127](https://arxiv.org/pdf/1812.06127)). The calibrated form is "improves X by N points on A and B under Dirichlet(0.1)", not "significantly outperforms all methods". FL reviewers also look for conventions on communication cost. A paper should say whether the cost is uplink-only or uplink plus downlink, per client or in total, and how "rounds-to-target" is defined and how runs that never converge are treated ([FedEval](https://arxiv.org/html/2011.09655v1)). Privacy claims should state epsilon, delta, the accountant and the protected unit ([NeurIPS 2020 meta-review](https://papers.neurips.cc/paper/2020/file/4311359ed4969e8401880e3c1836fbe1-MetaReview.html)).

**Results tables** follow a recalled convention. Methods go in rows with the proposed method last. Datasets or heterogeneity levels go in column groups. Each cell reports mean ± spread over the stated seeds with one or two decimals, and the header carries units and direction arrows. Convergence figures plot test accuracy against communication rounds with a shaded band; a min-max band across three runs is one published convention ([Seo et al., arXiv 2502.00182](https://arxiv.org/pdf/2502.00182)). Communication figures plot accuracy against cumulative bytes, because rounds and wall-clock time diverge ([FedEval](https://arxiv.org/html/2011.09655v1)). The recalled section order for an IEEE vehicular journal paper is:

1. Introduction.
2. Related Work, divided into themed subsections, often with a check-mark comparison table.
3. System Model and Problem Formulation, opening with a system-model figure.
4. Proposed Method, with the algorithm box.
5. Analysis.
6. Experiments.
7. Conclusion, in one or two paragraphs.

A regular TIV or TITS paper typically runs to 10-14 pages, but the page limit is unverified.

## The re-typeset draft has the right structure; a few details remain

Checked against these conventions, `paper/ieee/paper.tex` already uses the expected structure:

- `IEEEtran` journal mode with `cite`, `amsmath`, `booktabs`, `algorithm`/`algpseudocode`, `url` and `balance`.
- A drop-cap Introduction.
- An enumerated contribution list at the end of the introduction.
- `[!t]` floats with `figure*`/`table*` for wide items.
- `\appendices`.
- `IEEEtran.bst`.

Its abstract is already calibrated in the style the exemplars model. It names the setting and the method, reports headline numbers against named baselines (24.7% to 82.9%; 88.3% traffic reduction), states the failures (no faster convergence, a worse worst-served vehicle, low mAP) and scopes everything to one synthetic dataset and one seed. The remaining gaps are small, and most rest on recalled rather than verified rules:

- The abstract is about 255 words, slightly above the commonly cited 250-word ceiling.
- The title runs to three lines; Transactions titles are usually shorter.
- One table uses `[!h]` rather than `[!t]`.
- Related Work uses unnumbered run-in `\subsubsection*` heads. The IEEE norm for themed related work is lettered level-2 subsections ("A. Federated optimisation under heterogeneity").
- The "Declarations" block comes after the appendices. IEEE practice places the acknowledgment, including the AI-use disclosure IEEE requires ([IEEE AI-generated text guidelines](https://open.ieee.org/author-guidelines-for-artificial-intelligence-ai-generated-text/)), before the references.
- Biographies are absent. These are normally added only at final acceptance.

## Conclusion

The class fixes almost everything typographic about an IEEE look, and the class file on disk is a primary source. Wherever a recalled rule disagrees with `IEEEtran.cls`, the class wins, as it does on caption alignment and the bottom margin. Values the class leaves to the journal, such as running-head wording, page limits and abstract word caps, should be checked against one recent TIV or TITS PDF before submission, because no vehicular paper was actually read. The parts the class cannot supply are what make a paper read as professional in this field. These are a single named method, a quantified abstract tied to a named baseline, a numbered list of checkable contributions, results reported with stated seeds and spread, and claims scoped to the regime that was measured. Because the draft reports negative results and an audit, it already has the hardest of these to fake, which is calibration. The remaining work is mostly bringing the surrounding structure up to the same standard.
