# Study Design, Data Sourcing, Governance and Data Quality for Patient-Level Research

Research notes compiled 2026-10-08. Search-snippet evidence is marked as such. Several WebFetch attempts failed with DNS errors in this environment (pmc.ncbi.nlm.nih.gov, nature.com, pathnetwork.org, data.ohdsi.org), so primary full texts were not read. Where a claim rests only on a search snippet or on unsourced background knowledge, it is flagged.

## Q1. What design decisions must be justified and pre-specified?

### Takeaway
Reporting guidelines now expect the design, population, exposure, outcome, data source and analysis to be fixed in a protocol before analysis, and they add open-science items (registration, data and code sharing) to the checklist. Observational studies are expected to follow RECORD / RECORD-PE, and prediction-model studies TRIPOD+AI.

### Cited Findings
- SPIRIT 2025 replaces the 2013 protocol guideline. It has a 34-item minimum checklist plus a schedule-of-enrolment diagram, and adds a new open-science section, more emphasis on harms and on describing interventions and comparators, and a patient-and-public-involvement item — [Nature Medicine](https://www.nature.com/articles/s41591-025-03668-w); [PLOS Medicine](https://journals.plos.org/plosmedicine/article?id=10.1371%2Fjournal.pmed.1004589); [EASE summary](https://ease.org.uk/2025/08/spirit-2025-statement-updated-guideline-for-protocols-of-randomized-trials/)
- SPIRIT 2025 is explicitly not a quality-appraisal tool: a poorly designed trial can fully describe its flaws and still satisfy every item — [ScienceDirect (Lancet)](https://www.sciencedirect.com/science/article/pii/S0140673625007706)
- SPIRIT 2025 was published alongside CONSORT 2025 (reporting stage) — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0895435625001258)
- RECORD-PE is a 15-item checklist for pharmacoepidemiology studies using routinely collected health data. It is used alongside RECORD and STROBE, not instead of them. It covers exposure definitions, confounding and data linkage, and is endorsed by ISPE — [BMJ 2018, PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6234471/); [EQUATOR](https://www.equator-network.org/reporting-guidelines/record-pe/)
- A 2023-2024 road-map proposal for peer review of real-world evidence argues that journals should require, not merely recommend, RECORD-PE — [PMC10369122](https://pmc.ncbi.nlm.nih.gov/articles/PMC10369122/)
- TRIPOD+AI (BMJ, April 2024) covers regression and machine-learning prediction models. It has 27 main items and 10 sub-items, and adds patient and public involvement, open-science practices (registration, data and code sharing), and fairness across groups defined by sensitive attributes — [PMC11025451](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11025451/)
- A 2025 J Clin Epidemiol paper presents an updated tool for assessing adherence to TRIPOD+AI — [JCE](https://www.jclinepi.com/article/S0895-4356(25)00451-2/fulltext)
- TRIPOD-LLM extends TRIPOD+AI to large language models in biomedicine — [Nature Medicine](https://www.nature.com/articles/s41591-024-03425-5)
- TRIPOD-Code (protocol stage) targets transparent code availability and computational reproducibility for prediction-model studies — [PMC12888484](https://pmc.ncbi.nlm.nih.gov/articles/PMC12888484/)
- Uptake is slow. An 18-month bibliometric study of orthopaedic AI prediction-model abstracts found no demonstrable improvement in reporting quality after TRIPOD+AI — [PMC12627258](https://pmc.ncbi.nlm.nih.gov/articles/PMC12627258/). A COVID-19 ML prognostic-model review found reporting lagging behind TRIPOD/TRIPOD+AI — [PMC12866346](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12866346/)
- Pharmacoepidemiology-specific RWE guidance addresses the same design points (exposure, comparator, confounding, linkage) in more detail — [PMC13346231](https://pmc.ncbi.nlm.nih.gov/articles/PMC13346231/) (search snippet only)

### Inferences
- A credible patient-level paper should pre-specify, in a dated protocol or registration, at minimum: research question; design (RCT, cohort, case-control, cross-sectional, diagnostic accuracy, prediction model, or federated analysis); source populations and index date; inclusion and exclusion criteria; exposure, outcome and covariate definitions with code lists; follow-up and censoring; missing-data handling; and the statistical analysis plan (SAP) with primary and secondary analyses. Deviations should be labelled as such in the paper.
- The guideline that applies depends on the design: RCT → SPIRIT/CONSORT; observational routinely collected data → RECORD (+RECORD-PE for drug studies); prediction model → TRIPOD+AI (or TRIPOD-LLM if LLMs are involved). Diagnostic accuracy and other designs are covered by guidelines not checked in this session (STARD and similar).
- Reporting guidelines describe what must be reported, not whether the study is good. Pre-registration plus a checklist is therefore necessary but not sufficient evidence of quality.

### Gaps
- Pre-registration registries (OSF, ClinicalTrials.gov, EU PAS register, RWE registries) were not searched in this session, so their current requirements are unverified.
- Guidelines for diagnostic accuracy (STARD 2015), cohort/case-control (STROBE) and cross-sectional studies were not confirmed through search in this session.
- No source was checked for how these requirements differ by region (for example, EU EMA/ENCePP registration versus US practice).

## Q2. What data-quality dimensions must be reported, and how?

### Takeaway
The most-cited harmonised framework for EHR secondary use (Kahn et al., eGEMs 2016) defines shared data-quality terms, and OHDSI operationalises it as parameterised checks run against OMOP CDM databases. Regulators (FDA) frame quality as "fit for use" with emphasis on accuracy, completeness and traceability.

### Cited Findings
- Kahn et al. (2016) produced a harmonised data-quality terminology and framework for secondary use of EHR data by reconciling existing DQ vocabularies, with workshops and interviews, and wrote operational definitions to guide implementation. The authors state it may extend to administrative, research and patient-reported data — [Semantic Scholar](https://www.semanticscholar.org/paper/A-Harmonized-Data-Quality-Assessment-Terminology-of-Kahn-Callahan/ae4bde838e928666f88a8f6dbb29c236227607a2) (abstract text via search; full text not read)
- Testing of the Kahn framework in a secondary-use data warehouse found that revisions were needed, and a multi-level checklist was proposed — [eGEMs 10.5334/egems.298](https://dx.doi.org/10.5334/egems.298)
- A 2021 BMC Med Res Methodol paper provides a harmonised DQ framework for observational research data with software implementations in R — [BMC](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-021-01252-7)
- The European Health Data Space DQ framework (TEHDAS, 2022) uses Kahn as a base and adds categories for research-question context — [TEHDAS deliverable 6.1](https://tehdas.eu/app/uploads/2022/05/tehdas-european-health-data-space-data-quality-framework-2022-05-18.pdf) (search snippet)
- OHDSI Data Quality Dashboard (DQD) organises its checks by the Kahn framework. It applies 24 parameterised check types to an OMOP CDM instance, producing about 4,000 checks per GitHub README (the OHDSI CDM site cites more than 3,500; counts differ by version). Each check is evaluated against a pre-specified threshold, and the CDM_SOURCE table must be populated first — [OHDSI DQD GitHub](https://github.com/OHDSI/DataQualityDashboard); [OHDSI CDM docs](https://ohdsi.github.io/CommonDataModel/)
- The OHDSI CDM site lists CDM v5.5 as current and says DQD is required before participating in an OHDSI network study. It lists DQD as legacy-supported for v5.5, not feature-supported — [OHDSI CDM docs](https://ohdsi.github.io/CommonDataModel/) (search snippet; a third-party checklist calls v5.4 current, so the official site should be preferred)
- Achilles is the companion tool for descriptive characterisation; published work has used it for ETL validation by comparing source-system and CDM results — [JMIR Med Inform 2024](https://medinform.jmir.org/2024/1/e49542)
- FDA's drug/biologic RWE guidance (final, July 2024) names data reliability as the foundation, with accuracy, completeness and traceability as its dimensions. Sponsors should assess quality across capture, curation and transformation stages, consider bias from collection, and use risk-based quality management — [FDA guidance PDF](https://www.fda.gov/media/171667/download) and law-firm summaries (search snippets; FDA text not read in full)
- For devices, FDA's final guidance uses "relevance and reliability" instead of "fit for purpose" — [Federal Register draft, Dec 2023](https://www.federalregister.gov/documents/2023/12/19/2023-27852/use-of-real-world-evidence-to-support-regulatory-decision-making-for-medical-devices-draft-guidance) (draft text; final wording via search summary only)

### Inferences
- Dimensions that the sources support as reportable: conformance (format and value-set agreement with the model), completeness (missingness of required fields and expected records), plausibility (values within clinical or temporal bounds), accuracy/validity (agreement with source or chart review), and traceability (provenance of each transformation). Timeliness, which the task brief names, is not explicitly confirmed in the Kahn sources checked here; it should be reported as a dimension the team chose to add.
- Practical reporting: for each dimension, state the check, the threshold, the result per site or per data source, and the action taken for failures. DQD outputs (pass/fail by check and by threshold) can serve as the audit trail. Sites that fail thresholds should be documented, not silently excluded.
- Source-to-CDM reconciliation (counts and key clinical measures compared before and after ETL) is the standard check for accuracy at the transformation step.

### Gaps
- Full text of Kahn 2016 was not read (PMC and PaTH fetches failed), so the exact category and context names are unverified in this session. Background knowledge suggests three categories (conformance, completeness, plausibility) and three contexts (verification, validation, attestation); this needs confirmation.
- No source confirming how "timeliness" is defined in current DQ frameworks was found.
- Recommended numeric thresholds for completeness and plausibility were not found in sources; they are study-specific.

## Q3. Requirements for multi-site and federated studies

### Takeaway
Federated and multi-site designs need site-level agreements and governance, a shared data model and common analysis code, and explicit reporting of heterogeneity across sites. Governance is less settled than the technical methods, and the literature found here is mostly 2020 onward.

### Cited Findings
- A 2025 npj Digital Medicine scoping review (Eden et al., 39 articles) built a framework of federated-learning governance mechanisms: 12 procedural, 10 relational and 12 structural — [PubMed 40640574](https://pubmed.ncbi.nlm.nih.gov/40640574/); [npj Digit Med 2025](https://www.nature.com/articles/s41746-025-01836-3) (full text not fetched)
- A related review in the journal Healthcare found governance frameworks for healthcare FL remain underdeveloped. Of 38 included studies, 16 discussed governance mechanisms, with substantial heterogeneity in components and depth — [MDPI Healthcare 2025](https://www.mdpi.com/2227-9032/14/3/306) (search snippet)
- Rieke et al. (npj Digit Med 2020) state that in FL each data controller sets its own governance processes and privacy policies, and that data heterogeneity can mean the globally optimal model is not optimal for individual sites — [ResearchGate abstract](https://www.researchgate.net/publication/344709253_The_future_of_digital_health_with_federated_learning)
- A 2025 npj Digit Med paper (FedWeight) addresses cross-site distribution shift in EHR data by re-weighting patients. It was seen only via a secondary reference list — no direct URL verified
- A 2024 arXiv review lists pitfalls and recommendations for implementing FL in healthcare — [arXiv 2409.09727](https://arxiv.org/html/2409.09727v2) (not read in full)
- A 2024 PHG Foundation report describes federating two UK trusted research environments (Genomics England and NIHR Cambridge BRC) so analysis runs across both without moving the data. It notes added complexity: researchers may need to negotiate access separately with each institution — [PHG Foundation 2024](https://www.phgfoundation.org/wp-content/uploads/2024/11/The-federation-of-trusted-research-environments-for-genomics-and-health-.pdf)
- Output review (the "airlock" check of results for re-identification risk) is a recurring bottleneck in federated TRE work; the claim that manual review is slow and inconsistent comes from a vendor blog and is not independent evidence — [Lifebit blog](https://lifebit.ai/blog/federated-data-analysis-healthcare/)
- A validation study compared federated and pooled analysis on synthetic asthma data across TREs, testing whether GLM and ML results change — [PubMed 42630628](https://pubmed.ncbi.nlm.nih.gov/42630628/) (search snippet; results not extracted)
- A 2024 study (ICODA/SAIL) reports federated networks lower disclosure risk because researchers analyse data where it sits — [search snippet via Aridhia](https://www.aridhia.com/data-federation/) (vendor page; treat as a claim to verify)

### Inferences
- Minimum multi-site documentation: a data-sharing or site agreement per site; a named governance body (or data access committee) and its approval record; a common data model and version used at each site (for example OMOP CDM version and DQD results per site); a shared, versioned analysis protocol and code; and a per-site table of cohort sizes and data-quality results.
- Heterogeneity reporting: report site-level estimates alongside pooled estimates, a heterogeneity statistic (for example I² or tau²) with its limits, and sensitivity analyses excluding outlier sites. For federated learning, report the number of communication rounds, local versus global performance, and how non-identical data distributions were handled.
- Federated analysis reduces data transfer but does not remove the need for a DUA, output checking, or the risk of disclosure from aggregate results (for example, small cell counts in shared summaries).

### Gaps
- No review specifically on heterogeneity in healthcare FL governance was found; the FedWeight and heterogeneity points come from secondary references or general reviews.
- Standard templates for site agreements and multi-site SAP content were not located.
- Results of the federated-versus-pooled validation study were not extracted.

## Q4. Data governance steps before analysis

### Takeaway
Current practice before analysis is a formal data-access request, a DUA or site agreement, ethics or IRB approval (or a documented waiver), and access inside a controlled environment where outputs are checked before release. Trusted research environments (TREs) are the main model promoted in the UK and increasingly elsewhere.

### Cited Findings
- The Goldacre Review (2022, UK) is cited as the origin of the TRE model. It moved health-data sharing from direct transfer toward strictly controlled remote access — [PHG Foundation 2024](https://www.phgfoundation.org/wp-content/uploads/2024/11/The-federation-of-trusted-research-environments-for-genomics-and-health-.pdf) (secondary citation; Goldacre review not fetched)
- Under HIPAA (US), de-identification is achieved by Safe Harbor (removal of 18 identifier categories, with no actual knowledge that remaining data could identify a person) or by Expert Determination (documented statistical assessment that risk is "very small" for the intended recipients). The regulation does not define "very small" or the expert's qualifications. A DUA may be used with either method but cannot substitute for them — [HHS OCR guidance](https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html); [CASRAI list of 18 identifiers](https://casrai.org/guides/18-hipaa-identifiers) (summaries; HHS text not fully fetched)
- NIH DMS Policy (effective 25 January 2023) requires a data management and sharing plan with applications that generate scientific data, and says consent is still required for human-subjects research even when data are de-identified — [NIH sourcebook](https://oir.nih.gov/sourcebook/intramural-program-oversight/intramural-data-sharing/2023-nih-data-management-sharing-policy); [Penn summary](https://www.med.upenn.edu/evdresearch/nih-data-management/) (secondary summary)
- A secondary source says NIH introduced a new DMS plan template in 2026 (standardised questionnaire and data table as of 25 May 2026). This is not confirmed against NIH's own page and should be checked before use
- Wellcome's Data, software and materials management and sharing policy asks for maximum availability with as few restrictions as possible, with data underpinning papers available at publication. It also asks applicants to manage the risk of misuse, and to use "as open as possible, as closed as necessary" — [Wellcome policy](https://wellcome.org/research-funding/guidance/policies-grant-conditions/data-software-materials-management-and-sharing-policy); [Wellcome outputs management plan](https://wellcome.org/grant-funding/guidance/how-complete-outputs-management-plan)
- The EU Health Data Space and FDA RWE guidance require traceability and accountability of data sources, consistent with the governance steps above — see Q2 sources

### Inferences
- Typical pre-analysis sequence: (1) study protocol and registration; (2) ethics/IRB approval or waiver; (3) data-access request to each custodian; (4) DUA or site agreement, specifying permitted purposes, no re-identification, retention and deletion, and output rules; (5) de-identification or pseudonymisation documented against a stated standard (HIPAA Safe Harbor or Expert Determination in the US; GDPR pseudonymisation and anonymisation tests in the EU, not checked in this session); (6) access inside a TRE or federated node with logging; (7) output checking before release.
- Pseudonymised data remain personal data under GDPR. This is background knowledge and not sourced here.
- Access committees and TRE output-checking processes differ by country and by data custodian, so the paper should name the specific process used.

### Gaps
- GDPR, UK GDPR and EU EHDS requirements were not verified through search in this session.
- No source on current access-committee timelines or standard DUA clauses was found.
- Trusted research environment operating standards (for example, the UK Five Safes framework) were not checked directly.

## Q5. Accepted practices for data and code sharing compatible with privacy

### Takeaway
Funders now expect data and code to be shared as far as privacy allows, with a documented justification for any restriction. The usual compatible routes are controlled access (managed access via a repository or committee), synthetic data, and federated or remote analysis with shared code. Synthetic data and differential privacy are promising but are not a guarantee of privacy.

### Cited Findings
- Wellcome: data underpinning papers must be available at publication, with the minimum software needed to view or replicate the analysis; data must be shared unless commercial, privacy, security or confidentiality concerns apply — [Wellcome policy](https://wellcome.org/research-funding/guidance/policies-grant-conditions/data-software-materials-management-and-sharing-policy)
- NIH: scientific data supporting a publication must be shared by the time of publication (per the 2023 policy summary) — [Penn summary](https://www.med.upenn.edu/evdresearch/nih-data-management/)
- TRIPOD+AI and SPIRIT 2025 both add data and code sharing and registration to their checklists — see Q1 sources
- Synthetic data built from real patients can overfit (reproduce actual records) or be exposed to membership inference and linkage attacks. No widely accepted standards yet exist for generating or evaluating synthetic health data — [arXiv 2501.09031](https://arxiv.org/pdf/2501.09031)
- A 2024 arXiv paper proposes practical privacy metrics for synthetic data — [arXiv 2406.16826](https://arxiv.org/pdf/2406.16826)
- Differentially private synthetic data has been shown to give high utility with formal guarantees in a 2025 behavioural-health study. The snippet did not report the specific utility or privacy figures — [arXiv 2507.02971](https://arxiv.org/html/2507.02971v1)
- Singapore's PDPC proposed a step-by-step synthetic data guide (July 2024) covering re-identification risk, residual risk and ongoing monitoring — [PDPC] (search snippet, no direct URL verified)
- A 2025 scoping review of privacy and utility metrics in medical synthetic data (Kaabachi et al., npj Digit Med 8:60) is the most relevant review found, but only its bibliography entry was seen — no URL verified

### Inferences
- k-anonymity alone is a weak protection for patient-level data: the snippets reviewed do not quantify this, so the point rests on background knowledge (homogeneity and background-knowledge attacks; Sweeney 2002). It should be stated as such, with a citation to be added.
- Differential privacy provides a formal bound on what any single record can add to an output, but it reduces utility and requires an epsilon budget to be reported. Papers should report the mechanism, epsilon, and how it was chosen.
- Secure aggregation and federated analysis protect individual updates from the coordinator, but they do not protect against disclosure from the released model or aggregate statistics. Output checking is still needed.
- A data-sharing statement should state: which data are shared (full, de-identified, synthetic, aggregate or none); the access route and who decides; the legal or ethical reason for any restriction; the code repository and version; and the funder or journal policy being followed.

### Gaps
- No source from 2026 was found that sets current norms for k-anonymity thresholds or differential-privacy epsilon values in health research.
- Re-identification risk studies for genomic and wearable data were not searched.
- Journal-specific data-availability policies (for example, the BMJ or PLOS requirements) were not checked.

## Cross-cutting notes

### Regional differences
- US: HIPAA de-identification (Safe Harbor / Expert Determination); NIH DMS Policy for funded work.
- UK: TRE model after the Goldacre Review; Wellcome (UK-based) policy applies to its funded work globally.
- EU: EU Health Data Space framework; GDPR pseudonymisation rules; EMA and ENCePP guidance for pharmacoepidemiology (not verified in this session).
- Regulators: FDA drug RWE guidance (2024) and device RWE guidance use different terminology ("fit for use" versus "relevance and reliability").

### Source-quality caveats
- Several sources are vendor pages (Lifebit, Aridhia, BC Platforms) and were used only to show positioning, not as evidence.
- Several claims rest on search snippets and secondary summaries rather than the primary documents. The primary texts to check before citing are: Kahn et al. 2016 (eGEMs), Eden et al. 2025 (npj Digit Med), FDA RWE guidance, NIH DMS policy page, and the Kaabachi et al. 2025 review.
- Fetch attempts failed for pmc.ncbi.nlm.nih.gov, nature.com, pathnetwork.org and data.ohdsi.org in this environment. Those primary sources need re-checking.

## Sources list (for citation check)
- SPIRIT 2025: https://www.nature.com/articles/s41591-025-03668-w
- RECORD-PE: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6234471/
- TRIPOD+AI: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11025451/
- Kahn 2016 (abstract): https://www.semanticscholar.org/paper/A-Harmonized-Data-Quality-Assessment-Terminology-of-Kahn-Callahan/ae4bde838e928666f88a8f6dbb29c236227607a2
- OHDSI DQD: https://github.com/OHDSI/DataQualityDashboard
- OMOP CDM: https://ohdsi.github.io/CommonDataModel/
- FDA RWD/RWE drug guidance: https://www.fda.gov/media/171667/download
- FL governance scoping review: https://pubmed.ncbi.nlm.nih.gov/40640574/
- PHG Foundation TRE federation: https://www.phgfoundation.org/wp-content/uploads/2024/11/The-federation-of-trusted-research-environments-for-genomics-and-health-.pdf
- HHS de-identification: https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html
- NIH DMS policy: https://oir.nih.gov/sourcebook/intramural-program-oversight/intramural-data-sharing/2023-nih-data-management-sharing-policy
- Wellcome data policy: https://wellcome.org/research-funding/guidance/policies-grant-conditions/data-software-materials-management-and-sharing-policy
- Synthetic data and health privacy: https://arxiv.org/pdf/2501.09031
