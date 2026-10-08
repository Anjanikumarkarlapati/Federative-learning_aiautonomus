# Statistical and Methodological Requirements for Patient-Level Research Papers

Status note: WebFetch and curl could not reach the primary journal and PMC pages in this session (DNS failure and proxy 403 on CONNECT). Everything below comes from search-result snippets and abstracts, not full texts. Items marked "unverified" are background recollections that were not confirmed against a source in this session and must be checked before citing. Search date context: today is 2026-10-08.

## Q1. What are the minimum statistical items for each major design?

### Takeaway
For prediction-model studies the governing reporting standard is TRIPOD+AI (2024, 27-item checklist), which supersedes TRIPOD 2015 and covers both regression and machine-learning models. For causal observational designs the 2025 TARGET statement (21 items) applies when a study emulates a target trial. For methods reviewers, the baseline is: a pre-specified analysis plan, a justified sample size, handling of missing data, confounding control stated against a causal question, and effect sizes with confidence intervals.

### Cited Findings
- TRIPOD+AI was published in BMJ in April 2024 (Collins et al., doi 10.1136/bmj-2023-078378). It applies to prediction models built with regression or machine learning. The 27-item checklist supersedes the TRIPOD 2015 checklist, which should no longer be used. A TRIPOD+AI for Abstracts checklist is also provided. — [TRIPOD+AI statement, PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11025451/)
- A blog summary claims that late-2025 studies found only about 28% of ML prediction-model papers met TRIPOD+AI. I could not verify this against a primary study. — [journalmetrics summary](https://www.journalmetrics.org/blog/tripod-ai-clinical-prediction-model-reporting) (unverified figure)
- TARGET (2025) is a reporting guideline for observational studies that emulate a parallel-group, individually randomised target trial with adjustment for baseline confounders. It has 21 items in six sections (abstract, introduction, methods, results, discussion, other). Authors must flag the study as an emulation, state the causal question, specify the target-trial protocol and how it maps to the data, report each causal estimand with its precision, and give sensitivity analyses for assumptions and design or analysis choices. Published in JAMA (3 Sept 2025) and BMJ 390:e087179. — [EQUATOR entry](https://www.equator-network.org/reporting-guidelines/22462/); [Southampton eprint](https://eprints.soton.ac.uk/505877)
- PROBAST+AI (BMJ 2025;388:e082505) is the updated risk-of-bias tool for both regression and AI prediction models. It has four domains (participants and data sources, predictors, outcome, analysis). Model development uses 16 signalling questions and model evaluation uses 18. It may replace PROBAST (2019). — [ORA Oxford record](https://ora.ox.ac.uk/objects/uuid:8b15c087-459b-43fa-ad03-1f66ed6ba370)
- Sterne et al. (BMJ 2009;338:b2393) is an educational article on multiple imputation. EQUATOR lists it as a reporting guideline under statistical methods. It argues that complete-case analysis is not adequate by default. — [Sterne et al. DOI](https://doi.org/10.1136%2Fbmj.b2393); [PMC2714692](https://pmc.ncbi.nlm.nih.gov/articles/2714692/)
- STRATOS is an international expert initiative (launched 2013, ISCB) producing guidance on observational-study design and analysis. Its topic groups include missing data, initial data analysis, model selection, measurement error, study design, prediction models, causal inference, survival analysis, and high-dimensional data. Overview paper: Sauerbrei et al., Stat Med 2014;33:5413-5432, doi 10.1002/sim.6265. — [EMWA summary](https://emwa.org/journal/observational-studies/guidance-for-the-design-and-analysis-of-observational-studies-the-strengthening-analytical-thinking-for-observational-studies-stratos-initiative/)

### Inferences
- A paper that develops or validates a prediction model should cite TRIPOD+AI (not TRIPOD 2015) and assess its own risk of bias with PROBAST+AI. A reviewer will likely check both.
- A causal observational study with a treatment effect question should be framed as a target-trial emulation and cite TARGET if it follows that structure.
- Sterne et al. and the EQUATOR listing are the usual citations for missing-data handling, but no formal checklist item list was retrieved.

### Gaps
- Exact minimum items per design (RCT-like, cohort, survival, diagnostic accuracy) were not retrieved. Check the STROBE, CONSORT, and STARD checklists directly for their statistical items.
- The full TRIPOD+AI item wording on sample size, missing data, fairness, and ML leakage was not retrieved (the fetch failed). Use the PMC link above.

---

## Q2. What sample-size and events-per-variable rules are current for prediction models?

### Takeaway
Riley et al. replace fixed events-per-variable (EPV) rules of thumb with criteria-based sample sizes. For continuous outcomes there are four criteria (shrinkage ≥0.9, R² gap ≤0.05, residual SD precision, intercept precision). For binary and time-to-event outcomes there are three criteria (shrinkage ≥0.9, Nagelkerke R² change ≤0.05, precision of overall risk). The required n is the largest value across criteria. The authors advise against the 10-EPP rule.

### Cited Findings
- Continuous outcome, four criteria: (1) global shrinkage factor ≥0.9 (small optimism in predictor effects); (2) apparent vs adjusted R² absolute difference ≤0.05; (3) residual SD estimated with a margin of error ≤10% of the true value; (4) intercept (mean predicted outcome) estimated precisely. Worked example: 25 predictors needed at least 918 subjects (about 36.7 per parameter). — [Riley et al., Part I, continuous](https://dspace.library.uu.nl/handle/1874/386405)
- Binary and survival outcomes (Part II): three criteria, shrinkage ≥0.9, change in Nagelkerke R² ≤0.05, and precise estimation of overall risk. Criteria are expressed in EPP and the authors say rules of thumb such as 10 EPP should be avoided. — [Riley et al., Part II, binary and time-to-event](https://utrecht-main-test.atmire.com/items/0ff8b358-9ca8-4b96-9b4f-247b8df46220)
- Riley et al. (BMJ 2020, "Calculating the sample size required for developing a clinical prediction model") is the primary source for the criteria. The search returned the pmsampsize package, which implements the criteria. — [pmsampsize refman](https://cran.nics.utk.edu/cran/web/packages/pmsampsize/refman/pmsampsize.html)
- pmsampsize: type="b" for binary outcomes; inputs include Cox-Snell R² (csrsquared), outcome prevalence, and a shrinkage target (default 0.9), plus the number of candidate parameters. Example prevalence used in the package: 0.174. — [pmsampsize on rdrr.io](https://rdrr.io/pkg/pmsampsize/man/pmsampsize.html)
- Where a paper reports only a C-statistic, a separate Riley et al. note (Stat Med 2020) describes estimating Cox-Snell R² from the C-statistic and prevalence for the sample-size calculation. — [pmsampsize refman](https://cran.nics.utk.edu/cran/web/packages/pmsampsize/refman/pmsampsize.html)
- External validation sample size: Riley et al. (Stat Med 2021, doi 10.1002/sim.9025) is implemented in pmvalsampsize. — [pmvalsampsize refman](https://cran.nics.utk.edu/cran/web/packages/pmvalsampsize/refman/pmvalsampsize.html)
- Calibration needs large samples at validation. Van Calster et al. (2019) state that calibration curves at external validation need sufficiently large samples and that model complexity must be balanced against sample size. — [Van Calster et al., BMC Medicine 2019](https://bmcmedicine.biomedcentral.com/track/pdf/10.1186/s12916-019-1466-7)

### Inferences
- A reviewer will expect a stated sample size calculation for model development, using pmsampsize or equivalent, with the anticipated R² or C-statistic, prevalence, number of candidate parameters, and the criterion that binds.
- Reporting the final number of events and EPP (candidate parameters, not final parameters) is the transparent option.
- For ML models, the candidate parameter count is harder to define. The criteria still apply to the number of parameters the model can use, which needs explicit justification.

### Gaps
- The exact Riley binary criteria formulas and worked examples were not fetched. Verify against the BMJ 2020 paper.
- The sample-size treatment of ML models (where effective degrees of freedom are unclear) was not found in these sources.
- The rule that the 10-EPP rule is inadequate is established in the Riley papers, but the fetch failed, so only the search snippet is available.

---

## Q3. How should missing data and confounding be handled and reported?

### Takeaway
Multiple imputation (MI) is the default for missing data under MAR, with reporting of the imputation model and the number of imputations. Complete-case analysis is discouraged as the default. For confounding, the sources found here are mostly about study design (target-trial emulation) and reporting. Specific guidance on DAGs, propensity scores, and IV methods was not retrieved.

### Cited Findings
- Sterne et al. (2009): missing values can undermine validity; MI is practical in standard software; complete-case analyses can be misleading and inefficient, and should not be treated as adequate by default. — [Sterne et al., BMJ 2009](https://doi.org/10.1136%2Fbmj.b2393)
- TARGET requires authors to state the target-trial protocol, map it onto the observational data, report each causal estimand with its precision, and report sensitivity analyses for assumptions and design or analysis choices. — [TARGET EQUATOR entry](https://www.equator-network.org/reporting-guidelines/22462/)
- TRIPOD+AI covers missing data and predictor handling within its 27 items (wording not retrieved). — [TRIPOD+AI, PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11025451/)

### Inferences
- Missing data reporting should cover: the amount and pattern of missingness per variable, the assumed mechanism (MCAR/MAR/MNAR, with justification), the imputation method and model, the number of imputations, pooling rules (Rubin's rules), and a sensitivity analysis for MNAR where plausible.
- For confounding, a DAG or explicit causal assumptions should appear in the methods, consistent with the TARGET requirement to state the causal question.

### Gaps (unverified background, check before citing)
- MCAR/MAR/MNAR definitions (Rubin 1976) are standard but were not retrieved from a source here.
- Number of imputations: the common recommendation (White, Royston & Wood 2011) is to relate m to the percentage of missing information. This needs verification.
- DAG guidance (Pearl; Textor et al. 2016 on DAGitty), propensity-score diagnostics (standardised mean differences below 0.1 are a common threshold), and IV assumptions (relevance, exclusion, independence) were not retrieved from sources in this session.
- The E-value (VanderWeele & Ding 2017) as a sensitivity analysis for unmeasured confounding is not sourced here.

---

## Q4. What validation and calibration reporting is expected for prediction and ML models?

### Takeaway
A prediction-model paper should report discrimination and calibration, internal validation with optimism correction (e.g., bootstrap), and external or temporal validation with the same metrics. Calibration should be assessed at least at the moderate level (flexible calibration curve). Decision curve analysis (net benefit) is recommended by TRIPOD for models intended for clinical decisions. Net-benefit confidence intervals are a debated area.

### Cited Findings
- Van Calster et al. (2016, J Clin Epidemiol): four calibration levels (mean, weak via intercept and slope, moderate via flexible calibration curve, strong per covariate pattern). The authors recommend focusing on moderate calibration, and argue that moderate calibration is sufficient to avoid harm in decision-curve terms. Reliability of flexible calibration plots depends strongly on validation sample size. — [Search hit, KU Leuven repository](https://lirias.kuleuven.be/retrieve/577522) (abstract-level, not fetched)
- Van Calster et al. (2019, BMC Medicine, "Calibration: the Achilles heel of predictive analytics"): calibration deserves more attention; model complexity must be balanced against sample size; updating should be considered. — [BMC Medicine 2019](https://bmcmedicine.biomedcentral.com/track/pdf/10.1186/s12916-019-1466-7)
- Clustered flexible calibration (Barreñada et al., 2025): for multi-cluster data, two-stage meta-analysis with splines for the average curve with a 95% prediction interval, and mixed-model calibration for cluster-specific curves. Code is provided. — [Cambridge Research Synthesis Methods](https://www.cambridge.org/core/journals/research-synthesis-methods/article/clustered-flexible-calibration-plots-for-binary-outcomes-using-random-effects-modeling/FE857DCCAD4A37B60EF819F26B2F9411)
- Decision curve analysis: net benefit = sensitivity × prevalence − (1 − specificity) × (1 − prevalence) × w, where w = pt/(1 − pt). Compared with treat-all and treat-none across a range of thresholds. Net benefit captures decision consequences that discrimination and calibration do not; an accurate model can still have poor net benefit. — [Vickers & Elkin tutorial, PMC6777022](https://pmc.ncbi.nlm.nih.gov/articles/6777022)
- Net benefit inference: a paper with Van Calster, Wynants and Steyerberg argues that requiring statistical significance for net-benefit differences would be harmful, and suggests value-of-information analysis instead. The paper's identity is not confirmed. A related paper on CIs and hypothesis testing for net benefit exists. — [Vickers et al., CIs for net benefit](https://synapse.mskcc.org/synapse/works/216480)
- TRIPOD recommends decision curve analysis for prediction models. — [Search summary citing TRIPOD](https://pmc.ncbi.nlm.nih.gov/articles/PMC10454914) (secondary)
- Wynants et al. (BMJ 2020): the original review screened 2696 titles and included 27 studies with 31 COVID-19 models. Risk of bias was judged with PROBAST, and reported performance was probably optimistic. The living review later included 412 studies describing 731 models or validations. Of the prognostic models in the living review, 29 were low, 32 unclear, and 545 high risk of bias. Counts differ between versions, so check which one a citation requires. — [Wynants et al., BMJ 2020](https://bmj.com/content/369/bmj.m1328)

### Inferences
- A reviewer will check: discrimination (C-statistic with CI), calibration (intercept and slope, plus a flexible calibration plot), overall performance (Brier score or R²), optimism-corrected estimates from bootstrap internal validation, and a clear statement of temporal or external validation with its sample size.
- Decision curve analysis should be shown for the clinical use case with a justified threshold range. Its CIs should be reported with caution given the net-benefit inference debate.
- Model updating (recalibration) should be reported separately from validation.

### Gaps
- Specific bootstrap settings (number of resamples, typically 200 or more) and optimism-correction steps were not retrieved from a source.
- Specific thresholds for the calibration slope and intercept in acceptable ranges were not retrieved.
- Fairness and subgroup performance reporting (TRIPOD+AI items) were not retrieved.

---

## Q5. What are the common reasons statistical reviews reject papers, and how are they avoided?

### Takeaway
The main documented failures are poor reporting of the sample size and missing data, no external or temporal validation, no calibration assessment, high risk of bias (per PROBAST), and inflated performance from leakage or optimism. These are all checkable against TRIPOD+AI and PROBAST+AI.

### Cited Findings
- Wynants et al. (BMJ 2020) judged published COVID-19 prediction models poorly reported and at high risk of bias, with probably optimistic reported performance. None were recommended for practice. — [Wynants et al., BMJ 2020](https://bmj.com/content/369/bmj.m1328)
- TRIPOD+AI and PROBAST+AI are the current reporting and bias tools. — [TRIPOD+AI](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11025451/); [PROBAST+AI](https://ora.ox.ac.uk/objects/uuid:8b15c087-459b-43fa-ad03-1f66ed6ba370)
- Calibration at external validation requires large samples; reliability depends on n. — [Van Calster et al. 2016](https://lirias.kuleuven.be/retrieve/577522); [Van Calster et al. 2019](https://bmcmedicine.biomedcentral.com/track/pdf/10.1186/s12916-019-1466-7)

### Inferences
- Likely rejection grounds, to check against reviewer guidance: no pre-specified analysis plan; sample size not justified; EPV rule used instead of Riley criteria; complete-case analysis without justification; no calibration; performance reported only on the development data; data leakage (preprocessing or feature selection done before splitting); tuning on the test set; p-values without effect sizes or CIs; no subgroup or fairness check where equity is claimed; and an ML paper without a model card or code/data availability statement.

### Gaps
- Journal-specific statistical reviewer guidance (BMJ, Lancet, JAMA) was not retrieved. Those pages were not reachable from this session.
- Specific rejection data for ML leakage and fairness were not found.

---

## Items to verify (unverified background, not sourced in this session)

These are commonly cited but were not confirmed by a retrieved source. Do not cite them without checking:
- Multiplicity: Bonferroni, Holm, or a pre-specified primary endpoint.
- Subgroup analysis: interaction tests rather than separate subgroup p-values.
- Survival analysis: proportional-hazards checks (Schoenfeld residuals); competing risks (Fine-Gray or cause-specific hazards); censoring assumptions.
- Model cards (Mitchell et al. 2019) and CONSORT-AI / SPIRIT-AI (2020) for ML trials.
- Hyperparameter tuning must use nested or training-only CV to avoid optimistic bias.
- Bootstrap internal validation (typically 200 or more resamples) for optimism correction.

## Priority next steps for whoever writes the paper
1. Read TRIPOD+AI (PMC link above) and tick every checklist item.
2. Read Riley et al. BMJ 2020 and the Part I and II papers, then run pmsampsize.
3. Check PROBAST+AI signalling questions against the study design.
4. Verify the Sterne/White MI reporting recommendations and the Van Calster 2016 calibration paper directly.
5. Obtain journal-specific statistical guidance (BMJ, Lancet, JAMA) from the journal author instructions.
