# Reporting Guidelines, Ethics and Consent for Patient-Level Research Papers (as of Oct 2026)

Status: partial. Direct fetches of hhs.gov, ecfr.gov and eur-lex.europa.eu failed (DNS blocked in sandbox), so HIPAA, GDPR and Common Rule points rely on search excerpts and secondary commentary. Items marked UNVERIFIED are from general knowledge and were not confirmed.
(Saved by the coordinator from the subagent's returned text, because the subagent's direct write was refused.)

## Q1. Which reporting guideline applies to which design?
Takeaway: RCT = CONSORT 2025 (protocol: SPIRIT 2025). Cohort, case-control, cross-sectional = STROBE. Prediction model development/validation (regression or ML) = TRIPOD+AI. Diagnostic accuracy = STARD. Case report / small case series = CARE. Economic evaluation = CHEERS. Systematic review = PRISMA. AI trials = CONSORT-AI / SPIRIT-AI. Qualitative = COREQ or SRQR (UNVERIFIED).

- CONSORT 2025: 30-item checklist plus participant flow diagram. Published April 2025 (BMJ 2025;389:e081123; also PLoS Med, Nature Med). Supersedes CONSORT 2010. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11995449/ ; https://pubmed.ncbi.nlm.nih.gov/40228477/
- SPIRIT 2025: 34 minimum items, schedule of enrolment/interventions/assessments diagram; new items on open science and patient/public involvement. Supersedes SPIRIT 2013. Primary journal attribution contested (one source JAMA doi 10.1001/jama.2025.4486); confirm. https://pubmed.ncbi.nlm.nih.gov/40294593/
- TRIPOD+AI: BMJ 2024;385:e078378 (16 April 2024). Supersedes TRIPOD 2015. 27-item checklist plus TRIPOD+AI for Abstracts. Covers regression and ML; development, validation or both. Fairness is a headline theme. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11025451/
- Fairness-related TRIPOD+AI items (third-party copy, not official text): 3c (known health inequities in background), 25 (interpretation incl. fairness), 26 (limitations incl. bias, generalisability).
- CONSORT-AI: 14 added items. SPIRIT-AI: 15 added items. Cover AI intervention description, setting, input/output handling, human-AI interaction. CONSORT-AI: Lancet Digit Health 2020;2(10):e537–e548. SPIRIT-AI: BMJ 2020;370:m3210.
- STARD 2015: 30 essential items; participant flow diagram expected. https://pmc.ncbi.nlm.nih.gov/articles/PMC4623764
- CHEERS 2022: 28 items; replaces CHEERS 2013 (24 items).
- STROBE: 22 items; 18 common to all designs, 4 design-specific. 2007 checklist remains the base (secondary: casrai.org).
- CARE 2013: 13 items (title, key words, abstract, introduction, patient information, clinical findings, timeline, diagnostic assessment, therapeutic interventions, follow-up and outcomes, discussion, patient perspective, informed consent).
- UNVERIFIED: PRISMA 2020 (27 items + flow diagram), COREQ/SRQR.
Inferences: a 2026 ML prediction paper should cite TRIPOD+AI, not TRIPOD 2015. A 2026 RCT should cite CONSORT 2025.
Gaps: official numbered item wording for TRIPOD+AI, CONSORT 2025, SPIRIT 2025 not retrieved.

## Q2. Are checklists required at journal submission?
Takeaway: ICMJE does not mandate a specific checklist, but most ICMJE-following journals ask for the relevant EQUATOR checklist. Lancet, NEJM, BMJ, JAMA author instructions were NOT checked.
- ICMJE: trial registration in a public registry at or before first patient enrolment is a condition for consideration for publication. Acceptable: a WHO ICTRP primary registry with the 24-item minimum dataset, or ClinicalTrials.gov. Ethics approval does not count. Observational studies generally fall outside the ICMJE trial definition. https://icmje.org/recommendations/browse/publishing-and-editorial-issues/clinical-trial-registration.html
- ICMJE data-sharing (2017 proposal): describe a data-sharing plan in the manuscript.
- UNVERIFIED pattern: BMJ and JAMA ask for EQUATOR checklists; Lancet and NEJM ask for registration and data-availability statements.

## Q3. Ethics approval, exemption, consent
Takeaway: Declaration of Helsinki (revised 19 Oct 2024) requires ethics committee review and consent as an ongoing process, with extra protection for vulnerable groups. US Common Rule waivers require documented criteria. A consent waiver does not waive HIPAA.
- Helsinki 2024: adopted by WMA General Assembly, 19 Oct 2024. Themes: vulnerable populations, trial transparency, fairness and equity, consent as ongoing process, research integrity, data protection and AI. https://www.wma.net/news-post/revised-declaration-of-helsinki-adopted-by-the-global-medical-community-strengthening-ethical-standards-in-clinical-research-involving-humans/ (sources disagree on revision count; section numbers not verified)
- 45 CFR 46.116(f)(3) waiver/alteration criteria: (i) no more than minimal risk; (ii) could not practicably be done without the waiver; (iii) for identifiable private information or biospecimens, could not practicably be done without identifiable form; (iv) no adverse effect on rights and welfare; (v) pertinent information provided afterwards where appropriate. No waiver where the person refused broad consent for secondary storage and use.
- FDA-regulated research: 21 CFR 50.22 also applies (secondary; confirm).
- Waiver of consent does not waive HIPAA; PHI access still needs authorisation or a privacy-board waiver under 45 CFR 164.512(i).
- UNVERIFIED: ICH E6(R3), EU CTR 536/2014, 45 CFR 46.104 exemption categories (including 2018 changes).
Inferences: retrospective US patient data may be exempt or waived, but the IRB must determine this; authors should not self-declare. Prospective studies need documented consent unless a formal waiver applies.

## Q4. HIPAA de-identification (US)
Takeaway: 45 CFR 164.514 offers Safe Harbor (remove 18 identifier categories and have no actual knowledge the remainder could identify someone) and Expert Determination (qualified expert finds very small re-identification risk and documents methods). De-identified data is outside the Privacy Rule; residual risk remains.
- Safe Harbor: date elements more specific than year tied to an event must go; initials and identifier parts may not remain; first three ZIP digits only under the Census population rule.
- Expert Determination: expert needs knowledge of accepted statistical/scientific methods; can permit more dates/geography, with data use agreements.
Inferences: exact admission dates or full ZIP codes disqualify Safe Harbor; use Expert Determination or a limited data set with a DUA.
Gaps: full 18-identifier wording not retrieved.

## Q5. GDPR (EU)
Takeaway: Pseudonymised data is still personal data (Art. 4(5)). Anonymised data is outside GDPR (Recital 26). Health and genetic data are Art. 9 special categories. Research typically needs an Art. 6 basis, an Art. 9(2)(j) condition backed by EU or Member State law, and Art. 89(1) safeguards.
- Art. 9(2)(j): processing for scientific research, statistics or archiving only where Union or Member State law provides the basis and safeguards apply. https://gdpr-text.com/uk/read/article-89 (secondary)
- Art. 89(1): data minimisation; pseudonymisation where it still meets research aims; anonymous routes where aims allow.
- Art. 89(2): Member State derogations from access, rectification, restriction and objection rights, only where those rights would seriously impair the purpose.
Inferences: an ML study on patient data needs Art. 6 basis + Art. 9(2)(j) with national law + Art. 89 safeguards. Ethics approval does not supply a GDPR lawful basis.
Gaps: exact article text not retrieved; national research laws (e.g., Germany BDSG §27, France) not checked.

## Q6. UK (ICO)
Takeaway: UK GDPR/DPA 2018 research provisions cover scientific research, statistics and archiving. Consent is usually not the best lawful basis. Special-category data needs an Art. 9 condition plus Art. 89 safeguards. The common law duty of confidentiality is separate.
- ICO research provisions guidance: https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/the-research-provisions/ (search excerpts describe the 2022 draft; final version to check).
- ICO 2017 Royal Free finding: research processing breached common law confidentiality (secondary).
Gaps: Data (Use and Access) Act 2025 changes not checked; HRA/REC requirements not searched.

## Q7. Other national rules (UNVERIFIED, not searched)
- Canada: TCPS 2 (2022); PIPEDA; provincial health privacy laws (e.g., Ontario PHIPA).
- India: ICMR National Ethical Guidelines for Biomedical and Health Research (2017); Digital Personal Data Protection Act 2023.
- China: Personal Information Protection Law (2021); Data Security Law (2021); Human Genetic Resources rules.

## Q8. Trial registration
- ICMJE accepted registries: WHO ICTRP primary registries with the 24-item minimum dataset, or ClinicalTrials.gov.
- ICMJE trial definition: prospective assignment to an intervention, with or without concurrent control.
Gap: whether SPIRIT 2025 or CONSORT 2025 changed registration expectations not checked.

## Confidence summary
- High (official source seen): CONSORT 2025 (30), SPIRIT 2025 (34), TRIPOD+AI (27, BMJ 2024), CONSORT-AI/SPIRIT-AI, Helsinki 2024 adoption, ICMJE registration rule.
- Medium (secondary only): HIPAA Safe Harbor and Expert Determination; GDPR Art. 89 and 9(2)(j); ICO research provisions (2022 draft); 45 CFR 46.116(f).
- Low / not verified: PRISMA, COREQ/SRQR, ICH E6(R3), EU CTR, 45 CFR 46.104; journal-specific rules; Canada, India, China rules.
