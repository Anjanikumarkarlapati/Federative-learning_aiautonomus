# IP India filing checklist

*From `reports/Indian patent filing for FL AV.md`. Fees are "as reported" by
secondary sources because the IP India pages could not be opened during the
research; check them against the live First Schedule on the day of filing. Not
legal advice; a registered patent agent should file.*

## Order of work

1. **Do not publish anything yet.** No public repository, preprint, poster, talk
   or shared session link until the provisional is filed. Confirm the GitHub
   repository is private.
2. Complete and sign [INVENTION_DISCLOSURE.md](INVENTION_DISCLOSURE.md); settle
   inventors (natural persons only) and applicant (individuals, or the
   university / employer if its IP policy claims ownership).
3. Have an agent run a full-text prior-art search on the claims in
   [CLAIMS_DRAFT.md](CLAIMS_DRAFT.md), especially Toyota US 12,258,045 B2,
   Toyota US 12,337,867, MERL US 12,369,186, and work on heterogeneous
   secure aggregation (e.g. HeteroSAg).
4. File the **provisional**: Form 1 + Form 2 (provisional, from
   [FORM2_PROVISIONAL.md](FORM2_PROVISIONAL.md) and `drawings/drawings.pdf`).
   Write it as fully as a complete specification: a claim not fairly based on
   the provisional gets the later date.
5. After the filing receipt: publish if wanted; under s.39 file abroad only
   after six weeks (no secrecy direction) or with Form 25 permission.
6. Within **12 months**: complete specification (Form 2), Form 5, Form 3 if not
   already filed, Form 18 (request for examination) at the same time, PCT if
   going abroad.

## Forms

| Form | What | When | Fee, natural person, e-filing (as reported) | Status |
|---|---|---|---|---|
| 1 | Application for grant | with the provisional | INR 1,600 | [ ] |
| 2 | Provisional specification | now | in Form 1 fee (≤ 30 pages, ≤ 10 claims) | draft ready |
| 2 | Complete specification | ≤ 12 months after the provisional | extra page INR 160, extra claim INR 320 | [ ] |
| 3 | Foreign-filing statement | with the application or ≤ 6 months; update ≤ 3 months after the FER | — | [ ] |
| 5 | Declaration of inventorship | with the complete specification | — | [ ] |
| 9 | Early publication (optional) | with the complete | INR 2,500 | [ ] |
| 18 / 18A | Request for (expedited) examination | ≤ 31 months from priority; file with the complete | INR 4,000 / 8,000 | [ ] |
| 26 | Agent authorisation | if an agent files | — | [ ] |
| 28 | Startup / small entity / educational institution | if claimed | — | [ ] |

About **INR 8,100** in official fees to examination for natural persons
(1,600 + 2,500 + 4,000), plus agent fees. Form 18A (FER in ~6 months instead of
~25) is open only on Rule 24C grounds: startup, small entity, at least one
female natural-person applicant, government entity, or PCT with India as
ISA/IPEA.

## Format checks (run `python scripts/check_spec_limits.py`)

| Rule | Limit | Checked by |
|---|---|---|
| Abstract (Rule 13(7)) | ≤ 150 words, starts with the title, names a figure | script |
| Claims | ≤ 10 to avoid excess-claim fees | script |
| Pages | ≤ 30 to avoid excess-page fees | script (estimate) |
| Drawings (Rule 15) | A4, margins ≥ 4 cm top/left, ≥ 3 cm bottom/right, sheets numbered, applicant name top left, no dimensions | `scripts/make_patent_drawings.py` |
| Reference numerals | every numeral in the text appears in the drawings and vice versa | [REFERENCE_NUMERALS.md](REFERENCE_NUMERALS.md) |

## Deadlines (fill in once the provisional is filed)

| Event | Date |
|---|---|
| Provisional filed (priority date) | [ ] |
| Form 3 due (6 months) | [ ] |
| Complete specification + Form 5 due (12 months) | [ ] |
| PCT due (12 months) | [ ] |
| Form 18 latest (31 months) | [ ] |
