# Invention disclosure record

**Status: TEMPLATE — must be completed, signed and witnessed by the people
involved.** The technical sections are pre-filled from the repository; every
`[ ]` field is for a human, and nobody should sign anything here they did not do.

Why this matters for an Indian filing: Form 5 names the true and first inventors,
who must be natural persons (the Indian Patent Office refused the DABUS
application on 15 April 2026 because an AI system cannot be an inventor). Every
commit in this repository is authored by "Claude", an AI tool, and the
requirements document the design follows is not in the repository. The record
below is where the humans state, in their own words and with dates, what they
conceived. AI assistance is recorded as a tool, like a compiler or a search
engine.

Keep the signed copy (PDF, dated, witnessed) **outside git** and outside any
public place until the provisional specification is filed.

---

## 1. Title

Quality-hiding, airtime-tiered secure aggregation for federated training of
vehicle perception models over vehicle-to-infrastructure links.

## 2. People

| Role | Name | Affiliation / employer | Email | Contribution in their own words |
|---|---|---|---|---|
| Inventor 1 | [ ] | [ ] | [ ] | [ ] |
| Inventor 2 | [ ] | [ ] | [ ] | [ ] |
| Contributor (not inventor) | [ ] | [ ] | [ ] | [ ] |

Inventorship turns on conception of a claimed element, not on writing code or
running experiments. Tick, for each claim element in
[CLAIMS_DRAFT.md](CLAIMS_DRAFT.md), who conceived it:

| Claim element | Conceived by | Date first conceived | Where it was first written down |
|---|---|---|---|
| Quality weight and step count folded inside the masked payload so the RSU sees only sums (claim 1(d)–(f), 2(g)) | [ ] | [ ] | [ ] |
| One ordered support derived from the previous global update, no indices sent (claim 1, 2(e)) | [ ] | [ ] | [ ] |
| Airtime-equalising prefix tiers with per-segment normalisation and pairwise masks over common coordinates (claim 1(b), 2(h)) | [ ] | [ ] | [ ] |
| NACK-driven retention of the whole update in the residual (claim 2(j)) | [ ] | [ ] | [ ] |
| Seed reveal for dropped vehicles (claim 4) | [ ] | [ ] | [ ] |
| Starvation guard before the uptime gate (claim 7) | [ ] | [ ] | [ ] |
| Quality score = sensor health × (1 − Hellinger distance) (claim 8) | [ ] | [ ] | [ ] |

## 3. Tools used

| Tool | What it was used for |
|---|---|
| Claude Code (AI coding assistant) | writing and testing code to the inventors' specification; literature and patent searches; drafting documents |
| [ ] | [ ] |

State here who decided each design choice the AI tool proposed or implemented:
[ ]

## 4. Problem, solution, effect

* **Problem:** see [EMBODIMENTS.md §1](EMBODIMENTS.md#1-technical-problem).
* **Solution:** see [EMBODIMENTS.md §2](EMBODIMENTS.md#2-technical-solution-summary).
* **Measured effect:** see `results/patent/effects.md` (multi-seed, paired
  comparison against FedAvg and against FedAvg on the same compression
  transport).

## 5. Disclosure history (novelty)

India has no general grace period for an inventor's own publication.

| Question | Answer |
|---|---|
| Has the repository ever been public (GitHub visibility, forks, mirrors)? | [ ] |
| Has any claude.ai session link with this content been shared publicly? | [ ] |
| Any talk, poster, thesis, preprint, demo or report outside the team? | [ ] |
| Any NDA covering people outside the team who have seen it? | [ ] |

## 6. Ownership

| Question | Answer |
|---|---|
| Were the inventors students or employees when they conceived it? | [ ] |
| Does the university's / employer's IP policy claim ownership? | [ ] |
| Any funding agreement with IP terms? | [ ] |
| Applicant(s) for Form 1 (inventors themselves, or an assignee) | [ ] |

## 7. Signatures

| Name | Signature | Date |
|---|---|---|
| Inventor 1 | | |
| Inventor 2 | | |
| Witness (read and understood, not an inventor) | | |
