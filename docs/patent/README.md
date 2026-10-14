# Patent package (India, IP India)

Built from `reports/Indian patent filing for FL AV.md`. **Read
[FILING_CHECKLIST.md](FILING_CHECKLIST.md) first**: this repository is public, so
an agent must assess the disclosure before anything is filed. Nothing here is
legal advice.

| File | What it is | Requirement it serves |
|---|---|---|
| [FILING_CHECKLIST.md](FILING_CHECKLIST.md) | forms, fees, deadlines, order of work, disclosure status | Forms 1/2/3/5/18, s.9, s.39 |
| [INVENTION_DISCLOSURE.md](INVENTION_DISCLOSURE.md) | template for inventors to complete, sign and witness | Form 5, natural-person inventorship |
| [FORM2_PROVISIONAL.md](FORM2_PROVISIONAL.md) | provisional specification, rendered from the `.template.md` | Form 2, s.10(4), Rule 13 |
| [CLAIMS_DRAFT.md](CLAIMS_DRAFT.md) | 10 draft claims (method RSU, method OBU, system, 7 dependent) | s.10(4)(c), fee limit |
| [CLAIM_EVIDENCE_MATRIX.md](CLAIM_EVIDENCE_MATRIX.md) | claim element → code → test → result, checked by a test | s.10(4) fair basis; FER replies |
| [EMBODIMENTS.md](EMBODIMENTS.md) | technical problem, OBU/RSU hardware, best-method parameters, packet format | s.3(k) technical means, s.10(4) best method |
| [REFERENCE_NUMERALS.md](REFERENCE_NUMERALS.md) | numeral table, generated with the drawings | Rule 15 |
| `drawings/drawings.pdf` | FIG. 1–9, black and white, A4, Rule 15 margins | s.10(2), Rule 15 |
| `../../results/patent/effects.md` | multi-seed evidence of the technical effect | s.3(k) "demonstrable technical advancement" |

## Regenerate

```bash
python scripts/run_patent_evidence.py --seeds 1337 2024 7 --jobs 4   # ~2 h on 4 CPU cores
python scripts/make_patent_drawings.py --applicant "Applicant: NAME"
python scripts/render_spec.py          # fills FORM2_PROVISIONAL.md, checks limits
python scripts/check_claim_matrix.py
```

## What was built for this package

* `fedav/server/secagg.py` — claims K1 + K2: quality-hiding secure aggregation
  in Z_2^32 on a shared, tiered support with Diffie-Hellman pair masks and
  dropout seed reveal.
* `fedav/server/v2x.py` — 3GPP TR 38.901 / TR 37.885 V2I radio model turning
  bytes into airtime and energy.
* NACK residual retention (`TopKCompressor.restore`, `SimRunner._secure_round`),
  optional payload-corruption injection (`comm.corrupt_prob`).
* Fixes that kept the specification from over-claiming: FedNova normalisation
  was silently dropped under secure aggregation; the starvation guard skipped
  weak-link vehicles; the "uniform" FedAvg baseline favoured good links.
* `fedav/sim/stats.py`, `scripts/run_patent_evidence.py` — paired multi-seed
  comparison, including FedAvg on the same compression transport.
