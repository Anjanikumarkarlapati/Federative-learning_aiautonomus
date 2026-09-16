# Committed experiment evidence

The live output directories (`runs/`, `runs_attrib/`) are gitignored — they churn on
every round. This directory holds the stable artefacts the README's Results section
is rendered from, so the numbers there can be checked against their source.

```
benchmark/
├── benchmark_report.md     # the rendered comparison + PRD 12.3 acceptance
├── benchmark_report.json   # the same, machine-readable
└── <arm>/
    ├── config.json         # the exact config that produced the arm
    ├── summary.json        # final metrics
    └── rounds.csv          # per-round telemetry
```

Regenerate the README's Results section from these with:

```bash
python scripts/update_readme_results.py results/benchmark/benchmark_report.json README.md
```
