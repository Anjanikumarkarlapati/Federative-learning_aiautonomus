"""Render the Form 2 provisional from its template, the claims draft and the evidence.

    python scripts/render_spec.py

docs/patent/FORM2_PROVISIONAL.template.md + CLAIMS_DRAFT.md + results/patent/effects.json
-> docs/patent/FORM2_PROVISIONAL.md, then runs check_spec_limits.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAT = os.path.join(ROOT, "docs", "patent")
sys.path.insert(0, ROOT)

from fedav.sim.stats import fmt  # noqa: E402

ARMS = [("std_fedavg", "A"), ("fedavg_transport", "B"), ("proposed_secure", "C"),
        ("proposed_tiered", "D")]
ROWS = [("best_accuracy", "Best test accuracy (%)", 100, 1),
        ("worst_vehicle_accuracy", "Accuracy on the worst-served vehicle (%)", 100, 1),
        ("mb_per_round", "Traffic per round, up + down (MB)", 1, 2),
        ("uplink_airtime_s_per_round", "Uplink airtime per round (s)", 1, 1),
        ("max_round_airtime_s", "Longest round's uplink airtime (s)", 1, 1),
        ("uplink_energy_j_per_round", "Uplink radio energy per round (J)", 1, 1),
        ("accuracy_within_100s_airtime", "Best accuracy within 100 s of cumulative uplink airtime (%)", 100, 1),
        ("accuracy_within_300s_airtime", "Best accuracy within 300 s of cumulative uplink airtime (%)", 100, 1)]


def results_table(effects: dict) -> str:
    agg = effects["vs_std_fedavg"]["arms"]
    head = "| Quantity | " + " | ".join(f"{k}" for _, k in ARMS) + " |"
    lines = [head, "|---|" + "---|" * len(ARMS)]
    for m, label, scale, digits in ROWS:
        lines.append(f"| {label} | " + " | ".join(
            fmt(agg.get(a, {}).get(m, {}), scale, digits) for a, _ in ARMS) + " |")
    return "\n".join(lines)


PAIRED = [("uplink_airtime_s_per_round", "Uplink airtime per round (s)", 1, 1),
          ("max_round_airtime_s", "Longest round's uplink airtime (s)", 1, 1),
          ("uplink_energy_j_per_round", "Uplink radio energy per round (J)", 1, 1),
          ("mb_per_round", "Traffic per round (MB)", 1, 2),
          ("clients_reported_per_round", "Vehicles whose update arrived, per round", 1, 2),
          ("best_accuracy", "Best test accuracy (percentage points)", 100, 1)]


def paired_table(effects: dict) -> str:
    paired = effects["vs_fedavg_transport"]["paired_vs_reference"]
    lines = ["| Quantity | C − B | D − B |", "|---|---|---|"]
    for m, label, scale, digits in PAIRED:
        lines.append(f"| {label} | {fmt(paired.get('proposed_secure', {}).get(m, {}), scale, digits)}"
                     f" | {fmt(paired.get('proposed_tiered', {}).get(m, {}), scale, digits)} |")
    return "\n".join(lines)


def max_error(root: str) -> float:
    worst = 0.0
    for arm in ("proposed_secure", "proposed_tiered"):
        base = os.path.join(root, arm)
        for seed in os.listdir(base) if os.path.isdir(base) else []:
            with open(os.path.join(base, seed, "rounds.jsonl")) as fh:
                for line in fh:
                    worst = max(worst, json.loads(line).get("secagg_rel_error", 0.0))
    return worst


def main() -> None:
    with open(os.path.join(PAT, "FORM2_PROVISIONAL.template.md"), encoding="utf-8") as fh:
        text = fh.read()
    with open(os.path.join(PAT, "CLAIMS_DRAFT.md"), encoding="utf-8") as fh:
        claims = fh.read()
    body = claims.split("**We claim:**", 1)[1].split("\n---", 1)[0].strip()
    text = text.replace("[[CLAIMS]]", "We claim:\n\n" + body)
    eff = os.path.join(ROOT, "results", "patent", "effects.json")
    if os.path.exists(eff):
        with open(eff) as fh:
            effects = json.load(fh)
        text = text.replace("[[RESULTS TABLE]]", results_table(effects))
        text = text.replace("[[PAIRED TABLE]]", paired_table(effects))
        text = text.replace("[[ERR]]", f"at most {max_error(os.path.join(ROOT, 'results', 'patent', 'runs')):.1e}")
    out = os.path.join(PAT, "FORM2_PROVISIONAL.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    left = re.findall(r"\[\[[A-Z ]+\]\]", text)
    print(f"wrote {out}" + (f"; unfilled: {left}" if left else ""))
    sys.exit(subprocess.call([sys.executable, os.path.join(ROOT, "scripts", "check_spec_limits.py"), out]))


if __name__ == "__main__":
    main()
