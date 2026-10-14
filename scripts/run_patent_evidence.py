"""Multi-seed evidence for the patent specification.

    python scripts/run_patent_evidence.py --seeds 1337 2024 7 --jobs 4
    python scripts/run_patent_evidence.py --report-only

Arms (all on the same fleet per seed, configs/patent_eval.yaml):

    std_fedavg        McMahan FedAvg: uniform selection, n_k weights, full fp32 weights
    fedavg_transport  FedAvg on the proposed transport (fp16 + gzip top-k + error
                      feedback + multicast + MB budget) - isolates the claimed
                      mechanisms from known compression
    proposed_secure   K1: quality-hiding secure aggregation on a shared support
    proposed_tiered   K1 + K2: plus airtime-budget tiers and NACK retention

Writes per-run telemetry to results/patent/runs/<arm>/seed_<s>/ and the tables
to results/patent/effects.{json,md}.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import get_context
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fedav.sim import stats  # noqa: E402

OUT = os.path.join(ROOT, "results", "patent")
RUNS = os.path.join(OUT, "runs")
ARMS = ["std_fedavg", "fedavg_transport", "proposed_secure", "proposed_tiered"]
LABELS = {"std_fedavg": "Std. FedAvg", "fedavg_transport": "FedAvg + same transport",
          "proposed_secure": "Proposed (K1)", "proposed_tiered": "Proposed (K1 + K2)"}
THRESHOLDS = (0.40, 0.50, 0.60)


def arm_config(arm: str, seed: int, args):
    from fedav.config import FLConfig
    from fedav.sim.baselines import fedavg_config, fedavg_transport_config, proposed_config

    base = FLConfig.from_yaml(args.config).merged({"seed": seed})
    if args.rounds:
        base = base.merged({"train": {"rounds": args.rounds}})
    if arm == "std_fedavg":
        cfg = fedavg_config(base)
    elif arm == "fedavg_transport":
        cfg = fedavg_transport_config(base)
    elif arm == "proposed_secure":
        cfg = proposed_config(base).merged({"comm": {"airtime_budget_s": 0.0}})
    elif arm == "proposed_tiered":
        cfg = proposed_config(base).merged({"comm": {"airtime_budget_s": args.airtime_budget}})
    else:
        raise ValueError(arm)
    return cfg.merged({"name": f"seed_{seed}", "out_dir": os.path.join(RUNS, arm)})


def _task(job: Tuple[str, int, argparse.Namespace]) -> str:
    arm, seed, args = job
    import torch
    torch.set_num_threads(args.threads)
    from fedav.data.federated import build_fleet
    from fedav.sim.runner import run_experiment

    cfg = arm_config(arm, seed, args)
    run_experiment(cfg, build_fleet(cfg), echo=False)
    pt = os.path.join(RUNS, arm, f"seed_{seed}", "best_model.pt")
    if os.path.exists(pt):
        os.remove(pt)                       # checkpoints are not evidence; keep the repo small
    return f"{arm} seed {seed} done"


def load(arm: str, seed: int):
    d = os.path.join(RUNS, arm, f"seed_{seed}")
    with open(os.path.join(d, "rounds.jsonl")) as fh:
        history = [json.loads(line) for line in fh if line.strip()]
    with open(os.path.join(d, "summary.json")) as fh:
        summary = json.load(fh)
    return history, summary


def report() -> Dict:
    per_seed: Dict[str, Dict[int, Dict]] = {}
    for arm in ARMS:
        root = os.path.join(RUNS, arm)
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            if not os.path.exists(os.path.join(root, name, "summary.json")):
                continue
            seed = int(name.split("_")[1])
            history, summary = load(arm, seed)
            per_seed.setdefault(arm, {})[seed] = stats.run_metrics(history, summary, THRESHOLDS)
    out = {"vs_std_fedavg": stats.aggregate(per_seed, "std_fedavg"),
           "vs_fedavg_transport": stats.aggregate(per_seed, "fedavg_transport"),
           "per_seed": per_seed, "thresholds": THRESHOLDS}
    stats.write(os.path.join(OUT, "effects.json"), out)
    with open(os.path.join(OUT, "effects.md"), "w", encoding="utf-8") as fh:
        fh.write(markdown(out))
    return out


ROWS = [  # (metric, label, scale, digits)
    ("best_accuracy", "Best accuracy (%)", 100, 1),
    ("last5_accuracy", "Last-5-round accuracy (%)", 100, 1),
    ("worst_vehicle_accuracy", "Worst-vehicle accuracy (%)", 100, 1),
    ("client_acc_std", "Cross-vehicle accuracy std", 1, 4),
    ("mb_per_round", "Traffic per round (MB, up + down)", 1, 2),
    ("uplink_airtime_s_per_round", "Uplink airtime per round (s)", 1, 2),
    ("max_round_airtime_s", "Worst round uplink airtime (s)", 1, 2),
    ("uplink_energy_j_per_round", "Uplink radio energy per round (J)", 1, 2),
    ("clients_reported_per_round", "Vehicles reporting per round", 1, 2),
] + [(f"{k}_{int(100 * t)}", f"{lab} {int(100 * t)}% accuracy", 1, d)
     for t in THRESHOLDS
     for k, lab, d in (("rounds_to", "Rounds to", 1), ("mb_to", "MB to", 1),
                       ("airtime_s_to", "Uplink airtime (s) to", 1))]


def markdown(out: Dict) -> str:
    agg = out["vs_std_fedavg"]
    arms = [a for a in ARMS if a in agg["arms"]]
    n = {a: max((s.get("n", 0) for s in agg["arms"][a].values()), default=0) for a in arms}
    lines = ["# Patent evidence: multi-seed comparison", "",
             "Generated by `scripts/run_patent_evidence.py` from `results/patent/runs/`. "
             "Cells are mean ± 95% CI (Student t over seeds). Same fleet per seed for every arm.", "",
             "Seeds per arm: " + ", ".join(f"{LABELS[a]} n={n[a]}" for a in arms), "",
             "| Metric | " + " | ".join(LABELS[a] for a in arms) + " |",
             "|---|" + "---|" * len(arms)]
    for m, label, scale, digits in ROWS:
        cells = [stats.fmt(agg["arms"][a].get(m, {}), scale, digits) for a in arms]
        reached = [f"{len([v for v in agg['arms'][a].get(m, {}).get('values', [])])}/{n[a]}"
                   for a in arms] if m.startswith(("rounds_to", "mb_to", "airtime_s_to")) else None
        if reached:
            cells = [f"{c} ({r})" for c, r in zip(cells, reached)]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "Threshold rows show (seeds that reached it / seeds run); the mean is over the seeds "
              "that reached it.", ""]
    for ref_key, title in (("vs_std_fedavg", "Std. FedAvg"),
                           ("vs_fedavg_transport", "FedAvg + same transport")):
        paired = out[ref_key]["paired_vs_reference"]
        if not paired:
            continue
        lines += [f"## Paired difference vs {title} (arm − reference, per seed)", "",
                  "| Metric | " + " | ".join(LABELS[a] for a in paired) + " |",
                  "|---|" + "---|" * len(paired)]
        for m, label, scale, digits in ROWS[:9]:
            lines.append(f"| {label} | " + " | ".join(
                stats.fmt(paired[a].get(m, {}), scale, digits) for a in paired) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "patent_eval.yaml"))
    parser.add_argument("--seeds", type=int, nargs="*", default=[1337, 2024, 7])
    parser.add_argument("--arms", nargs="*", default=ARMS)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--airtime-budget", type=float, default=12.0)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if not args.report_only:
        jobs = [(arm, seed, args) for seed in args.seeds for arm in args.arms]
        with get_context("spawn").Pool(args.jobs) as pool:
            for msg in pool.imap_unordered(_task, jobs):
                print(msg, flush=True)
    out = report()
    print(markdown(out))


if __name__ == "__main__":
    main()
