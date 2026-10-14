"""Measure each optional mechanism against the current method, over seeds.

    python scripts/run_option_ablation.py --seeds 1337 2024 7 --jobs 4
    python scripts/run_option_ablation.py --report-only

Base: the proposed configuration on a 30-vehicle, 9-class fleet (rare hazards
on), plaintext aggregation so that every option applies, 20 rounds. Each arm
switches on one option; ``all_options`` switches on all of them. Writes
results/options/runs/<arm>/seed_<s>/ and results/options/options.{json,md}.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import get_context
from typing import Dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fedav.sim import stats  # noqa: E402

OUT = os.path.join(ROOT, "results", "options")
RUNS = os.path.join(OUT, "runs")

BASE = {
    "num_clients": 30,
    "data": {"num_classes": 9, "rare_hazard_share": 0.06},
    "train": {"rounds": 20, "local_epochs": 2},
    "dp": {"secure_aggregation": False},
    "comm": {"corrupt_prob": 0.0},
}
ARMS: Dict[str, Dict] = {
    "base": {},
    "focal_loss": {"model": {"focal_gamma": 1.5}},
    "kst_quality": {"aggregation": {"quality_score": "kst"}},
    "kst_ucb_selection": {"selection": {"mode": "kst_ucb"}},
    "scaffold": {"train": {"scaffold": True}},
    "simplex_cap_downweight": {"aggregation": {"alpha_cap_ratio": 2.5, "norm_downweight": 2.5}},
    "nesterov": {"aggregation": {"nesterov": True}},
    "ef_fix": {"comm": {"ef_normalize_steps": True, "ef_quantization": True}},
    "all_options": {"model": {"focal_gamma": 1.5},
                    "aggregation": {"quality_score": "kst", "alpha_cap_ratio": 2.5,
                                    "norm_downweight": 2.5, "nesterov": True},
                    "selection": {"mode": "kst_ucb"},
                    "train": {"scaffold": True, "grad_clip": 2.5},
                    "comm": {"ef_normalize_steps": True, "ef_quantization": True}},
}
ROWS = [("best_accuracy", "Best accuracy (%)", 100, 1),
        ("last5_accuracy", "Last-5-round accuracy (%)", 100, 1),
        ("best_rare_class_accuracy", "Best rare-hazard accuracy (%)", 100, 1),
        ("last5_rare_class_accuracy", "Last-5 rare-hazard accuracy (%)", 100, 1),
        ("worst_vehicle_accuracy", "Worst-vehicle accuracy (%)", 100, 1),
        ("worst_10pct_vehicle_accuracy", "Worst-10%-of-vehicles accuracy (%)", 100, 1),
        ("client_acc_std", "Cross-vehicle accuracy std", 1, 4),
        ("mb_per_round", "Traffic per round (MB)", 1, 2),
        ("clients_reported_per_round", "Vehicles reporting per round", 1, 2)]


def config(arm: str, seed: int, args):
    from fedav.config import FLConfig
    from fedav.sim.baselines import proposed_config
    base = FLConfig.from_yaml(args.config).merged({"seed": seed}).merged(BASE)
    if args.rounds:
        base = base.merged({"train": {"rounds": args.rounds}})
    return proposed_config(base).merged(BASE).merged(ARMS[arm]).merged(
        {"name": f"seed_{seed}", "out_dir": os.path.join(RUNS, arm)})


def _task(job):
    arm, seed, args = job
    import torch
    torch.set_num_threads(args.threads)
    from fedav.data.federated import build_fleet
    from fedav.sim.runner import run_experiment
    cfg = config(arm, seed, args)
    run_experiment(cfg, build_fleet(cfg), echo=False)
    pt = os.path.join(RUNS, arm, f"seed_{seed}", "best_model.pt")
    if os.path.exists(pt):
        os.remove(pt)
    return f"{arm} seed {seed} done"


def metrics(run_dir: str) -> Dict:
    import numpy as np
    with open(os.path.join(run_dir, "rounds.jsonl")) as fh:
        hist = [json.loads(line) for line in fh if line.strip()]
    with open(os.path.join(run_dir, "summary.json")) as fh:
        summary = json.load(fh)
    out = stats.run_metrics(hist, summary, ())
    rare = [r["rare_class_accuracy"] for r in hist if "rare_class_accuracy" in r]
    out["best_rare_class_accuracy"] = max(rare) if rare else None
    out["last5_rare_class_accuracy"] = float(np.mean(rare[-5:])) if rare else None
    vs = os.path.join(run_dir, "vehicle_summary.json")
    if os.path.exists(vs):
        with open(vs) as fh:
            accs = sorted(v["final_accuracy"] for v in json.load(fh)["vehicles"]
                          if v.get("final_accuracy") is not None)
        if accs:
            k = max(1, int(round(0.1 * len(accs))))
            out["worst_10pct_vehicle_accuracy"] = float(np.mean(accs[:k]))
    return out


def report() -> Dict:
    per_seed: Dict[str, Dict[int, Dict]] = {}
    for arm in ARMS:
        root = os.path.join(RUNS, arm)
        for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
            d = os.path.join(root, name)
            if os.path.exists(os.path.join(d, "summary.json")):
                per_seed.setdefault(arm, {})[int(name.split("_")[1])] = metrics(d)
    agg = stats.aggregate(per_seed, "base")
    out = {"setting": BASE, "arms": {a: ARMS[a] for a in per_seed}, "result": agg,
           "per_seed": per_seed}
    stats.write(os.path.join(OUT, "options.json"), out)
    arms = [a for a in ARMS if a in agg["arms"]]
    lines = ["# Optional mechanisms: multi-seed ablation", "",
             "Generated by `scripts/run_option_ablation.py`. 30 vehicles, 9 classes (rare hazards "
             "on), 20 rounds, plaintext aggregation, seeds as listed in `options.json`. Cells: mean "
             "± 95% CI over seeds. Each arm switches one option on over `base` (the proposed "
             "configuration).", "",
             "| Metric | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for m, label, scale, d in ROWS:
        lines.append(f"| {label} | " + " | ".join(
            stats.fmt(agg["arms"][a].get(m, {}), scale, d) for a in arms) + " |")
    lines += ["", "## Paired difference vs base (option − base, per seed)", "",
              "| Metric | " + " | ".join(a for a in arms if a != "base") + " |",
              "|---|" + "---|" * (len(arms) - 1)]
    for m, label, scale, d in ROWS:
        lines.append(f"| {label} | " + " | ".join(
            stats.fmt(agg["paired_vs_reference"][a].get(m, {}), scale, d)
            for a in arms if a != "base") + " |")
    lines += ["", "## Reading (14 Oct 2026 run)", "",
              "* The setting is too hard to resolve most options: base accuracy is ~23% after 20 "
              "rounds on 9 classes (chance 11%), and **rare-hazard accuracy is 0% in every arm and "
              "seed** - at a 6% rare share the model never predicts a hazard class in this budget, "
              "so this run cannot test whether focal loss or the KST score helps rare hazards.",
              "* Only two differences have a 95% CI excluding zero, both costs: SCAFFOLD adds "
              "~4.2 MB/round (the control-variate uplink), and `kst_ucb` selection delivers ~0.6 "
              "fewer vehicle updates per round (uptime is no longer a gate). SCAFFOLD's +2.0 points "
              "of last-5 accuracy is borderline.",
              "* No option, and not all options together, measurably improves accuracy, worst-"
              "vehicle or worst-10% accuracy here. A longer run (more rounds, larger rare share) is "
              "needed before any of them is claimed."]
    with open(os.path.join(OUT, "options.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "patent_eval.yaml"))
    parser.add_argument("--seeds", type=int, nargs="*", default=[1337, 2024, 7])
    parser.add_argument("--arms", nargs="*", default=list(ARMS))
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if not args.report_only:
        jobs = [(a, s, args) for s in args.seeds for a in args.arms]
        with get_context("spawn").Pool(args.jobs) as pool:
            for msg in pool.imap_unordered(_task, jobs):
                print(msg, flush=True)
    report()
    with open(os.path.join(OUT, "options.md")) as fh:
        print(fh.read())


if __name__ == "__main__":
    main()
