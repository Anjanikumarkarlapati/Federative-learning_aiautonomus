"""Print runs/ablation.json as a compact table (used by scripts/make_run_guide.sh)."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "runs/ablation.json"
with open(path, encoding="utf-8") as fh:
    res = json.load(fh)
print(f"{'arm':<30}{'accuracy %':>11}{'ADE':>10}{'MB/round':>10}{'client std':>12}{'rounds':>8}")
print("-" * 81)
for name, s in res.items():
    rob = s.get("robustness", {})
    print(f"{name:<30}{100 * s['accuracy']:>11.2f}{s['ade']:>10.4g}{s['mb_per_round']:>10.2f}"
          f"{rob.get('client_acc_std', float('nan')):>12.4f}{s['rounds_run']:>8}")
print()
print(f"full results: {path}")
