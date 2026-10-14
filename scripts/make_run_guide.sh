#!/usr/bin/env bash
# Regenerate the step-by-step screenshots in docs/run_guide/.
#
#   bash scripts/make_run_guide.sh            # terminal steps (≈10 min on 4 CPU cores)
#   SKIP_SLOW=1 bash scripts/make_run_guide.sh  # skip benchmark + ablation
#
# Each step runs the real command, captures its output and renders it as a PNG with
# scripts/render_terminal.py. Dashboard pages are captured separately:
#
#   streamlit run dashboard/app.py &          # then
#   python scripts/screenshot_dashboard.py     # (CHROMIUM_PATH=... if Playwright has no browser)
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=docs/run_guide
TMP=$(mktemp -d)
mkdir -p "$OUT"
trap 'rm -rf "$TMP"' EXIT

shot() {  # shot FILE "TITLE" "COMMAND" [render args...]  (reads output from $TMP/cur.txt)
  local file=$1 title=$2 cmd=$3; shift 3
  python scripts/render_terminal.py "$OUT/$file" "$title" "$cmd" "$TMP/cur.txt" "$@"
  echo "  wrote $OUT/$file"
}

echo "Step 1: installed packages"
pip list 2>/dev/null | grep -i -E "^(torch|numpy|PyYAML|pytest|streamlit|plotly|pandas) " > "$TMP/cur.txt"
shot step01_install.png "Step 1 - Install dependencies" \
  "pip install -r requirements.txt && pip list | grep -E 'torch|numpy|streamlit|...'"

echo "Step 2: tests"
python -m pytest -q -p no:cacheprovider > "$TMP/cur.txt" 2>&1 || true
shot step02_tests.png "Step 2 - Run the test suite" "python -m pytest -q" --max-lines 30

echo "Step 3: CLI help"
python -m fedav.cli --help > "$TMP/cur.txt" 2>&1
shot step03_help.png "Step 3 - See the CLI commands" "python -m fedav.cli --help"

echo "Step 4: fleet info"
python -m fedav.cli info --config configs/default.yaml > "$TMP/cur.txt" 2>&1
shot step04_info.png "Step 4 - Inspect the fleet" "python -m fedav.cli info --config configs/default.yaml"

echo "Step 5: quick run"
python -m fedav.cli run --config configs/quick.yaml > "$TMP/cur.txt" 2>&1
shot step05_quick_run.png "Step 5 - Quick smoke run (8 rounds, ~40 s)" \
  "python -m fedav.cli run --config configs/quick.yaml"

echo "Step 6: run outputs"
{
  ls -l runs/quick | tail -n +2 | awk '{printf "%-24s %10s bytes\n",$9,$5}'
  echo
  echo "\$ head -3 runs/quick/rounds.csv"
  head -3 runs/quick/rounds.csv | cut -c1-108
} > "$TMP/cur.txt"
shot step06_run_outputs.png "Step 6 - Files every run writes" "ls runs/quick"

if [ -z "${SKIP_SLOW:-}" ]; then
  echo "Step 7: benchmark"
  python -m fedav.cli benchmark --config configs/quick.yaml > "$TMP/bench.txt" 2>&1
  { grep -E "^\[" "$TMP/bench.txt" | awk '{last[$1" "$2]=$0} END{for(k in last) print last[k]}' | sort
    echo
    sed -n '/^# FL-AV benchmark report/,$p' "$TMP/bench.txt" | sed -E 's/ \(\{.*\}\)//'
  } > "$TMP/cur.txt"
  shot step07_benchmark.png "Step 7 - Benchmark all four arms (PRD 12.2/12.3)" \
    "python -m fedav.cli benchmark --config configs/quick.yaml" --width 120

  echo "Step 8: ablation"
  python -m fedav.cli ablation --config configs/quick.yaml > "$TMP/abl.txt" 2>&1
  grep -v -E "^\[" "$TMP/abl.txt" > "$TMP/cur.txt"
  shot step08_ablation.png "Step 8 - Ablation: which enhancement does the work" \
    "python -m fedav.cli ablation --config configs/quick.yaml" --width 120
fi

printf '\n  You can now view your Streamlit app in your browser.\n\n  Local URL: http://localhost:8501\n  Network URL: http://<your-machine-ip>:8501\n\n  (open the Local URL in a browser; press Ctrl+C here to stop)\n' > "$TMP/cur.txt"
shot step09_start_dashboard.png "Step 9 - Start the dashboard" "streamlit run dashboard/app.py"

echo "Done. For steps 10-17 start the dashboard and run scripts/screenshot_dashboard.py"
