"""FL-AV Streamlit dashboard.

    streamlit run dashboard/app.py

Pages
-----
Overview        headline numbers + PRD 12.3 acceptance
Benchmark       the four arms of PRD 12.2, round by round
Ablation        which enhancement does the work
Launch a run    configure, start and watch a federated run live
Run explorer    any run under runs/ or results/
Validation      the 14 mathematical-validation figures
"""
from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard import data  # noqa: E402

st.set_page_config(page_title="FL-AV Dashboard", page_icon="🚗", layout="wide")

GRID = dict(showgrid=True, gridwidth=1, zeroline=False)


# ------------------------------------------------------------------ chart helpers

def _style(fig: go.Figure, y_title: str, x_title: str = "Round", height: int = 320) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
    )
    fig.update_xaxes(title=x_title, **GRID)
    fig.update_yaxes(title=y_title, **GRID)
    fig.update_traces(line=dict(width=2), selector=dict(type="scatter"))
    return fig


def arm_lines(df: pd.DataFrame, y: str, y_title: str, title: str, scale: float = 1.0) -> go.Figure:
    fig = go.Figure()
    for arm in data.ARM_ORDER:
        sub = df[df["arm"] == arm]
        if sub.empty or y not in sub:
            continue
        fig.add_trace(go.Scatter(
            x=sub["round"], y=sub[y] * scale, name=data.ARM_LABELS[arm], mode="lines",
            line=dict(color=data.ARM_COLORS[arm]),
            hovertemplate="%{y:.3g}",
        ))
    fig.update_layout(title=title)
    return _style(fig, y_title)


def single_line(df: pd.DataFrame, y: str, title: str, y_title: str,
                scale: float = 1.0, color: str = "#2a78d6") -> Optional[go.Figure]:
    if y not in df or df[y].dropna().empty:
        return None
    sub = df[["round", y]].dropna()
    fig = go.Figure(go.Scatter(x=sub["round"], y=sub[y] * scale, mode="lines+markers",
                               line=dict(color=color), marker=dict(size=6),
                               name=title, hovertemplate="%{y:.4g}"))
    fig.update_layout(title=title, showlegend=False)
    if len(sub) <= 12:
        fig.update_xaxes(dtick=1)
    return _style(fig, y_title, height=280)


def show(fig: Optional[go.Figure]) -> None:
    if fig is None:
        return
    title = fig.layout.title.text
    if title:
        st.markdown(f"**{title}**")
        fig.update_layout(title=None)
    st.plotly_chart(fig, width="stretch", theme="streamlit")


# --------------------------------------------------------------------- cached io

@st.cache_data(ttl=30)
def cached_benchmark():
    report = data.load_benchmark()
    if report is None:
        return None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    return report, data.benchmark_table(report), data.acceptance_table(report), data.benchmark_rounds()


# ------------------------------------------------------------------------- pages

def page_overview() -> None:
    st.title("Federated learning for collaborative AV training")
    st.caption("A fleet trains one shared perception / trajectory model. Only model deltas "
               "leave the vehicle; the road-side unit aggregates them.")
    report, table, acc, _ = cached_benchmark()
    if report is None:
        st.info("No benchmark report yet. Run `python -m fedav.cli benchmark --config "
                "configs/default.yaml`, then refresh.")
        return

    rows = table.set_index("arm")
    prop, fed, cen = (rows.loc[a] if a in rows.index else None
                      for a in ("proposed_fl", "std_fedavg", "centralized"))
    cfg = report.get("config", {})
    st.markdown(f"**Fleet** N={cfg.get('num_clients')} · C={cfg.get('selection', {}).get('fraction')} · "
                f"E={cfg.get('train', {}).get('local_epochs')} · R_max={cfg.get('train', {}).get('rounds')} · "
                f"Dirichlet α={cfg.get('data', {}).get('dirichlet_alpha')}")

    c1, c2, c3, c4 = st.columns(4)
    if prop is not None and fed is not None:
        c1.metric("Proposed FL accuracy", f"{prop['Accuracy (%)']:.1f}%",
                  f"{prop['Accuracy (%)'] - fed['Accuracy (%)']:+.1f} pts vs FedAvg")
        saving = 1 - prop["MB / round"] / fed["MB / round"]
        c2.metric("Uplink per round", f"{prop['MB / round']:.1f} MB",
                  f"-{saving:.0%} vs FedAvg", delta_color="inverse")
        c3.metric("Worst-served vehicle", f"{prop['Worst client (%)']:.1f}%",
                  f"{prop['Worst client (%)'] - fed['Worst client (%)']:+.1f} pts vs FedAvg")
    if cen is not None and prop is not None:
        c4.metric("Share of centralized ceiling",
                  f"{prop['Accuracy (%)'] / cen['Accuracy (%)']:.1%}",
                  "with no raw data egress", delta_color="off", delta_arrow="off")

    st.subheader("PRD 12.3 acceptance")
    if not acc.empty:
        passed = int(acc["pass"].sum())
        verdict = (report.get("acceptance") or {}).get("status")
        (st.success if verdict == "PASS" else st.warning)(
            f"{passed} of {len(acc)} checks pass — overall **{verdict or ('PASS' if passed == len(acc) else 'FAIL')}**")
        view = acc.assign(Result=acc["pass"].map({True: "✅ Pass", False: "❌ Fail"}))
        st.dataframe(view[["Result", "Requirement", "Observed"]], hide_index=True,
                     width="stretch")


def page_benchmark() -> None:
    st.title("Benchmark — four arms")
    report, table, _, rounds = cached_benchmark()
    if report is None or rounds.empty:
        st.info("No benchmark telemetry found under results/benchmark/.")
        return
    labels = [data.ARM_LABELS[a] for a in data.ARM_ORDER if a in set(rounds["arm"])]
    chosen = st.multiselect("Arms", labels, default=labels)
    keep = [a for a in data.ARM_ORDER if data.ARM_LABELS[a] in chosen]
    df = rounds[rounds["arm"].isin(keep)]

    left, right = st.columns(2)
    with left:
        show(arm_lines(df, "accuracy", "Accuracy (%)", "Global test accuracy", scale=100))
    with right:
        show(arm_lines(df, "mb_cumulative", "MB", "Cumulative uplink traffic"))
    left, right = st.columns(2)
    with left:
        show(arm_lines(df, "loss", "Loss", "Global test loss"))
    with right:
        show(arm_lines(df, "ade", "ADE", "Trajectory error (ADE, lower is better)"))

    st.subheader("Final metrics")
    st.dataframe(table[table["arm"].isin(keep)].drop(columns=["arm"]), hide_index=True,
                 width="stretch",
                 column_config={
                     "Accuracy (%)": st.column_config.NumberColumn(format="%.2f"),
                     "Worst client (%)": st.column_config.NumberColumn(format="%.1f"),
                     "MB / round": st.column_config.NumberColumn(format="%.2f"),
                     "Total MB": st.column_config.NumberColumn(format="%.1f"),
                     "Client acc. std": st.column_config.NumberColumn(format="%.4f"),
                 })

    st.subheader("Fairness across vehicles")
    fair = table[table["arm"].isin(keep)]
    fig = go.Figure(go.Bar(
        x=fair["Arm"], y=fair["Worst client (%)"],
        marker_color=[data.ARM_COLORS.get(a, "#888") for a in fair["arm"]],
        text=[f"{v:.1f}%" for v in fair["Worst client (%)"]], textposition="outside",
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>"))
    fig.update_layout(title="Worst-served vehicle accuracy", showlegend=False)
    show(_style(fig, "Accuracy (%)", x_title=""))


def page_ablation() -> None:
    st.title("Ablation — which enhancement does the work")
    study = st.radio("Study", list(data.ABLATION_STUDIES),
                     format_func=lambda k: data.ABLATION_STUDIES[k][1], horizontal=True)
    table = data.ablation_table(study)
    if table.empty:
        st.info("No results for this study.")
        return
    metric = st.selectbox("Metric", ["Accuracy (%)", "Client acc. std", "Worst client (%)", "MB / round"])
    t = table.sort_values(metric, ascending=metric == "Client acc. std")
    fig = go.Figure(go.Bar(x=t[metric], y=t["Variant"], orientation="h",
                           marker_color="#2a78d6",
                           text=[f"{v:.3g}" for v in t[metric]], textposition="outside",
                           hovertemplate="%{y}: %{x:.4g}<extra></extra>"))
    fig.update_layout(title=metric, showlegend=False, yaxis=dict(autorange="reversed"))
    show(_style(fig, "", x_title=metric, height=80 + 50 * len(t)))
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format=f)
                                for c, f in [("Accuracy (%)", "%.1f"), ("Worst client (%)", "%.1f"),
                                             ("Client acc. std", "%.4f"), ("MB / round", "%.2f")]})


def page_launch() -> None:
    st.title("Launch a federated run")
    st.caption("Runs the same CLI you would by hand and streams each round from `rounds.jsonl`.")
    procs: Dict[str, dict] = st.session_state.setdefault("procs", {})

    with st.form("launch"):
        c1, c2, c3 = st.columns(3)
        preset = c1.selectbox("Base config", list(data.CONFIG_FILES), index=0)
        name = c2.text_input("Run name", value=f"dash_{time.strftime('%H%M%S')}")
        rounds = c3.number_input("Rounds (R_max)", 1, 200, 8)
        c1, c2, c3 = st.columns(3)
        clients = c1.number_input("Vehicles (N)", 2, 500, 12)
        fraction = c2.slider("Participation fraction (C)", 0.05, 1.0, 0.3, 0.05)
        epochs = c3.number_input("Local epochs (E)", 1, 20, 2)
        c1, c2, c3 = st.columns(3)
        quality = c1.toggle("Quality-aware aggregation", True)
        traj = c1.toggle("Trajectory-aware selection", True)
        dp_on = c2.toggle("Differential privacy", preset == "dp")
        epsilon = c2.number_input("ε budget", 0.5, 64.0, 8.0, disabled=False)
        topk = c3.slider("Top-k sparsity ratio", 0.02, 1.0, 0.25, 0.01)
        byz = c3.toggle("Byzantine-robust (trimmed mean)", False)
        submitted = st.form_submit_button("Start run", type="primary")

    if submitted:
        safe = "".join(ch for ch in name if ch.isalnum() or ch in "-_") or "dash_run"
        overrides = {
            "num_clients": int(clients), "train.rounds": int(rounds),
            "train.local_epochs": int(epochs), "selection.fraction": float(fraction),
            "selection.trajectory_aware": traj, "aggregation.quality_aware": quality,
            "aggregation.byzantine_robust": byz, "dp.enabled": dp_on,
            "dp.epsilon": float(epsilon), "comm.topk_ratio": float(topk),
        }
        if byz:
            overrides["dp.secure_aggregation"] = False   # trimmed mean needs individual updates
        cmd = data.build_run_command(data.CONFIG_FILES[preset], safe, "runs", overrides)
        run_dir = os.path.join(data.REPO_ROOT, "runs", safe)
        proc = data.launch_run(cmd, os.path.join(run_dir, "stdout.log"))
        procs[safe] = {"proc": proc, "dir": run_dir, "rounds": int(rounds), "cmd": "python " + " ".join(cmd[1:])}
        st.session_state["watch"] = safe

    if not procs:
        st.info("Configure a run above and press **Start run**. Small fleets (N≈12, 8 rounds) "
                "finish in about 30 s on CPU.")
        return
    watch = st.selectbox("Watching", list(procs)[::-1], key="watch")
    live_panel(procs[watch])


@st.fragment(run_every=2)
def live_panel(job: dict) -> None:
    proc, run_dir, total = job["proc"], job["dir"], job["rounds"]
    code = proc.poll()
    df = data.load_rounds(run_dir)
    done = len(df)

    st.code(job["cmd"], language="bash")
    if code is None:
        st.progress(min(done / total, 1.0), text=f"Running — round {done} of {total}")
        if st.button("Stop run"):
            proc.terminate()
    elif code == 0:
        st.success(f"Finished — {done} rounds.")
    else:
        st.error(f"Exited with code {code}.")
        log = os.path.join(run_dir, "stdout.log")
        if os.path.exists(log):
            with open(log, encoding="utf-8", errors="replace") as fh:
                st.code(fh.read()[-3000:])

    if df.empty:
        st.caption("Waiting for the first round (building the fleet and model)…")
        return
    run_charts(df)
    if code == 0:
        summary = data.load_summary(run_dir)
        if summary:
            with st.expander("summary.json"):
                st.json(summary)


def run_charts(df: pd.DataFrame) -> None:
    last = df.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{100 * last.get('accuracy', float('nan')):.1f}%",
              f"best {100 * df['accuracy'].max():.1f}%" if "accuracy" in df else None,
              delta_color="off", delta_arrow="off")
    if "mb_cumulative" in df:
        c2.metric("Uplink so far", f"{last['mb_cumulative']:.1f} MB",
                  f"{df['mb_round'].mean():.2f} MB/round" if "mb_round" in df else None,
                  delta_color="off", delta_arrow="off")
    if "clients_reported" in df:
        c3.metric("Vehicles reporting", f"{int(last['clients_reported'])} / "
                  f"{int(last.get('clients_selected', last['clients_reported']))}")
    if "epsilon_spent" in df:
        c4.metric("Privacy ε spent", f"{last['epsilon_spent']:.3g}")

    left, right = st.columns(2)
    with left:
        show(single_line(df, "accuracy", "Global accuracy", "Accuracy (%)", scale=100))
        show(single_line(df, "client_acc_std", "Cross-vehicle accuracy std", "Std", color="#eb6834"))
        show(single_line(df, "mean_q", "Mean quality weight q̄", "q", color="#1baf7a"))
    with right:
        show(single_line(df, "loss", "Global test loss", "Loss", color="#4a3aa7"))
        show(single_line(df, "mb_round", "Uplink per round", "MB", color="#eda100"))
        show(single_line(df, "delta_norm", "Update norm ‖Δw‖", "Norm", color="#e87ba4"))


def page_explorer() -> None:
    st.title("Run explorer")
    runs = data.discover_runs()
    if not runs:
        st.info("No runs found under runs/ or results/.")
        return
    labels = {f"{'🟢' if r.source == 'live' else '📦'} {r.name}": r for r in runs}
    pick = st.selectbox("Run", list(labels))
    run = labels[pick]
    df = data.load_rounds(run.path)
    if df.empty:
        st.warning("This run has no rounds yet.")
        return
    run_charts(df)
    tabs = st.tabs(["Per-round table", "Summary", "Config"])
    with tabs[0]:
        st.dataframe(df, hide_index=True, width="stretch")
        st.download_button("Download CSV", df.to_csv(index=False).encode(),
                           file_name=f"{os.path.basename(run.path)}_rounds.csv")
    with tabs[1]:
        st.json(data.load_summary(run.path) or {"note": "no summary.json (run unfinished?)"})
    with tabs[2]:
        st.json(data.load_config(run.path) or {"note": "no config.json"})


def page_validation() -> None:
    st.title("Mathematical validation")
    st.caption("Generated by `scripts/make_proof_figures.py`; explained in "
               "docs/MATHEMATICAL_VALIDATION.md.")
    figs = data.validation_figures()
    if not figs:
        st.info("No figures in docs/figures/.")
        return
    cols = st.columns(2)
    for i, path in enumerate(figs):
        title = os.path.splitext(os.path.basename(path))[0].split("_", 1)[-1].replace("_", " ")
        with cols[i % 2]:
            st.image(path, caption=title.capitalize(), width="stretch")


PAGES = {
    "Overview": page_overview,
    "Benchmark": page_benchmark,
    "Ablation": page_ablation,
    "Launch a run": page_launch,
    "Run explorer": page_explorer,
    "Validation": page_validation,
}

with st.sidebar:
    st.header("FL-AV")
    page = st.radio("Page", list(PAGES), label_visibility="collapsed")
    st.divider()
    if st.button("Reload results"):
        st.cache_data.clear()
    st.caption("Synthetic data, simulated V2I. Relative ordering of arms is what's measured.")

PAGES[page]()
