"""FL-AV Streamlit dashboard.

    streamlit run dashboard/app.py

Pages
-----
Overview        headline numbers + PRD 12.3 acceptance
Fleet           every vehicle: profile, participation, per-car accuracy
Benchmark       the four arms of PRD 12.2, round by round
Ablation        which enhancement does the work
Patent evidence multi-seed technical effect, drawings (docs/patent/)
Optional mech.  each checklist option against the proposed method
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
    vr = data.load_vehicle_rounds(run_dir)
    if not vr.empty:
        participation_heatmap(vr)
        if code is not None:
            st.caption("Full per-vehicle view: **Fleet (every vehicle)** page.")
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


# ------------------------------------------------------------------ fleet / cars

REGION_COLORS = {"urban": "#2a78d6", "highway": "#eb6834", "rural": "#1baf7a"}
STATUS_STYLE = {   # status palette: reserved colours, always shown with a label
    "reported": ("#1baf7a", "✅ reported"),
    "not_selected": ("#d9d8d4", "· not selected"),
    "dropped": ("#e34948", "❌ dropped (link lost)"),
    "straggler": ("#eda100", "⏱ straggler (timed out)"),
    "corrupt": ("#eb6834", "⚠ corrupt payload"),
    "rejected": ("#4a3aa7", "🛡 rejected (Byzantine filter)"),
}


def _mtime(run_dir: str) -> float:
    paths = [os.path.join(run_dir, f) for f in
             ("vehicles.json", "vehicle_rounds.jsonl", "vehicle_summary.json", "config.json")]
    return max((os.path.getmtime(p) for p in paths if os.path.exists(p)), default=0.0)


@st.cache_data(show_spinner="Loading fleet…")
def cached_fleet(run_dir: str, _stamp: float):
    return data.fleet_table(run_dir), data.load_vehicle_rounds(run_dir)


def fleet_runs() -> List[data.RunInfo]:
    runs = [r for r in data.discover_runs() if data.has_vehicle_data(r.path)]
    def key(r):  # federated runs with per-round vehicle telemetry first
        has_rounds = os.path.exists(os.path.join(r.path, "vehicle_rounds.jsonl"))
        return (not has_rounds, not r.name.endswith("proposed_fl"), r.source != "live", r.name)
    return sorted(runs, key=key)


def page_fleet() -> None:
    st.title("Fleet — every vehicle")
    st.caption("Who each car is, what it contributed every round, and how well the shared "
               "model serves it on its own held-out data.")
    runs = fleet_runs()
    if not runs:
        st.info("No runs found. Launch one from **Launch a run**.")
        return
    labels = {f"{'🟢' if r.source == 'live' else '📦'} {r.name}": r for r in runs}
    run = labels[st.selectbox("Run", list(labels))]
    fleet_view(run.path, key=run.name)


def fleet_view(run_dir: str, key: str = "fleet", compact: bool = False) -> None:
    table, rounds = cached_fleet(run_dir, _mtime(run_dir))
    if table.empty:
        st.warning("No vehicle data for this run.")
        return
    if not os.path.exists(os.path.join(run_dir, "vehicles.json")):
        st.caption("ℹ️ Profiles rebuilt exactly from this run's config.json (the fleet is "
                   "deterministic in config + seed). Per-vehicle performance needs a run made "
                   "with the per-vehicle telemetry.")
    acc_col = "final_accuracy" if "final_accuracy" in table else (
        "latest_accuracy" if "latest_accuracy" in table else None)

    # ---- headline
    c = st.columns(6)
    c[0].metric("Vehicles", len(table))
    c[1].metric("Training samples", f"{int(table['n_train'].sum()):,}")
    c[2].metric("Uncalibrated sensors", int((~table["calibrated"]).sum()))
    if "times_reported" in table:
        c[3].metric("Never contributed", int((table["times_reported"].fillna(0) == 0).sum()))
    if acc_col:
        acc = table[acc_col].dropna()
        c[4].metric("Mean vehicle accuracy", f"{100 * acc.mean():.1f}%")
        worst = table.loc[table[acc_col].idxmin()]
        c[5].metric("Worst-served vehicle", f"{100 * worst[acc_col]:.1f}%",
                    f"car #{int(worst['client_id'])} · {worst['region']}/{worst['weather']}",
                    delta_color="off", delta_arrow="off")
    if acc_col and "n_test" in table:
        which = "the best global checkpoint"
        g = data.load_rounds(run_dir)
        if acc_col == "final_accuracy" and "accuracy" in g and not g.empty:
            which += f" (round {int(g.loc[g['accuracy'].idxmax(), 'round'])})"
        elif acc_col == "latest_accuracy":
            which = "the latest global model (run in progress)"
        st.caption(f"Per-vehicle accuracy = {which} on each car's own test split "
                   f"({int(table['n_test'].min())}–{int(table['n_test'].max())} samples), "
                   "so single-car numbers carry sampling noise of several points.")

    tabs = st.tabs(["All vehicles", "Vehicle detail", "Rounds × vehicles",
                    "By region & weather", "Across methods"])

    # ---- table
    with tabs[0]:
        f1, f2, f3 = st.columns(3)
        regions = f1.multiselect("Region", sorted(table["region"].unique()), key=f"{key}_reg")
        weathers = f2.multiselect("Weather", sorted(table["weather"].unique()), key=f"{key}_wx")
        calib = f3.selectbox("Sensor calibration", ["all", "calibrated", "uncalibrated"],
                             key=f"{key}_cal")
        view = table
        if regions:
            view = view[view["region"].isin(regions)]
        if weathers:
            view = view[view["weather"].isin(weathers)]
        if calib != "all":
            view = view[view["calibrated"] == (calib == "calibrated")]
        show_cols = [c for c in [
            "client_id", "region", "weather", acc_col, "rank", "calibrated", "byzantine",
            "n_train", "n_test",
            "link_uptime", "compute_capability", "sensor_quality", "label_similarity",
            "dominant_class", "times_selected", "times_reported", "times_dropped",
            "times_straggler", "times_rejected", "delivery_rate", "mean_q", "mean_alpha",
            "last_contribution_round", "final_mAP50", "final_ade", "final_fde",
            "final_loss"] if c and c in view]
        pct = dict(format="percent", min_value=0.0, max_value=1.0)
        st.dataframe(
            view[show_cols].sort_values(acc_col or "client_id"), hide_index=True,
            width="stretch", height=min(38 * (len(view) + 1), 640),
            column_config={
                "client_id": st.column_config.NumberColumn("Car #"),
                "link_uptime": st.column_config.ProgressColumn("V2I uptime", **pct),
                "sensor_quality": st.column_config.ProgressColumn("Sensor quality", **pct),
                "label_similarity": st.column_config.ProgressColumn("Label similarity", **pct),
                "delivery_rate": st.column_config.ProgressColumn("Delivery rate", **pct),
                (acc_col or "x"): st.column_config.ProgressColumn("Accuracy (own test)", **pct),
                "compute_capability": st.column_config.NumberColumn("Compute (×ref)", format="%.2f"),
                "mean_q": st.column_config.NumberColumn("Mean q_k", format="%.3f"),
                "mean_alpha": st.column_config.NumberColumn("Mean weight α", format="%.3f"),
                "final_mAP50": st.column_config.NumberColumn("mAP@0.5", format="%.3f"),
                "final_ade": st.column_config.NumberColumn("ADE", format="%.3f"),
                "final_fde": st.column_config.NumberColumn("FDE", format="%.3f"),
                "final_loss": st.column_config.NumberColumn("Loss", format="%.3f"),
                "rank": st.column_config.NumberColumn("Rank", format="%d"),
            })
        st.download_button("Download fleet CSV", view.to_csv(index=False).encode(),
                           file_name="fleet.csv", key=f"{key}_dl")

        if acc_col:
            ranked = view.sort_values(acc_col)
            fig = go.Figure()
            for region, color in REGION_COLORS.items():
                sub = ranked[ranked["region"] == region]
                fig.add_trace(go.Bar(
                    x=sub["client_id"].map(lambda c: f"car {c}"), y=100 * sub[acc_col], name=region,
                    marker_color=color,
                    customdata=sub[["weather", "n_test", "sensor_quality"]],
                    hovertemplate="%{x}<br>%{y:.1f}% accuracy<br>weather %{customdata[0]}"
                                  "<br>%{customdata[1]} test samples"
                                  "<br>sensor quality %{customdata[2]:.2f}<extra>%{fullData.name}</extra>"))
            fig.update_layout(title="Accuracy per vehicle (own test split), worst → best",
                              barmode="overlay", bargap=0.15)
            fig.update_xaxes(type="category", categoryorder="array",
                             categoryarray=ranked["client_id"].map(lambda c: f"car {c}").tolist())
            show(_style(fig, "Accuracy (%)", x_title="Vehicle"))

            left, right = st.columns(2)
            for col, xcol, xt in ((left, "sensor_quality", "Sensor quality"),
                                  (right, "label_similarity", "Label similarity to fleet")):
                fig = go.Figure()
                for region, color in REGION_COLORS.items():
                    sub = view[view["region"] == region]
                    fig.add_trace(go.Scatter(
                        x=sub[xcol], y=100 * sub[acc_col], mode="markers", name=region,
                        marker=dict(color=color, size=10, line=dict(width=2, color="white")),
                        text=sub["client_id"].map(lambda c: f"car #{c}"),
                        hovertemplate="%{text}<br>" + xt + " %{x:.2f}<br>%{y:.1f}%<extra></extra>"))
                fig.update_layout(title=f"{xt} vs. accuracy", hovermode="closest")
                with col:
                    show(_style(fig, "Accuracy (%)", x_title=xt))

    # ---- one vehicle
    with tabs[1]:
        ids = table["client_id"].astype(int).tolist()
        default = int(table.loc[table[acc_col].idxmin(), "client_id"]) if acc_col else ids[0]
        cid = st.selectbox("Vehicle", ids, index=ids.index(default), key=f"{key}_car",
                           format_func=lambda c: _car_label(table, c))
        vehicle_detail(table[table["client_id"] == cid].iloc[0], rounds, acc_col)

    # ---- heatmaps
    with tabs[2]:
        if rounds.empty:
            st.info("This run has no per-round vehicle telemetry.")
        else:
            participation_heatmap(rounds)
            if "accuracy" in rounds:
                accuracy_heatmap(rounds, table)

    # ---- groups
    with tabs[3]:
        if not acc_col:
            st.info("No per-vehicle accuracy for this run.")
        else:
            left, right = st.columns(2)
            for col, by in ((left, "region"), (right, "weather")):
                g = data.group_accuracy(table, by, acc_col)
                fig = go.Figure(go.Bar(
                    x=g[by], y=100 * g["mean"], marker_color="#2a78d6",
                    error_y=dict(type="data", symmetric=False,
                                 array=100 * (g["max"] - g["mean"]),
                                 arrayminus=100 * (g["mean"] - g["min"]), color="#888"),
                    customdata=g[["vehicles", "min", "max"]],
                    hovertemplate="%{x}: mean %{y:.1f}%<br>%{customdata[0]} vehicles"
                                  "<br>range %{customdata[1]:.1%} – %{customdata[2]:.1%}<extra></extra>"))
                fig.update_layout(title=f"Accuracy by {by} (bar = mean, whisker = min–max)",
                                  showlegend=False)
                with col:
                    show(_style(fig, "Accuracy (%)", x_title=""))
            g = table.dropna(subset=[acc_col]).assign(sensors=table["calibrated"].map(
                {True: "calibrated", False: "uncalibrated"}))
            g = data.group_accuracy(g, "sensors", acc_col)
            st.dataframe(g.assign(mean=100 * g["mean"], min=100 * g["min"], max=100 * g["max"]),
                         hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(f"{c} acc. (%)", format="%.1f")
                                        for c in ("mean", "min", "max")})

    # ---- across arms
    with tabs[4]:
        across_methods(table)


def _car_label(table: pd.DataFrame, cid: int) -> str:
    row = table[table["client_id"] == cid].iloc[0]
    flags = ("" if row["calibrated"] else " · uncalibrated") + (" · byzantine" if row["byzantine"] else "")
    return f"Car #{cid} — {row['region']}, {row['weather']}{flags}"


def vehicle_detail(row: pd.Series, rounds: pd.DataFrame, acc_col: Optional[str]) -> None:
    cid = int(row["client_id"])
    c = st.columns(5)
    c[0].metric("Region / weather", f"{row['region']} / {row['weather']}")
    c[1].metric("Training samples", int(row["n_train"]), f"{int(row['n_test'])} test",
                delta_color="off", delta_arrow="off")
    c[2].metric("V2I uptime", f"{row['link_uptime']:.0%}")
    c[3].metric("Sensor quality", f"{row['sensor_quality']:.2f}",
                "calibrated" if row["calibrated"] else "uncalibrated",
                delta_color="normal" if row["calibrated"] else "inverse", delta_arrow="off")
    if acc_col and pd.notna(row.get(acc_col)):
        c[4].metric("Accuracy (own test)", f"{100 * row[acc_col]:.1f}%",
                    f"rank {int(row['rank'])}" if "rank" in row and pd.notna(row["rank"]) else None,
                    delta_color="off", delta_arrow="off")

    left, right = st.columns([2, 1])
    with right:
        shares = row.filter(like="share_")
        fig = go.Figure(go.Bar(x=100 * shares.values, y=[s[6:] for s in shares.index],
                               orientation="h", marker_color="#2a78d6",
                               hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
        fig.update_layout(title="What this car sees (label mix)", showlegend=False,
                          yaxis=dict(autorange="reversed"))
        show(_style(fig, "", x_title="% of training samples", height=300))
        facts = {k: row.get(k) for k in ("compute_capability", "label_similarity", "times_selected",
                                         "times_reported", "times_dropped", "mean_q", "mean_alpha",
                                         "final_mAP50", "final_ade", "final_fde") if k in row}
        st.dataframe(pd.DataFrame({"value": facts}).astype(str), width="stretch")
    with left:
        mine = rounds[rounds["client_id"] == cid] if not rounds.empty else rounds
        if mine.empty:
            st.info("No per-round telemetry for this car in this run.")
            return
        fig = go.Figure()
        if "accuracy" in mine:
            fig.add_trace(go.Scatter(x=mine["round"], y=100 * mine["accuracy"], mode="lines",
                                     name="global model on this car", line=dict(color="#2a78d6"),
                                     hovertemplate="round %{x}: %{y:.1f}%<extra></extra>"))
        for status, (color, label) in STATUS_STYLE.items():
            if status == "not_selected":
                continue
            ev = mine[mine["status"] == status]
            if ev.empty:
                continue
            y = 100 * ev["accuracy"] if "accuracy" in ev else [0] * len(ev)
            fig.add_trace(go.Scatter(x=ev["round"], y=y, mode="markers", name=label,
                                     marker=dict(color=color, size=10, line=dict(width=2, color="white")),
                                     hovertemplate="round %{x}: " + label + "<extra></extra>"))
        fig.update_layout(title=f"Car #{cid}: accuracy each round, and when it took part",
                          hovermode="closest")
        show(_style(fig, "Accuracy (%)"))
        cols = [c for c in ("round", "status", "accuracy", "q_k", "alpha", "local_epochs",
                            "train_loss", "update_norm", "uplink_kb") if c in mine]
        st.dataframe(mine[mine["status"] != "not_selected"][cols], hide_index=True,
                     width="stretch", height=260)


def participation_heatmap(rounds: pd.DataFrame) -> None:
    order = list(STATUS_STYLE)
    code = {s: i for i, s in enumerate(order)}
    grid = rounds.pivot_table(index="client_id", columns="round", values="status",
                              aggfunc="first").reindex(columns=sorted(rounds["round"].unique()))
    z = grid.apply(lambda col: col.map(code)).astype(float)
    n = len(order)
    scale = []
    for i, s in enumerate(order):
        scale += [(i / n, STATUS_STYLE[s][0]), ((i + 1) / n, STATUS_STYLE[s][0])]
    fig = go.Figure(go.Heatmap(
        z=z.values, x=z.columns, y=[f"car {c}" for c in z.index], zmin=-0.5, zmax=n - 0.5,
        colorscale=scale, xgap=1, ygap=1, text=grid.values,
        hovertemplate="%{y} · round %{x}<br>%{text}<extra></extra>",
        colorbar=dict(tickvals=list(range(n)), ticktext=[STATUS_STYLE[s][1] for s in order],
                      title=None)))
    fig.update_layout(title="Who took part each round", yaxis=dict(type="category"))
    show(_style(fig, "Vehicle", height=max(320, 14 * len(z) + 80)))


def accuracy_heatmap(rounds: pd.DataFrame, table: pd.DataFrame) -> None:
    grid = rounds.pivot_table(index="client_id", columns="round", values="accuracy")
    final = grid.iloc[:, -1].sort_values()
    grid = grid.loc[final.index]
    fig = go.Figure(go.Heatmap(
        z=100 * grid.values, x=grid.columns, y=[f"car {c}" for c in grid.index],
        colorscale=[(0, "#eef4fc"), (1, "#0d3d75")], zmin=0, zmax=100, xgap=1, ygap=1,
        hovertemplate="%{y} · round %{x}<br>%{z:.1f}%<extra></extra>",
        colorbar=dict(title="Acc. %")))
    fig.update_layout(title="Global model accuracy on each car, round by round (worst at bottom)",
                      yaxis=dict(type="category"))
    show(_style(fig, "Vehicle", height=max(320, 14 * len(grid) + 80)))


def across_methods(table: pd.DataFrame) -> None:
    wide = data.compare_arms_per_vehicle()
    if wide.empty or wide.shape[1] < 2:
        st.info("Per-vehicle results for the benchmark arms are not available yet. Run "
                "`python -m fedav.cli benchmark --config configs/default.yaml` with this "
                "version and copy the arm folders into results/benchmark/.")
        return
    st.markdown("**Each car's accuracy under every method** (own test split)")
    long = wide.reset_index(names="client_id").melt("client_id", var_name="arm",
                                                      value_name="acc").dropna()
    order = (wide["proposed_fl"] if "proposed_fl" in wide else wide.iloc[:, 0]).sort_values().index
    fig = go.Figure()
    for arm in data.ARM_ORDER:
        sub = long[long["arm"] == arm].set_index("client_id").reindex(order).dropna()
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=[f"car {c}" for c in sub.index], y=100 * sub["acc"], mode="markers",
            name=data.ARM_LABELS[arm],
            marker=dict(color=data.ARM_COLORS[arm], size=9, line=dict(width=2, color="white")),
            hovertemplate="%{x}: %{y:.1f}%<extra>" + data.ARM_LABELS[arm] + "</extra>"))
    fig.update_xaxes(type="category", categoryorder="array",
                     categoryarray=[f"car {c}" for c in order])
    show(_style(fig, "Accuracy (%)", x_title="Vehicle (sorted by Proposed FL)"))
    if {"proposed_fl", "std_fedavg"} <= set(wide):
        diff = (wide["proposed_fl"] - wide["std_fedavg"]).dropna()
        c = st.columns(3)
        c[0].metric("Cars better off with Proposed FL", int((diff > 0).sum()))
        c[1].metric("Cars worse off", int((diff < 0).sum()))
        c[2].metric("Unchanged", int((diff == 0).sum()))
        merged = table[["client_id", "region", "weather", "calibrated"]].merge(
            (100 * wide).round(1).reset_index(names="client_id"), on="client_id")
        merged["Δ proposed − FedAvg (pts)"] = (100 * diff).round(1).reindex(merged["client_id"]).values
        st.dataframe(merged.sort_values("Δ proposed − FedAvg (pts)").rename(columns=data.ARM_LABELS),
                     hide_index=True, width="stretch")
    if "local_only" in wide:
        st.caption("Local-only trains 12 of the vehicles alone; the others have no local-only model.")



# ------------------------------------------------------------- patent evidence

def _arm_bars(df: pd.DataFrame, title: str, y_title: str) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=df["Arm"], y=df["mean"], marker_color=[data.PATENT_COLORS[a] for a in df["arm"]],
        error_y=dict(type="data", array=df["ci95"], color="#888"),
        text=[f"{v:.3g}" for v in df["mean"]], textposition="outside",
        hovertemplate="%{x}: %{y:.4g}<extra></extra>"))
    fig.update_layout(title=title, showlegend=False)
    return _style(fig, y_title, x_title="", height=320)


def page_patent() -> None:
    st.title("Patent evidence (India filing)")
    st.caption("Multi-seed comparison behind the Form 2 working example. Bars: mean over seeds; "
               "whiskers: 95% CI. Arm B uses the same compression transport as the invention, so "
               "D − B is what the claimed mechanisms add. See docs/patent/.")
    effects = data.load_patent_effects()
    if effects is None:
        st.info("No results yet. Run `python scripts/run_patent_evidence.py`.")
        return
    c = st.columns(4)
    for col, (metric, label, unit, digits) in zip(c, [
            ("uplink_airtime_s_per_round", "Uplink airtime / round", "s", 1),
            ("max_round_airtime_s", "Longest round airtime", "s", 1),
            ("uplink_energy_j_per_round", "Radio energy / round", "J", 1),
            ("mb_per_round", "Traffic / round", "MB", 2)]):
        d = data.patent_table(effects, metric).set_index("arm")
        p = data.patent_paired(effects, "proposed_tiered", metric) or {}
        if "proposed_tiered" in d.index:
            col.metric(f"{label} (D)", f"{d.loc['proposed_tiered', 'mean']:.{digits}f} {unit}",
                       f"{p.get('mean', 0):+.{digits}f} ± {p.get('ci95') or 0:.{digits}f} vs B",
                       delta_color="inverse")
    st.warning("Accuracy does not differ significantly between arms at three seeds; the "
               "invention is claimed for airtime and energy, not accuracy.")
    left, right = st.columns(2)
    for i, (metric, title, unit, scale) in enumerate([
            ("uplink_airtime_s_per_round", "Uplink airtime per round", "s", 1),
            ("uplink_energy_j_per_round", "Uplink radio energy per round", "J", 1),
            ("mb_per_round", "Traffic per round", "MB", 1),
            ("best_accuracy", "Best test accuracy", "%", 100)]):
        with (left if i % 2 == 0 else right):
            show(_arm_bars(data.patent_table(effects, metric, scale), title, unit))
    with st.expander("Full tables (results/patent/effects.md)"):
        path = os.path.join(data.REPO_ROOT, "results", "patent", "effects.md")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                st.markdown(fh.read())
    st.subheader("Patent drawings (FIG. 1–9)")
    figs = data.patent_drawings()
    cols = st.columns(3)
    for i, f in enumerate(figs):
        with cols[i % 3]:
            st.image(f, caption=os.path.basename(f).replace(".png", "").replace("FIG0", "FIG. "),
                     width="stretch")


def page_options() -> None:
    st.title("Optional mechanisms")
    st.caption("Each option switched on alone over the proposed configuration (30 vehicles, "
               "9 classes with rare hazards, 20 rounds). Bars: mean of option − base per seed; "
               "whiskers: 95% CI. A whisker crossing zero means no measurable effect.")
    options = data.load_options()
    if options is None:
        st.info("No results yet. Run `python scripts/run_option_ablation.py`.")
        return
    metric_labels = {"best_accuracy": ("Best accuracy", 100), "best_rare_class_accuracy":
                     ("Best rare-hazard accuracy", 100), "worst_10pct_vehicle_accuracy":
                     ("Worst-10%-of-vehicles accuracy", 100), "worst_vehicle_accuracy":
                     ("Worst-vehicle accuracy", 100), "mb_per_round": ("Traffic per round (MB)", 1)}
    metric = st.selectbox("Metric", list(metric_labels), format_func=lambda m: metric_labels[m][0])
    label, scale = metric_labels[metric]
    df = data.options_paired(options, metric, scale)
    if df.empty:
        st.info("No paired results for this metric.")
        return
    fig = go.Figure(go.Bar(
        x=df["mean"], y=df["option"], orientation="h", marker_color="#2a78d6",
        error_x=dict(type="data", array=df["ci95"], color="#888"),
        hovertemplate="%{y}: %{x:+.3g}<extra></extra>"))
    fig.add_vline(x=0, line_color="#888")
    fig.update_layout(title=f"{label}: option − base", showlegend=False,
                      yaxis=dict(autorange="reversed"))
    show(_style(fig, "", x_title="difference", height=80 + 42 * len(df)))
    path = os.path.join(data.REPO_ROOT, "results", "options", "options.md")
    if os.path.exists(path):
        with st.expander("Full tables (results/options/options.md)"):
            with open(path, encoding="utf-8") as fh:
                st.markdown(fh.read())


PAGES = {
    "Overview": page_overview,
    "Fleet (every vehicle)": page_fleet,
    "Benchmark": page_benchmark,
    "Ablation": page_ablation,
    "Patent evidence": page_patent,
    "Optional mechanisms": page_options,
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
