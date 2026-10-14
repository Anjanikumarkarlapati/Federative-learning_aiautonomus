"""FL-AV Streamlit dashboard.

    streamlit run dashboard/app.py

Pages
-----
Overview        headline numbers, learning curves, cost vs. accuracy, PRD 12.3 acceptance
Fleet           every vehicle: profile, participation, per-car accuracy
Benchmark       the four arms of PRD 12.2: curves, efficiency, fairness, operations, config
Ablation        which enhancement does the work, as deltas against the baseline
Compare runs    any runs side by side: curves, final metrics, config differences
Launch a run    configure, start and watch a federated run live
Run explorer    any run under runs/ or results/
Validation      the 14 mathematical-validation figures with their explanations
"""
from __future__ import annotations

import os
import re
import sys
import time
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard import data  # noqa: E402

st.set_page_config(page_title="FL-AV Dashboard", page_icon="🚗", layout="wide",
                   initial_sidebar_state="expanded")

GRID = dict(showgrid=True, gridwidth=1, zeroline=False)
# Categorical slots in fixed order (validated: adjacent pairs clear the CVD floor).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GOOD, BAD, NEUTRAL = "#2a78d6", "#e34948", "#a3a29c"   # diverging poles + gray midpoint

st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; padding-bottom: 3rem;}
  [data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.28); border-radius: .6rem;
                            padding: .65rem .9rem;}
  [data-testid="stMetricValue"] {font-size: 1.75rem;}
  [data-testid="stSidebar"] [data-testid="stRadio"] label p {font-size: 0.95rem;}
  .fl-sub {color: var(--text-color, inherit); opacity: .72; margin: -.6rem 0 1rem 0;}
  .fl-chip {display:inline-block; padding:.15rem .6rem; margin:0 .35rem .35rem 0;
            border-radius:999px; font-size:.82rem; border:1px solid rgba(128,128,128,.35);}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ layout helpers

def header(title: str, subtitle: str = "", how_to_read: str = "") -> None:
    st.title(title)
    if subtitle:
        html = re.sub(r"`([^`]+)`", r"<code>\1</code>", subtitle)
        st.markdown(f"<p class='fl-sub'>{html}</p>", unsafe_allow_html=True)
    if how_to_read:
        with st.expander("How to read this page"):
            st.markdown(how_to_read)


def chips(items: Dict[str, object]) -> None:
    html = "".join(f"<span class='fl-chip'><b>{k}</b> {v}</span>" for k, v in items.items()
                   if v is not None)
    st.markdown(html, unsafe_allow_html=True)


def fleet_chips(cfg: dict) -> None:
    chips({"Vehicles N": cfg.get("num_clients"),
           "Participation C": cfg.get("selection", {}).get("fraction"),
           "Local epochs E": cfg.get("train", {}).get("local_epochs"),
           "Rounds R_max": cfg.get("train", {}).get("rounds"),
           "Dirichlet α": cfg.get("data", {}).get("dirichlet_alpha"),
           "Seed": cfg.get("seed")})


def download(df: pd.DataFrame, name: str, key: str, label: str = "Download CSV") -> None:
    st.download_button(f"⬇ {label}", df.to_csv(index=False).encode(), file_name=name,
                       mime="text/csv", key=key)


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


def arm_lines(df: pd.DataFrame, y: str, y_title: str, title: str, scale: float = 1.0,
              x: str = "round", x_title: str = "Round", window: int = 1) -> go.Figure:
    fig = go.Figure()
    for arm in data.ARM_ORDER:
        sub = df[df["arm"] == arm].dropna(subset=[y]) if y in df else pd.DataFrame()
        if sub.empty or x not in sub:
            continue
        fig.add_trace(go.Scatter(
            x=sub[x], y=data.smooth(sub[y], window) * scale, name=data.ARM_LABELS[arm],
            mode="lines", line=dict(color=data.ARM_COLORS[arm]),
            hovertemplate="%{y:.3g}"))
    fig.update_layout(title=title)
    return _style(fig, y_title, x_title=x_title)


def metric_lines(df: pd.DataFrame, group: str, y: str, colors: Dict[str, str],
                 window: int = 1, x: str = "round", x_title: str = "Round") -> Optional[go.Figure]:
    """One line per group for a metric from data.ROUND_METRICS."""
    if y not in df:
        return None
    title, unit, scale, better = data.ROUND_METRICS.get(y, (y, y, 1, "—"))
    fig = go.Figure()
    for key, color in colors.items():
        sub = df[df[group] == key].dropna(subset=[y])
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(x=sub[x], y=data.smooth(sub[y], window) * scale, name=str(key),
                                 mode="lines", line=dict(color=color), hovertemplate="%{y:.4g}"))
    if not fig.data:
        return None
    hint = {"higher": " ↑ higher is better", "lower": " ↓ lower is better"}.get(better, "")
    fig.update_layout(title=title + hint)
    return _style(fig, unit, x_title=x_title, height=300)


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


def show(fig: Optional[go.Figure], caption: str = "") -> None:
    if fig is None:
        return
    title = fig.layout.title.text
    if title:
        st.markdown(f"**{title}**")
        fig.update_layout(title=None)
    st.plotly_chart(fig, width="stretch", theme="streamlit")
    if caption:
        st.caption(caption)


def grid_of(figs: List[Optional[go.Figure]], cols: int = 2) -> None:
    figs = [f for f in figs if f is not None]
    for i in range(0, len(figs), cols):
        for col, fig in zip(st.columns(cols), figs[i:i + cols]):
            with col:
                show(fig)


def delta_bars(names: List[str], deltas: List[float], title: str, unit: str,
               higher_is_better: bool = True) -> go.Figure:
    """Diverging bars around zero: blue = better than baseline, red = worse."""
    colors = [NEUTRAL if abs(d) < 1e-9 else (GOOD if (d > 0) == higher_is_better else BAD)
              for d in deltas]
    fig = go.Figure(go.Bar(x=deltas, y=names, orientation="h", marker_color=colors,
                           text=[f"{d:+.2f}" for d in deltas], textposition="outside",
                           cliponaxis=False,
                           hovertemplate="%{y}: %{x:+.3f} " + unit + "<extra></extra>"))
    fig.update_layout(title=title, showlegend=False, yaxis=dict(autorange="reversed"),
                      hovermode="closest")
    fig.add_vline(x=0, line_width=1, line_color=NEUTRAL)
    m = max([abs(d) for d in deltas] + [1e-9])
    fig.update_xaxes(range=[-1.45 * m, 1.45 * m])    # room for the outside labels on both sides
    return _style(fig, "", x_title=unit, height=90 + 42 * len(names))


# --------------------------------------------------------------------- cached io

@st.cache_data(ttl=30)
def cached_benchmark():
    report = data.load_benchmark()
    if report is None:
        return None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    return report, data.benchmark_table(report), data.acceptance_table(report), data.benchmark_rounds()


@st.cache_data(ttl=30)
def cached_benchmark_full():
    report = data.load_benchmark()
    return pd.DataFrame() if report is None else data.benchmark_full_table(report)


@st.cache_data(ttl=60)
def cached_per_vehicle():
    return data.compare_arms_per_vehicle()


# ------------------------------------------------------------------------- pages

def page_overview() -> None:
    header("Federated learning for collaborative AV training",
           "A fleet trains one shared perception / trajectory model. Only model deltas leave "
           "the vehicle; the road-side unit aggregates them.",
           "- **Proposed FL** is this framework; **Std. FedAvg** is the textbook baseline; "
           "**Centralized** uploads all raw data (the accuracy ceiling, no privacy); "
           "**Local-only** never shares anything (the privacy floor).\n"
           "- Deltas under each number compare Proposed FL with Std. FedAvg.\n"
           "- Everything here is read from `results/benchmark/`; press **Reload results** in "
           "the sidebar after a new benchmark.")
    report, table, acc, rounds = cached_benchmark()
    if report is None:
        st.info("No benchmark report yet. Run `python -m fedav.cli benchmark --config "
                "configs/default.yaml`, then press **Reload results**.")
        return
    full = cached_benchmark_full().set_index("arm")
    fleet_chips(report.get("config", {}))

    rows = table.set_index("arm")
    prop, fed, cen = (rows.loc[a] if a in rows.index else None
                      for a in ("proposed_fl", "std_fedavg", "centralized"))
    c = st.columns(5)
    if prop is not None and fed is not None:
        c[0].metric("Proposed FL accuracy", f"{prop['Accuracy (%)']:.1f}%",
                    f"{prop['Accuracy (%)'] - fed['Accuracy (%)']:+.1f} pts vs FedAvg",
                    help="Best-round accuracy of the global model on the held-out global test set.")
        saving = 1 - prop["MB / round"] / fed["MB / round"]
        c[1].metric("Uplink per round", f"{prop['MB / round']:.1f} MB",
                    f"-{saving:.0%} vs FedAvg", delta_color="inverse",
                    help="Mean V2I uplink traffic per round across all reporting vehicles.")
        c[2].metric("Worst-served vehicle", f"{prop['Worst client (%)']:.1f}%",
                    f"{prop['Worst client (%)'] - fed['Worst client (%)']:+.1f} pts vs FedAvg",
                    help="Accuracy of the global model on the single worst vehicle's own test split.")
    if cen is not None and prop is not None:
        c[3].metric("Of centralized ceiling",
                    f"{prop['Accuracy (%)'] / cen['Accuracy (%)']:.1%}",
                    "with no raw data egress", delta_color="off", delta_arrow="off",
                    help="Proposed FL accuracy divided by the accuracy of training on all raw data.")
    thr = 0.80
    hits = data.rounds_to_threshold(rounds, thr)
    if hits.get("proposed_fl") is not None:
        fed_hit = hits.get("std_fedavg")
        c[4].metric(f"Rounds to {thr:.0%}", hits["proposed_fl"],
                    f"{hits['proposed_fl'] - fed_hit:+d} vs FedAvg" if fed_hit else "FedAvg never got there",
                    delta_color="inverse",
                    help=f"First round at which the global test accuracy reached {thr:.0%}.")

    left, right = st.columns([3, 2])
    with left:
        show(arm_lines(rounds, "accuracy", "Accuracy (%)", "Learning curves — global test accuracy",
                       scale=100))
    with right:
        show(cost_vs_accuracy(full), "Total uplink over the whole run vs. best accuracy. "
             "Centralized moves raw data once instead of model updates every round.")

    st.subheader("Key findings")
    for line in key_findings(full, hits):
        st.markdown(f"- {line}")

    st.subheader("PRD 12.3 acceptance")
    if not acc.empty:
        passed = int(acc["pass"].sum())
        verdict = (report.get("acceptance") or {}).get("status") or (
            "PASS" if passed == len(acc) else "FAIL")
        st.progress(passed / len(acc), text=f"{passed} of {len(acc)} checks pass — overall {verdict}")
        view = acc.assign(Result=acc["pass"].map({True: "✅ Pass", False: "❌ Fail"}))
        st.dataframe(view[["Result", "Requirement", "Observed"]], hide_index=True, width="stretch")

    with st.expander("How one federated round works"):
        st.markdown(
            "1. **Select** — the RSU picks ⌈C·N⌉ vehicles with stable V2I links, stratified over "
            "regions, with a fairness guard so no car starves.\n"
            "2. **Train locally** — each selected car runs E epochs of SGD on its private shard.\n"
            "3. **Transmit** — top-k sparsified, fp16 + gzip, CRC-checked updates over TLS; "
            "optional DP noise and secure-aggregation masks.\n"
            "4. **Aggregate** — quality-weighted mean `α_k = q_k n_k / Σ q_j n_j`, server "
            "momentum, new global model; stop at `‖Δw‖ ≤ ε` or `R_max`.")


def cost_vs_accuracy(full: pd.DataFrame) -> Optional[go.Figure]:
    if full.empty:
        return None
    fig = go.Figure()
    xmax = float(full["Total MB"].max())
    for arm, row in full.iterrows():
        # Labels point away from the nearest neighbour and the plot edge.
        pos = "middle left" if row["Total MB"] > 0.6 * xmax else (
            "bottom right" if arm == "proposed_fl" else "top right")
        fig.add_trace(go.Scatter(
            x=[row["Total MB"]], y=[row["Accuracy (%)"]], mode="markers+text",
            name=row["Arm"], text=[f"  {row['Arm']}  "], textposition=pos,
            marker=dict(color=data.ARM_COLORS.get(arm, "#888"), size=14,
                        line=dict(width=2, color="white")),
            hovertemplate=f"{row['Arm']}<br>%{{x:,.0f}} MB total<br>%{{y:.1f}}% accuracy<extra></extra>"))
    fig.update_layout(title="Cost vs. accuracy", hovermode="closest", showlegend=False)
    fig.update_yaxes(range=[0, 105])
    fig.update_xaxes(range=[-0.06 * xmax, 1.08 * xmax])
    return _style(fig, "Accuracy (%)", x_title="Total uplink (MB)")


def key_findings(full: pd.DataFrame, hits: Dict[str, Optional[int]]) -> List[str]:
    out: List[str] = []
    if not {"proposed_fl", "std_fedavg"} <= set(full.index):
        return out
    p, f = full.loc["proposed_fl"], full.loc["std_fedavg"]
    out.append(f"Proposed FL matches FedAvg's accuracy ({p['Accuracy (%)']:.1f}% vs "
               f"{f['Accuracy (%)']:.1f}%) while sending **{f['Total MB'] / p['Total MB']:.1f}× less** "
               f"data ({p['Total MB']:,.0f} MB vs {f['Total MB']:,.0f} MB).")
    out.append(f"The worst-served vehicle improves from {f['Worst client (%)']:.1f}% to "
               f"**{p['Worst client (%)']:.1f}%**; the 10th-percentile vehicle sits at "
               f"{p['P10 client (%)']:.1f}% (FedAvg {f['P10 client (%)']:.1f}%).")
    if hits.get("proposed_fl") and hits.get("std_fedavg"):
        out.append(f"80% accuracy is reached in round **{hits['proposed_fl']}** vs "
                   f"{hits['std_fedavg']} for FedAvg.")
    if "centralized" in full.index:
        c = full.loc["centralized"]
        out.append(f"The privacy cost is {c['Accuracy (%)'] - p['Accuracy (%)']:.1f} accuracy points "
                   f"below the centralized ceiling ({c['Accuracy (%)']:.1f}%), which needs every raw "
                   "sensor sample to leave the car.")
    if "local_only" in full.index:
        out.append(f"Training alone on one car tops out at {full.loc['local_only', 'Accuracy (%)']:.1f}% — "
                   "collaboration is what makes the model usable.")
    return out


BENCH_HELP = (
    "- **Learning curves**: pick any logged metric; smoothing averages the last *w* rounds; "
    "switch the x-axis to cumulative uplink to compare at equal communication cost.\n"
    "- **Efficiency**: drag the threshold to see how many rounds and MB each arm needs to reach it.\n"
    "- **Fairness**: how the shared model serves individual cars, not just the fleet average.\n"
    "- **Operations**: participation, wall-clock time, learning-rate schedule and privacy spend.\n"
    "- **All metrics**: every number from each arm's `summary.json`, and the config keys that "
    "differ between arms.")


def page_benchmark() -> None:
    header("Benchmark — four arms", "PRD 12.2: centralized, local-only, standard FedAvg and the "
           "proposed framework, trained on the same fleet.", BENCH_HELP)
    report, table, _, rounds = cached_benchmark()
    if report is None or rounds.empty:
        st.info("No benchmark telemetry found under results/benchmark/.")
        return
    labels = [data.ARM_LABELS[a] for a in data.ARM_ORDER if a in set(rounds["arm"])]
    chosen = st.pills("Arms", labels, default=labels, selection_mode="multi")
    keep = [a for a in data.ARM_ORDER if data.ARM_LABELS[a] in (chosen or [])]
    if not keep:
        st.warning("Select at least one arm.")
        return
    df = rounds[rounds["arm"].isin(keep)]
    full = cached_benchmark_full()
    full = full[full["arm"].isin(keep)]

    tabs = st.tabs(["📈 Learning curves", "⚡ Efficiency", "⚖️ Fairness", "🛠 Operations",
                    "📋 All metrics"])
    colors = {data.ARM_LABELS[a]: data.ARM_COLORS[a] for a in keep}
    df = df.assign(Arm=df["arm"].map(data.ARM_LABELS))

    with tabs[0]:
        avail = data.available_metrics(df)
        c1, c2, c3 = st.columns([3, 1, 1])
        metrics = c1.multiselect("Metrics", avail, default=[m for m in
                                 ("accuracy", "loss", "ade", "client_acc_min") if m in avail],
                                 format_func=lambda m: data.ROUND_METRICS[m][0])
        window = c2.slider("Smoothing (rounds)", 1, 10, 1)
        xaxis = c3.segmented_control("X-axis", ["Round", "Cumulative MB"], default="Round")
        x, xt = ("mb_cumulative", "Cumulative uplink (MB)") if xaxis == "Cumulative MB" else ("round", "Round")
        grid_of([metric_lines(df, "Arm", m, colors, window, x=x, x_title=xt) for m in metrics])
        if xaxis == "Cumulative MB":
            st.caption("Centralized and Local-only send no model updates, so they collapse at 0 MB.")

    with tabs[1]:
        best = float(df["accuracy"].max())
        thr = st.slider("Accuracy threshold", 0.30, max(0.31, round(best, 2)),
                        min(0.80, round(best, 2)), 0.01, format="%.2f")
        hits = data.rounds_to_threshold(df, thr)
        rows = []
        for a in keep:
            r = hits.get(a)
            sub = df[(df["arm"] == a) & (df["round"] == r)] if r else pd.DataFrame()
            mb = float(sub["mb_cumulative"].iloc[0]) if not sub.empty and "mb_cumulative" in sub else None
            rows.append({"arm": a, "Arm": data.ARM_LABELS[a], "Rounds": r, "MB": mb})
        eff = pd.DataFrame(rows)
        left, right = st.columns(2)
        for col, field, unit in ((left, "Rounds", "rounds"), (right, "MB", "MB uplink")):
            reached = eff.dropna(subset=[field])
            fig = go.Figure(go.Bar(
                x=reached["Arm"], y=reached[field],
                marker_color=[data.ARM_COLORS[a] for a in reached["arm"]],
                text=[f"{v:,.0f}" for v in reached[field]], textposition="outside", cliponaxis=False,
                hovertemplate="%{x}: %{y:,.1f} " + unit + "<extra></extra>"))
            fig.update_layout(title=f"{field} needed to reach {thr:.0%}", showlegend=False,
                              hovermode="closest")
            with col:
                show(_style(fig, unit, x_title=""))
        missing = eff[eff["Rounds"].isna()]["Arm"].tolist()
        if missing:
            st.caption("Never reached the threshold: " + ", ".join(missing) + ".")
        fed = df[df["arm"].isin(["proposed_fl", "std_fedavg"])]
        if not fed.empty:
            show(arm_lines(fed, "accuracy", "Accuracy (%)", "Accuracy per MB sent (federated arms)",
                           scale=100, x="mb_cumulative", x_title="Cumulative uplink (MB)"),
                 "Further left at the same height = the same accuracy for less communication.")

    with tabs[2]:
        fair_cols = ["Mean client (%)", "P10 client (%)", "Worst client (%)"]
        fig = go.Figure()
        for _, row in full.iterrows():
            fig.add_trace(go.Bar(x=fair_cols, y=[row[c] for c in fair_cols], name=row["Arm"],
                                 marker_color=data.ARM_COLORS.get(row["arm"], "#888"),
                                 hovertemplate="%{x}: %{y:.1f}%<extra>" + row["Arm"] + "</extra>"))
        fig.update_layout(title="Per-vehicle accuracy: mean, 10th percentile, worst",
                          barmode="group", bargap=0.25, bargroupgap=0.08, hovermode="closest")
        show(_style(fig, "Accuracy (%)", x_title=""))
        wide = cached_per_vehicle()
        if not wide.empty:
            fig = go.Figure()
            for a in keep:
                if a in wide:
                    fig.add_trace(go.Box(y=100 * wide[a].dropna(), name=data.ARM_LABELS[a],
                                         marker_color=data.ARM_COLORS[a], boxpoints="all",
                                         jitter=0.4, pointpos=0, marker=dict(size=6),
                                         line=dict(width=1.5),
                                         hovertemplate="%{y:.1f}%<extra>" + data.ARM_LABELS[a] + "</extra>"))
            fig.update_layout(title="Distribution of per-vehicle accuracy (each dot is a car)",
                              showlegend=False, hovermode="closest")
            show(_style(fig, "Accuracy (%)", x_title="", height=380),
                 "Local-only has a model for 12 of the cars only. The **Fleet** page drills into "
                 "individual vehicles.")

    with tabs[3]:
        ops = [m for m in ("clients_reported", "round_seconds", "lr", "epsilon_spent", "delta_norm",
                           "mb_round") if m in df and df[m].fillna(0).abs().max() > 0]
        grid_of([metric_lines(df, "Arm", m, colors) for m in ops])
        if not full.empty:
            st.dataframe(full[["Arm", "Selection Gini", "Mean participation", "Never selected",
                               "Wall clock (min)", "Privacy", "Secure agg."]],
                         hide_index=True, width="stretch",
                         column_config={"Selection Gini": st.column_config.NumberColumn(
                                            format="%.3f", help="0 = every car selected equally often"),
                                        "Mean participation": st.column_config.NumberColumn(format="%.1f"),
                                        "Never selected": st.column_config.NumberColumn(format="%d"),
                                        "Wall clock (min)": st.column_config.NumberColumn(format="%.1f")})

    with tabs[4]:
        num = {c: st.column_config.NumberColumn(format=f) for c, f in [
            ("Accuracy (%)", "%.2f"), ("Mean client (%)", "%.1f"), ("P10 client (%)", "%.1f"),
            ("Worst client (%)", "%.1f"), ("MB / round", "%.2f"), ("Total MB", "%.1f"),
            ("Client acc. std", "%.4f"), ("mAP@0.5", "%.4f"), ("ADE", "%.4f"), ("FDE", "%.4f"),
            ("Selection Gini", "%.3f"), ("Wall clock (min)", "%.1f")]}
        st.dataframe(full.drop(columns=["arm"]), hide_index=True, width="stretch", column_config=num)
        download(full.drop(columns=["arm"]), "benchmark_summary.csv", "bench_dl")
        st.markdown("**Configuration differences between arms**")
        base = os.path.join(data.REPO_ROOT, "results", "benchmark")
        diff = data.config_diff({data.ARM_LABELS[a]: data.load_config(os.path.join(base, a))
                                 for a in keep})
        if diff.empty:
            st.caption("The selected arms share one configuration.")
        else:
            st.dataframe(diff.astype(str), width="stretch")


ABLATION_METRICS = {   # metric -> higher is better?
    "Accuracy (%)": True, "Worst client (%)": True, "P10 client (%)": True,
    "Client acc. std": False, "MB / round": False, "ADE": False, "mAP@0.5": True,
}


def page_ablation() -> None:
    header("Ablation — which enhancement does the work",
           "Each variant switches mechanisms on or off; bars show the change against the "
           "study's FedAvg baseline.",
           "Blue bars are better than the baseline, red are worse, gray is unchanged. The "
           "direction accounts for the metric: lower is better for uplink, ADE and spread.")
    studies = data.ablation_studies_available()
    study = st.radio("Study", list(studies), format_func=lambda k: studies[k], horizontal=True)
    table = data.ablation_full_table(study)
    if table.empty:
        st.info("No results for this study.")
        return
    base = table.attrs.get("baseline")
    metric = st.segmented_control("Metric", [m for m in ABLATION_METRICS if table[m].notna().any()],
                                  default="Accuracy (%)") or "Accuracy (%)"
    higher = ABLATION_METRICS[metric]
    left, right = st.columns(2)
    t = table.sort_values(metric, ascending=not higher)
    fig = go.Figure(go.Bar(x=t[metric], y=t["Variant"], orientation="h",
                           marker_color=[NEUTRAL if v == base else SERIES[0] for v in t["Variant"]],
                           text=[f"{v:.3g}" for v in t[metric]], textposition="outside",
                           cliponaxis=False, hovertemplate="%{y}: %{x:.4g}<extra></extra>"))
    fig.update_layout(title=f"{metric} (best first)", showlegend=False,
                      yaxis=dict(autorange="reversed"), hovermode="closest")
    with left:
        show(_style(fig, "", x_title=metric, height=90 + 42 * len(t)),
             f"Gray bar = the baseline: {base}." if base else "")
    if base is not None:
        ref = table.set_index("Variant").loc[base, metric]
        d = table[table["Variant"] != base].assign(delta=lambda x: x[metric] - ref)
        d = d.sort_values("delta", ascending=not higher)
        with right:
            show(delta_bars(d["Variant"].tolist(), d["delta"].tolist(),
                            f"Change vs. {base}", metric, higher))
    num = {c: st.column_config.NumberColumn(format="%.3g") for c in table.columns
           if c != "Variant" and pd.api.types.is_numeric_dtype(table[c])}
    st.dataframe(table, hide_index=True, width="stretch", column_config=num)
    download(table, f"ablation_{study}.csv", f"abl_dl_{study}")


def page_compare() -> None:
    header("Compare runs", "Pick any runs — committed results or your own — and see them side "
           "by side.",
           "Colours follow the order you pick runs in. Up to eight runs; beyond that the lines "
           "become hard to tell apart. **Config differences** lists only the keys that differ.")
    runs = data.discover_runs()
    if len(runs) < 2:
        st.info("Need at least two runs. Launch one from **Launch a run**.")
        return
    labels = {r.name: r for r in runs}
    defaults = [n for n in (os.path.join("results", "benchmark", "proposed_fl"),
                            os.path.join("results", "benchmark", "std_fedavg")) if n in labels]
    picked = st.multiselect("Runs", list(labels), default=defaults, max_selections=len(SERIES))
    if not picked:
        st.warning("Select at least one run.")
        return
    colors = {n: SERIES[i] for i, n in enumerate(picked)}
    df = data.runs_rounds({n: labels[n].path for n in picked})
    if df.empty:
        st.warning("None of the selected runs has per-round telemetry.")
        return
    c1, c2 = st.columns([4, 1])
    avail = data.available_metrics(df)
    metrics = c1.multiselect("Metrics", avail,
                             default=[m for m in ("accuracy", "loss", "mb_cumulative", "client_acc_min")
                                      if m in avail],
                             format_func=lambda m: data.ROUND_METRICS[m][0], key="cmp_metrics")
    window = c2.slider("Smoothing", 1, 10, 1, key="cmp_smooth")
    grid_of([metric_lines(df, "run", m, colors, window) for m in metrics])

    st.subheader("Final metrics")
    rows = []
    for n in picked:
        s = data.load_summary(labels[n].path)
        rows.append({"Run": n, **(data.run_summary_row(s) if s else {})})
    summ = pd.DataFrame(rows)
    st.dataframe(summ, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3g") for c in summ.columns
                                if c != "Run" and pd.api.types.is_numeric_dtype(summ[c])})
    download(summ, "compare_runs.csv", "cmp_dl")

    st.subheader("Config differences")
    only = st.toggle("Only keys that differ", True)
    diff = data.config_diff({n: data.load_config(labels[n].path) for n in picked}, only)
    if diff.empty:
        st.caption("No config.json found, or the configs are identical.")
    else:
        st.dataframe(diff.astype(str), width="stretch")


def page_launch() -> None:
    header("Launch a federated run",
           "Runs the same CLI you would by hand and streams each round from `rounds.jsonl`.",
           "- **quick** = 12 vehicles, small images — about 30–40 s for 8 rounds on CPU.\n"
           "- **default** = the full PRD experiment (50 vehicles) — minutes per round on CPU.\n"
           "- **dp** = default with differential privacy on.\n"
           "Finished runs appear in **Run explorer**, **Compare runs** and **Fleet**.")
    procs: Dict[str, dict] = st.session_state.setdefault("procs", {})

    with st.form("launch"):
        c1, c2, c3 = st.columns(3)
        preset = c1.selectbox("Base config", list(data.CONFIG_FILES), index=0,
                              help="Values below override the chosen YAML file.")
        name = c2.text_input("Run name", value=f"dash_{time.strftime('%H%M%S')}",
                             help="Folder under runs/. Letters, digits, - and _ only.")
        rounds = c3.number_input("Rounds (R_max)", 1, 200, 8)
        c1, c2, c3 = st.columns(3)
        clients = c1.number_input("Vehicles (N)", 2, 500, 12)
        fraction = c2.slider("Participation fraction (C)", 0.05, 1.0, 0.3, 0.05,
                             help="Share of the fleet selected each round.")
        epochs = c3.number_input("Local epochs (E)", 1, 20, 2)
        c1, c2, c3 = st.columns(3)
        quality = c1.toggle("Quality-aware aggregation", True,
                            help="Weight each update by data similarity × sensor quality.")
        traj = c1.toggle("Trajectory-aware selection", True,
                         help="Prefer vehicles with stable V2I links, stratified by region.")
        dp_on = c2.toggle("Differential privacy", preset == "dp")
        epsilon = c2.number_input("ε budget", 0.5, 64.0, 8.0,
                                  help="Used only when differential privacy is on. Lower = more private.")
        topk = c3.slider("Top-k sparsity ratio", 0.02, 1.0, 0.25, 0.01,
                         help="Fraction of coordinates each vehicle sends. 1.0 = dense updates.")
        byz = c3.toggle("Byzantine-robust (trimmed mean)", False,
                        help="Disables secure aggregation: the trimmed mean needs individual updates.")
        est = 5 * int(rounds) * max(1, int(clients) * float(fraction) * int(epochs)) / 7
        st.caption(f"Rough CPU estimate for the quick preset: ~{est:.0f} s.")
        submitted = st.form_submit_button("▶ Start run", type="primary")

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
        st.toast(f"Started {safe}", icon="🚀")

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
        if st.button("■ Stop run"):
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
        show(single_line(df, "loss", "Global test loss", "Loss", color="#6250d6"))
        show(single_line(df, "mb_round", "Uplink per round", "MB", color="#eda100"))
        show(single_line(df, "delta_norm", "Update norm ‖Δw‖", "Norm", color="#e87ba4"))


def summary_metrics(summary: dict) -> None:
    row = data.run_summary_row(summary)
    c = st.columns(6)
    fmt = [("Accuracy (%)", "{:.1f}%"), ("Worst client (%)", "{:.1f}%"), ("ADE", "{:.3f}"),
           ("MB / round", "{:.2f}"), ("Rounds", "{:d}"), ("Wall clock (min)", "{:.1f}")]
    for col, (k, f) in zip(c, fmt):
        v = row.get(k)
        col.metric(k.replace(" (%)", ""), "—" if v is None else f.format(int(v) if f == "{:d}" else v))


def page_explorer() -> None:
    header("Run explorer", "Any run under runs/ (🟢 yours) or results/ (📦 committed).")
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
    summary = data.load_summary(run.path)
    if summary:
        summary_metrics(summary)
    tabs = st.tabs(["📈 Curves", "🚗 Participation", "📋 Per-round table", "🧾 Summary", "⚙️ Config"])
    with tabs[0]:
        avail = data.available_metrics(df)
        metrics = st.multiselect("Metrics", avail,
                                 default=[m for m in ("accuracy", "loss", "mb_round", "client_acc_std")
                                          if m in avail],
                                 format_func=lambda m: data.ROUND_METRICS[m][0], key="exp_metrics")
        grid_of([metric_lines(df.assign(run="run"), "run", m, {"run": SERIES[0]}) for m in metrics])
    with tabs[1]:
        vr = data.load_vehicle_rounds(run.path)
        status = data.status_per_round(vr)
        if status.empty:
            st.info("This run has no per-vehicle telemetry.")
        else:
            fig = go.Figure()
            for s in status.columns:
                if s == "not_selected":
                    continue
                color, label = STATUS_STYLE[s]
                fig.add_trace(go.Bar(x=status.index, y=status[s], name=label, marker_color=color,
                                     hovertemplate="round %{x}: %{y} " + label + "<extra></extra>"))
            fig.update_layout(title="Selected vehicles per round, by outcome", barmode="stack",
                              bargap=0.15)
            show(_style(fig, "Vehicles"))
            participation_heatmap(vr)
    with tabs[2]:
        st.dataframe(df, hide_index=True, width="stretch")
        download(df, f"{os.path.basename(run.path)}_rounds.csv", "exp_dl")
    with tabs[3]:
        if summary:
            flat = pd.DataFrame({"value": data.flatten(summary)}).astype(str)
            st.dataframe(flat, width="stretch")
        else:
            st.info("No summary.json (run unfinished?).")
    with tabs[4]:
        cfg = data.load_config(run.path)
        if cfg:
            st.dataframe(pd.DataFrame({"value": data.flatten(cfg)}).astype(str), width="stretch")
        else:
            st.info("No config.json.")


def page_validation() -> None:
    header("Mathematical validation",
           "Each figure calls the shipped code and checks it against a closed form or a "
           "theoretical bound. Generated by `scripts/make_proof_figures.py`.")
    sections = data.validation_sections()
    if not sections:
        st.info("No figures in docs/figures/.")
        return
    mode = st.segmented_control("View", ["Gallery", "One at a time"], default="Gallery")
    if mode == "One at a time":
        titles = [f"{s['number']}. {s['title']}" for s in sections]
        i = titles.index(st.selectbox("Figure", titles))
        s = sections[i]
        st.image(s["image"], width="stretch")
        if s["text"]:
            st.markdown(s["text"])
        return
    cols = st.columns(2)
    for i, s in enumerate(sections):
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(f"**{s['number']}. {s['title']}**")
                st.image(s["image"], width="stretch")
                if s["text"]:
                    with st.expander("What this shows"):
                        st.markdown(s["text"])


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


PAGES = {   # name -> (icon, page function, one-line description shown in the sidebar)
    "Overview": ("🏠", page_overview, "Headline results & acceptance"),
    "Benchmark": ("📊", page_benchmark, "Four methods, round by round"),
    "Fleet (every vehicle)": ("🚗", page_fleet, "Every car's profile & accuracy"),
    "Ablation": ("🧪", page_ablation, "Which enhancement helps"),
    "Compare runs": ("🔀", page_compare, "Any runs side by side"),
    "Run explorer": ("🗂", page_explorer, "One run in detail"),
    "Launch a run": ("🚀", page_launch, "Train and watch live"),
    "Validation": ("📐", page_validation, "Maths checked against code"),
}


def sidebar_status() -> None:
    report = data.load_benchmark()
    runs = data.discover_runs()
    live = sum(r.source == "live" for r in runs)
    if report:
        cfg = report.get("config", {})
        st.caption(f"📦 Benchmark: N={cfg.get('num_clients')}, "
                   f"{cfg.get('train', {}).get('rounds')} rounds")
    else:
        st.caption("📦 No committed benchmark found")
    st.caption(f"🗂 {len(runs)} runs found · {live} of yours in runs/")
    running = sum(1 for j in st.session_state.get("procs", {}).values() if j["proc"].poll() is None)
    if running:
        st.caption(f"🟢 {running} run(s) in progress — see **Launch a run**")


with st.sidebar:
    st.markdown("## 🚗 FL-AV")
    st.caption("Federated learning for connected & autonomous vehicles")
    page = st.radio("Page", list(PAGES), label_visibility="collapsed",
                    format_func=lambda p: f"{PAGES[p][0]}  {p}",
                    captions=[PAGES[p][2] for p in PAGES])
    st.divider()
    sidebar_status()
    if st.button("↻ Reload results", width="stretch"):
        st.cache_data.clear()
        st.toast("Results reloaded", icon="✅")
    st.caption("Synthetic data, simulated V2I. Relative ordering of arms is what's measured.")

PAGES[page][1]()
