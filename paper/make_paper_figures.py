#!/usr/bin/env python3
"""Regenerate every figure in paper/ from the committed evidence in results/.

    python paper/make_paper_figures.py

Nothing is typed by hand: each number plotted is read from results/**/summary.json,
rounds.csv or the *.json attribution files. Figure 1 is a schematic and plots no data.
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper", "figures")
os.makedirs(OUT, exist_ok=True)

# Okabe-Ito colour-blind-safe palette.
C = {
    "central": "#000000",
    "local": "#999999",
    "fedavg": "#0072B2",
    "proposed": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "yellow": "#E69F00",
    "sky": "#56B4E9",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.png"))
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))
    plt.close(fig)
    print("wrote", name)


def rounds(arm):
    return pd.read_csv(os.path.join(RES, "benchmark", arm, "rounds.csv"))


def summary(arm):
    with open(os.path.join(RES, "benchmark", arm, "summary.json")) as fh:
        return json.load(fh)


def smooth(y, k=3):
    y = np.asarray(y, float)
    pad = np.pad(y, (k // 2, k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


# --------------------------------------------------------------------------- Fig 1
def fig_architecture():
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 66)
    ax.axis("off")

    def panel(x, w, ec, fc, title):
        ax.add_patch(FancyBboxPatch((x, 2), w, 62, boxstyle="round,pad=0.3,rounding_size=1.5",
                                    fc=fc, ec=ec, lw=1.6))
        ax.text(x + w / 2, 60.2, title, ha="center", va="center", fontsize=9,
                fontweight="bold", color=ec)

    def box(x, y, w, h, head, body, fc="#FFFFFF"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.0",
                                    fc=fc, ec="#333333", lw=0.9))
        ax.text(x + w / 2, y + h - 2.4, head, ha="center", va="center", fontsize=8.2, fontweight="bold")
        ax.text(x + w / 2, y + (h - 4.2) / 2, body, ha="center", va="center", fontsize=7.0,
                linespacing=1.35)

    def arrow(x0, y0, x1, y1, color="#555555", lw=1.2):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10,
                                     lw=lw, color=color))

    panel(2, 47, C["proposed"], "#FFF4E8", "Road-side unit (RSU)")
    ax.text(94.5, 56.6, "raw sensor data never leaves the vehicle", ha="center", va="center", fontsize=6.8, style="italic", color=C["fedavg"])
    panel(71, 47, C["fedavg"], "#EAF4FB", "Vehicle k in S(r)")

    box(5, 40.5, 41, 14.5, "(1) Select vehicles",
        "stable V2I link, region-stratified,\nfairness guard so no vehicle starves")
    box(5, 21.5, 41, 16.5, "(4) Aggregate",
        "quality-weighted mean of deltas,\nFedNova step normalisation,\nserver momentum; sees only the sum\nunder secure aggregation")
    box(5, 5, 41, 14, "Global model  w(r)", "700,485 parameters\nmulticast once per round")

    box(74, 40.5, 41, 14.5, "(2) Train locally", "E epochs of SGD on the vehicle's\nprivate camera / LiDAR shard")
    box(74, 21.5, 41, 16.5, "(3) Build the update",
        "quality score q_k computed on-vehicle,\ntop-k + error feedback, fp16 + gzip,\nsecure-aggregation mask (+ optional DP)")
    box(74, 5, 41, 14, "Private shard  D_k", "non-IID: label, feature and\nquantity skew", fc="#F2F2F2")

    arrow(46.8, 48, 73.2, 48, color=C["fedavg"], lw=1.6)
    ax.text(60, 51.2, "broadcast w(r)", ha="center", fontsize=7.5, color=C["fedavg"])
    arrow(73.2, 30, 46.8, 30, color=C["proposed"], lw=1.6)
    ax.text(60, 33.0, "masked sparse delta", ha="center", fontsize=6.8, color=C["proposed"])
    ax.text(60, 26.4, "+ (n_k, q_k)", ha="center", fontsize=6.8, color=C["proposed"])
    arrow(25.5, 40.3, 25.5, 38.2)
    arrow(25.5, 21.3, 25.5, 19.2)
    arrow(94.5, 40.3, 94.5, 38.2)
    arrow(94.5, 19.2, 94.5, 21.3)
    save(fig, "fig1_architecture")


# --------------------------------------------------------------------------- Fig 2
def fig_learning_curves():
    cen, fed, prop, loc = (rounds(a) for a in ("centralized", "std_fedavg", "proposed_fl", "local_only"))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(cen["round"], cen["accuracy"] * 100, color=C["central"], lw=1.4, ls="--",
            label="Centralized (upper bound, pools raw data)")
    ax.plot(fed["round"], fed["accuracy"] * 100, color=C["fedavg"], lw=1.0, alpha=0.35)
    ax.plot(prop["round"], prop["accuracy"] * 100, color=C["proposed"], lw=1.0, alpha=0.35)
    ax.plot(fed["round"], smooth(fed["accuracy"] * 100), color=C["fedavg"], lw=2.0,
            label="Standard FedAvg")
    ax.plot(prop["round"], smooth(prop["accuracy"] * 100), color=C["proposed"], lw=2.0,
            label="Proposed")
    loc_mean = loc["accuracy"].iloc[-10:].mean() * 100
    ax.axhline(loc_mean, color=C["local"], lw=1.4, ls=":", label=f"Local-only (last-10 mean, {loc_mean:.1f}%)")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Accuracy on IID global test set (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(1, 60)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=8)
    save(fig, "fig2_learning_curves")


# --------------------------------------------------------------------------- Fig 3
def fig_comm_efficiency():
    fed, prop = rounds("std_fedavg"), rounds("proposed_fl")
    cen = summary("centralized")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    ax.plot(fed["mb_cumulative"], smooth(fed["accuracy"] * 100), color=C["fedavg"], lw=2, label="Standard FedAvg")
    ax.plot(prop["mb_cumulative"], smooth(prop["accuracy"] * 100), color=C["proposed"], lw=2, label="Proposed")
    ax.axhline(cen["accuracy"] * 100, color=C["central"], ls="--", lw=1.0)
    ax.text(fed["mb_cumulative"].max() * 0.99, cen["accuracy"] * 100 + 1.0, "centralized ceiling",
            ha="right", fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Cumulative uplink + downlink (MB, log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    # annotate the budget at which the proposed run has finished
    x_end = prop["mb_cumulative"].iloc[-1]
    ax.axvline(x_end, color=C["proposed"], lw=0.8, ls=":")
    acc_fed_at = float(np.interp(x_end, fed["mb_cumulative"], smooth(fed["accuracy"] * 100)))
    ax.annotate(f"at {x_end:.0f} MB:\nFedAvg {acc_fed_at:.0f}%,\nproposed {smooth(prop['accuracy']*100)[-1]:.0f}%",
                xy=(x_end, acc_fed_at), xytext=(7.5, 62), fontsize=7.5,
                arrowprops=dict(arrowstyle="-", color="#555555", lw=0.7))
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("(a) Accuracy per megabyte", loc="left")

    ax = axes[1]
    tot = [summary("std_fedavg")["total_mb"], summary("proposed_fl")["total_mb"]]
    bars = ax.bar(["Standard\nFedAvg", "Proposed"], tot, color=[C["fedavg"], C["proposed"]], width=0.55)
    for b, v in zip(bars, tot):
        ax.text(b.get_x() + b.get_width() / 2, v + 60, f"{v:,.0f} MB", ha="center", fontsize=8)
    red = 1 - tot[1] / tot[0]
    ax.annotate(f"-{red*100:.1f}%", xy=(1, tot[1] + 430), xytext=(1, tot[0] * 0.5), ha="center",
                fontsize=13, fontweight="bold", color=C["proposed"],
                arrowprops=dict(arrowstyle="->", color=C["proposed"], lw=1.4))
    ax.set_ylabel("Total traffic over 60 rounds (MB)")
    ax.set_ylim(0, max(tot) * 1.15)
    ax.set_title("(b) Total communication", loc="left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig3_communication")


# --------------------------------------------------------------------------- Fig 4
def fig_ablation():
    def load(name):
        with open(os.path.join(RES, name)) as fh:
            return json.load(fh)

    att, ing, fix = load("attribution/attribution.json"), load("ingredients/ingredients.json"), load("fix/fix.json")
    ctrl = att["fedavg_control"]
    rows = [
        ("Connectivity / trajectory\nselection", ing["selection_only"], C["green"]),
        ("Server momentum", ing["momentum_only"], C["green"]),
        ("Quality-aware weighting", ing["quality_only"], C["green"]),
        ("Adaptive local epochs\n(plain averaging)", ing["adaptive_epochs_only"], C["proposed"]),
        ("Adaptive epochs +\nnormalised averaging", fix["adaptive_epochs_normalised"], C["yellow"]),
        ("All algorithmic\n(normalised)", fix["all_algorithmic_normalised"], C["yellow"]),
        ("Compression + secure agg.\n(vs same arm without)", att["algorithmic_plus_compression"], C["sky"]),
    ]
    refs = [ctrl] * (len(rows) - 1) + [att["algorithmic_only"]]
    # the last row is relative to the *pre-fix* algorithmic arm for a clean attribution of transport cost
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9), sharey=True, gridspec_kw={"width_ratios": [1.25, 1]})
    y = np.arange(len(rows))[::-1]
    d_acc = [(r[1]["accuracy"] - ref["accuracy"]) * 100 for r, ref in zip(rows, refs)]
    d_std = [(r[1]["robustness"]["client_acc_std"] / ref["robustness"]["client_acc_std"] - 1) * 100
             for r, ref in zip(rows, refs)]
    ax = axes[0]
    ax.barh(y, d_acc, color=[r[2] for r in rows], height=0.62)
    ax.axvline(0, color="#333333", lw=0.8)
    for yi, v in zip(y, d_acc):
        ax.text(v + (0.5 if v >= 0 else -0.5), yi, f"{v:+.1f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    ax.set_xlabel("Accuracy vs FedAvg control (points)")
    ax.set_xlim(-16, 13)
    ax.set_title("(a) Mean accuracy", loc="left")
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    ax.barh(y, d_std, color=[r[2] for r in rows], height=0.62)
    ax.axvline(0, color="#333333", lw=0.8)
    for yi, v in zip(y, d_std):
        ax.text(v + (1.5 if v >= 0 else -1.5), yi, f"{v:+.0f}%", va="center",
                ha="left" if v >= 0 else "right", fontsize=7.5)
    ax.set_xlabel("Cross-client std vs reference (%, lower is better)")
    ax.set_xlim(-50, 45)
    ax.set_title("(b) Fleet dispersion", loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig4_ablation")


# --------------------------------------------------------------------------- Fig 5
def fig_tail():
    arms = [("std_fedavg", "Standard\nFedAvg", C["fedavg"]), ("proposed_fl", "Proposed", C["proposed"])]
    with open(os.path.join(RES, "benchmark", "benchmark_report.json")) as fh:
        rep = json.load(fh)
    s = {a: summary(a) for a, _, _ in arms}
    metrics = [("Mean client accuracy", lambda x: x["robustness"]["client_acc_mean"] * 100, "%"),
               ("Cross-client std", lambda x: x["robustness"]["client_acc_std"] * 100, "pts"),
               ("Worst client", lambda x: x["robustness"]["client_acc_min"] * 100, "%")]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9))
    for ax, (title, fn, unit) in zip(axes, metrics):
        vals = [fn(s[a]) for a, _, _ in arms]
        b = ax.bar([n for _, n, _ in arms], vals, color=[c for _, _, c in arms], width=0.55)
        for bi, v in zip(b, vals):
            ax.text(bi.get_x() + bi.get_width() / 2, v + max(vals) * 0.02, f"{v:.1f}", ha="center", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, max(vals) * 1.2)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel(unit)
    fig.tight_layout()
    save(fig, "fig5_tail")


if __name__ == "__main__":
    fig_architecture()
    fig_learning_curves()
    fig_comm_efficiency()
    fig_ablation()
    fig_tail()
