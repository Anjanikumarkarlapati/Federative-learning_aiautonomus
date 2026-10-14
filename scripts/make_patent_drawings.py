"""Patent drawings for the Indian application (Rule 15 format), black and white.

    python scripts/make_patent_drawings.py [--applicant "NAME OF APPLICANT"]

Writes docs/patent/drawings/:
    drawings.pdf          every sheet, A4, numbered "n/N"
    FIGnn.png             preview of each sheet (for review only, not for filing)
and docs/patent/REFERENCE_NUMERALS.md from the same numeral table.

Rule 15 layout: A4 portrait, margins >= 4 cm top and left, >= 3 cm bottom and
right, no dimensions, sheets numbered consecutively, applicant name at the top
left. Text appears only as short labels and in flow diagrams; every element
carries the reference numeral used in the specification.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "patent", "drawings")

CM = 1 / 2.54
A4 = (21.0 * CM, 29.7 * CM)
LEFT, TOP, RIGHT, BOTTOM = 4.0, 4.0, 3.0, 3.0      # cm
W, H = 100.0, 100.0 * (29.7 - TOP - BOTTOM) / (21.0 - LEFT - RIGHT)   # drawing units

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                     "lines.color": "black", "text.color": "black"})

NUMERALS = {
    # system
    100: "vehicle", 102: "further vehicles of the fleet", 110: "on-board unit (OBU)",
    120: "vehicle sensors (camera, LiDAR, radar)", 130: "V2I radio link",
    200: "road-side unit (RSU)", 210: "coverage cell of the RSU", 300: "backhaul network",
    310: "model repository",
    # OBU
    111: "sensor interface", 112: "sensor-health / calibration monitor",
    113: "processor and neural accelerator", 114: "memory",
    115: "error-feedback residual buffer", 116: "local training data store",
    117: "secure element holding the private key", 118: "V2X modem (PC5 / Uu)", 119: "antenna",
    # RSU
    201: "V2I transceiver", 202: "link-rate estimator", 203: "vehicle selector",
    204: "tier scheduler", 205: "secure aggregation unit", 206: "global model store",
    207: "backhaul interface", 208: "processor", 209: "memory",
    # packet
    700: "masked update message", 701: "magic / protocol identifier", 702: "vehicle identifier",
    703: "round identifier", 704: "assigned tier", 705: "value count", 706: "CRC-32 checksum",
    707: "masked weighted update values on the shared support prefix",
    708: "masked per-segment weight sums and weight-step products",
    # support
    800: "ordered shared support", 801: "segment of the support", 802: "tier prefix lengths",
}


# ------------------------------------------------------------------ helpers
def sheet(applicant: str):
    fig = plt.figure(figsize=A4)
    ax = fig.add_axes([LEFT / 21.0, BOTTOM / 29.7, (21.0 - LEFT - RIGHT) / 21.0,
                       (29.7 - TOP - BOTTOM) / 29.7])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fig.text(LEFT / 21.0, 1 - 1.8 / 29.7, applicant, ha="left", va="center", fontsize=9)
    return fig, ax


def box(ax, x, y, w, h, text="", num=None, round_=False, lw=1.2, fs=8.5, dashed=False):
    style = "round,pad=0.0,rounding_size=1.5" if round_ else "square,pad=0.0"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fill=False, lw=lw,
                                ec="black", ls="--" if dashed else "-"))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)
    if num is not None:
        tag(ax, x + w, y + h, num)


def tag(ax, x, y, num, dx=3.0, dy=2.0):
    """Reference numeral with a short leader line."""
    ax.plot([x, x + dx], [y, y + dy], color="black", lw=0.7)
    ax.text(x + dx + 0.4, y + dy, str(num), fontsize=8.5, ha="left", va="center")


def arrow(ax, p, q, both=False, lw=1.0, dashed=False):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="<|-|>" if both else "-|>", mutation_scale=9,
                                 lw=lw, color="black", ls="--" if dashed else "-"))


def caption(ax, text):
    ax.text(W / 2, 2, text, ha="center", va="bottom", fontsize=11, fontweight="bold")


def flow(ax, steps, x=12, w=76, top=None, gap=3.2, h=7.5):
    """Vertical flow diagram; steps = [(label, text, kind)] kind in {box, decision}."""
    y = (top if top is not None else H - 6) - h
    centers = []
    for label, text, kind in steps:
        if kind == "decision":
            cx, cy = x + w / 2, y + h / 2
            ax.add_patch(plt.Polygon([(cx, y + h + 1), (x + w * 0.85, cy), (cx, y - 1),
                                      (x + w * 0.15, cy)], fill=False, lw=1.2, ec="black"))
            ax.text(cx, cy, text, ha="center", va="center", fontsize=8)
        else:
            box(ax, x, y, w, h, text, fs=8)
        ax.text(x - 1.2, y + h / 2, label, ha="right", va="center", fontsize=8.5)
        centers.append((x + w / 2, y, y + h))
        y -= h + gap
    for (cx, y0, _), (_, _, y1n) in zip(centers, centers[1:]):
        arrow(ax, (cx, y0 - (1 if False else 0)), (cx, y1n))
    return centers


# ------------------------------------------------------------------- sheets
def fig_system(ax):
    box(ax, 30, 92, 40, 12, "Road-side unit", 200)
    ax.add_patch(matplotlib.patches.Ellipse((50, 57), 96, 62, fill=False, lw=0.9, ls="--"))
    tag(ax, 88, 82, 210)
    box(ax, 30, 116, 40, 9, "Model repository", 310)
    arrow(ax, (50, 104), (50, 116), both=True)
    tag(ax, 52, 109, 300, dx=4, dy=0)
    xs = [(8, 40), (38, 30), (66, 42)]
    for i, (x, y) in enumerate(xs):
        box(ax, x, y, 26, 14, "", 100 if i == 0 else None, round_=True)
        box(ax, x + 3, y + 3, 11, 8, "OBU", 110 if i == 0 else None, fs=7.5)
        ax.add_patch(Rectangle((x + 17, y + 4), 6, 6, fill=False, lw=0.9))
        if i == 0:
            tag(ax, x + 23, y + 4, 120, dx=4, dy=-4)
        arrow(ax, (x + 13, y + 14), (44 + 6 * i, 92), both=True, dashed=True)
    tag(ax, 27, 70, 130, dx=-6, dy=3)
    tag(ax, 92, 40, 102, dx=2, dy=6)
    ax.text(50, 18, "only masked model updates cross link 130;\nraw sensor data stays in 110",
            ha="center", fontsize=8)


def fig_obu(ax):
    box(ax, 6, 20, 88, 110, "", 110)
    box(ax, 12, 104, 30, 14, "Sensors", 120)
    box(ax, 52, 104, 36, 14, "Sensor interface", 111)
    box(ax, 52, 82, 36, 14, "Sensor-health /\ncalibration monitor", 112)
    box(ax, 12, 60, 76, 14, "Processor + neural accelerator", 113)
    box(ax, 12, 32, 46, 20, "", 114)
    ax.text(14, 49, "Memory", fontsize=8, va="center")
    box(ax, 15, 35, 19, 10, "Residual\nbuffer", 115, fs=7.5)
    box(ax, 36, 35, 19, 10, "Training\ndata", 116, fs=7.5)
    box(ax, 64, 42, 24, 10, "Secure\nelement", 117, fs=7.5)
    box(ax, 64, 26, 24, 10, "V2X modem", 118, fs=7.5)
    ax.plot([88, 91, 91], [31, 31, 136], color="black", lw=1)
    ax.plot([87, 91, 95], [140, 136, 140], color="black", lw=1)
    tag(ax, 95, 140, 119, dx=2, dy=-1)
    arrow(ax, (42, 111), (52, 111))
    arrow(ax, (70, 104), (70, 96))
    arrow(ax, (70, 82), (70, 74))
    arrow(ax, (35, 60), (35, 52), both=True)
    arrow(ax, (76, 60), (76, 52), both=True)
    arrow(ax, (76, 42), (76, 36), both=True)


def fig_rsu(ax):
    box(ax, 6, 22, 88, 110, "", 200)
    box(ax, 12, 112, 34, 12, "V2I transceiver", 201)
    box(ax, 54, 112, 34, 12, "Backhaul interface", 207)
    box(ax, 12, 92, 34, 12, "Link-rate estimator", 202)
    box(ax, 54, 92, 34, 12, "Vehicle selector", 203)
    box(ax, 12, 72, 34, 12, "Tier scheduler", 204)
    box(ax, 54, 72, 34, 12, "Secure aggregation\nunit", 205)
    box(ax, 12, 50, 76, 12, "Processor", 208)
    box(ax, 12, 30, 76, 12, "Memory / global model store", 209)
    tag(ax, 88, 36, 206, dx=3, dy=6)
    for a, b in [((29, 112), (29, 104)), ((29, 92), (29, 84)), ((46, 98), (54, 98)),
                 ((46, 78), (54, 78)), ((71, 72), (71, 62)), ((71, 112), (71, 104)),
                 ((50, 50), (50, 42))]:
        arrow(ax, a, b, both=True)


VEHICLE_STEPS = [
    ("S401", "Receive global model delta, tier and peer public keys", "box"),
    ("S402", "Train locally on private sensor data (tau_k steps)", "box"),
    ("S403", "Compute quality score q_k from sensor health,\ncalibration and label similarity", "box"),
    ("S404", "u_k = dW_k / tau_k + residual;  w_k = q_k n_k / n_ref", "box"),
    ("S405", "Take prefix of the shared support for the assigned tier", "box"),
    ("S406", "Encode [w_k u_k , w_k , w_k tau_k] as fixed point in Z_2^32,\nclip excess to residual", "box"),
    ("S407", "Add pairwise masks from Diffie-Hellman seeds\nover the coordinates shared with each peer", "box"),
    ("S408", "Transmit masked message 700 with CRC", "box"),
    ("S409", "ACK received?", "decision"),
    ("S410", "Yes: residual keeps only the unsent part\nNo (NACK / timeout): residual keeps the whole update", "box"),
]

RSU_STEPS = [
    ("S501", "Select vehicles (link uptime, regional coverage,\nstarvation guard)", "box"),
    ("S502", "Relay public keys; predict each vehicle's link rate", "box"),
    ("S503", "Assign tier: longest support prefix that fits\nan equal share of the airtime budget", "box"),
    ("S504", "Broadcast global delta, tiers and value bound", "box"),
    ("S505", "Receive masked messages; check CRC", "box"),
    ("S506", "Any selected vehicle missing?", "decision"),
    ("S507", "Survivors reveal pair seeds with missing vehicles;\nremove those mask terms", "box"),
    ("S508", "Sum ring vectors: S = sum w u, W_t = sum w, T_t = sum w tau", "box"),
    ("S509", "d_t = (T_t / W_t) * S_t / W_t per segment;\nscatter onto the shared support", "box"),
    ("S510", "Server momentum; w <- w + v; derive next support", "box"),
]


def fig_vehicle_flow(ax):
    flow(ax, VEHICLE_STEPS, top=H - 2, h=9.2, gap=3.0)


def fig_rsu_flow(ax):
    flow(ax, RSU_STEPS, top=H - 2, h=9.2, gap=3.0)


def fig_sequence(ax):
    cols = [(15, "Vehicle i\n(110)"), (40, "Vehicle j\n(110)"), (65, "Vehicle m\n(110)"), (88, "RSU\n(200)")]
    for x, name in cols:
        ax.text(x, H - 4, name, ha="center", va="top", fontsize=8.5)
        ax.plot([x, x], [10, H - 12], color="black", lw=0.8, ls="--")
    rows = [
        (88, [15, 40, 65], "global delta, tier, keys (S504)"),
        (None, None, "local training (S402-S407)"),
        (15, [88], "masked msg 700"),
        (40, [88], "masked msg 700"),
        (65, [88], "lost / CRC fail"),
        (88, [15, 40], "request seeds for m"),
        (15, [88], "seed(i,m)"),
        (40, [88], "seed(j,m)"),
        (88, [65], "NACK: m keeps whole update"),
    ]
    y = H - 18
    for src, dsts, text in rows:
        if src is None:
            for x, _ in cols[:3]:
                box(ax, x - 6, y - 3, 12, 6, "", fs=7)
            ax.text(52, y + 4.5, text, ha="center", fontsize=8)
        else:
            for d in dsts:
                lost = "lost" in text
                arrow(ax, (src, y), (d if not lost else (src + d) / 2, y), dashed=lost)
            mid = (src + dsts[-1]) / 2 if len(dsts) == 1 else (src + min(dsts)) / 2
            ax.text(mid, y + 2.0, text, ha="center", fontsize=7.5,
                    bbox=dict(fc="white", ec="none", pad=0.6))
            if "lost" in text:
                ax.text((src + dsts[0]) / 2 + 2, y - 0.6, "X", fontsize=12, va="center")
        y -= 12


def fig_packet(ax):
    fields = [(701, "magic\n4 B"), (702, "vehicle id\n4 B"), (703, "round\n4 B"),
              (704, "tier\n4 B"), (705, "count\n4 B"), (706, "CRC-32\n4 B")]
    x = 5
    for num, text in fields:
        box(ax, x, 100, 15, 14, text, None, fs=7.5)
        tag(ax, x + 7.5, 114, num, dx=0.5, dy=4)
        x += 15
    box(ax, 5, 76, 66, 14, "masked values  w_k u_k  on support prefix\n4 B each, length L_tier", None, fs=8)
    tag(ax, 71, 90, 707)
    box(ax, 71, 76, 24, 14, "masked\n[W, W*tau]\nper segment", None, fs=7.5)
    tag(ax, 95, 90, 708, dx=0.5, dy=4)
    box(ax, 3, 72, 94, 46, "", 700, dashed=True)
    ax.text(50, 60, "No coordinate indices are sent: every vehicle derives the same\n"
            "ordered support 800 from the previous global update.\n"
            "q_k, n_k and tau_k appear only inside masked sums.", ha="center", fontsize=8)


def fig_support(ax):
    y0 = 100
    lengths = [10, 18, 32, 56, 90]
    ax.add_patch(Rectangle((5, y0), 90, 8, fill=False, lw=1.2))
    for i, L in enumerate(lengths):
        ax.plot([5 + L, 5 + L], [y0 - 2, y0 + 10], color="black", lw=0.8)
        ax.text(5 + L, y0 + 11.5, f"L{i + 1}", ha="center", fontsize=8)
    tag(ax, 95, y0 + 8, 800)
    tag(ax, 5 + 25, y0, 801, dx=2, dy=-5)
    ax.text(50, y0 + 18, "coordinates ordered by |previous global update|  ->", ha="center", fontsize=8)
    tiers = [("vehicle on slow link", 1), ("vehicle on medium link", 3), ("vehicle on fast link", 5)]
    y = 76
    for text, t in tiers:
        ax.add_patch(Rectangle((5, y), lengths[t - 1], 6, fill=True, fc="white", hatch="///", lw=1))
        ax.text(5, y + 8, f"{text}: tier {t}", va="center", fontsize=8)
        y -= 12
    tag(ax, 5 + lengths[0], 82, 802, dx=6, dy=0)
    ax.text(50, 30, "Segment t is normalised by the weights of exactly the vehicles\n"
            "whose prefix covers it; a pair of vehicles masks only the\n"
            "coordinates both of them send, so all masks cancel.", ha="center", fontsize=8)


def fig_results(ax, effects):
    """Line-style (not colour) bars of the multi-seed means."""
    agg = effects["vs_std_fedavg"]["arms"]
    arms = [a for a in ("std_fedavg", "fedavg_transport", "proposed_secure", "proposed_tiered") if a in agg]
    names = {"std_fedavg": "A", "fedavg_transport": "B", "proposed_secure": "C", "proposed_tiered": "D"}
    panels = [("mb_per_round", "traffic per round (MB)"),
              ("uplink_airtime_s_per_round", "uplink airtime per round (s)"),
              ("best_accuracy", "best accuracy (fraction)")]
    hatches = ["", "...", "///", "xxx"]
    for p, (metric, title) in enumerate(panels):
        top = H - 8 - p * 42
        vals = [agg[a][metric]["mean"] or 0 for a in arms]
        cis = [agg[a][metric]["ci95"] or 0 for a in arms]
        vmax = max(v + c for v, c in zip(vals, cis)) or 1
        ax.text(5, top + 2, title, fontsize=8.5)
        for i, (a, v, c) in enumerate(zip(arms, vals, cis)):
            x = 15 + i * 20
            hgt = 28 * v / vmax
            ax.add_patch(Rectangle((x, top - 32), 12, hgt, fill=False, hatch=hatches[i], lw=1))
            ax.plot([x + 6, x + 6], [top - 32 + 28 * (v - c) / vmax, top - 32 + 28 * (v + c) / vmax],
                    color="black", lw=0.9)
            ax.text(x + 6, top - 35, names[a], ha="center", fontsize=8.5)
            ax.text(x + 6, top - 32 + hgt + 1.5, f"{v:.3g}", ha="center", fontsize=7.5)
        ax.plot([12, 95], [top - 32, top - 32], color="black", lw=0.8)
    ax.text(5, 12, "A: standard FedAvg   B: FedAvg on the same compression transport\n"
            "C: quality-hiding secure aggregation   D: C with airtime-budget tiers\n"
            "bars: mean over seeds; vertical line: 95% confidence interval",
            fontsize=8)


SHEETS = [
    ("FIG. 1", fig_system), ("FIG. 2", fig_obu), ("FIG. 3", fig_rsu),
    ("FIG. 4", fig_vehicle_flow), ("FIG. 5", fig_rsu_flow), ("FIG. 6", fig_sequence),
    ("FIG. 7", fig_packet), ("FIG. 8", fig_support),
]


def write_numerals() -> None:
    path = os.path.join(ROOT, "docs", "patent", "REFERENCE_NUMERALS.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Reference numerals\n\nGenerated by `scripts/make_patent_drawings.py`; "
                 "use exactly these numerals in the specification.\n\n| Numeral | Element |\n|---|---|\n")
        for n in sorted(NUMERALS):
            fh.write(f"| {n} | {NUMERALS[n]} |\n")
        fh.write("\nFlow-diagram steps: S401–S410 (vehicle, FIG. 4), S501–S510 (RSU, FIG. 5).\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--applicant", default="Applicant: [NAME OF APPLICANT]")
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    sheets = list(SHEETS)
    eff_path = os.path.join(ROOT, "results", "patent", "effects.json")
    if os.path.exists(eff_path):
        with open(eff_path) as fh:
            effects = json.load(fh)
        sheets.append(("FIG. 9", lambda ax: fig_results(ax, effects)))
    total = len(sheets)
    with PdfPages(os.path.join(OUT, "drawings.pdf")) as pdf:
        for i, (label, draw) in enumerate(sheets, start=1):
            fig, ax = sheet(args.applicant)
            fig.text(1 - RIGHT / 21.0, 1 - 1.8 / 29.7, f"{i}/{total}", ha="right", va="center",
                     fontsize=9)
            draw(ax)
            caption(ax, label)
            pdf.savefig(fig)
            fig.savefig(os.path.join(OUT, f"FIG{i:02d}.png"), dpi=80)
            plt.close(fig)
    write_numerals()
    print(f"wrote {total} sheets to {OUT}")


if __name__ == "__main__":
    main()
