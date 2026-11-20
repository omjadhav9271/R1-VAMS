"""Paper B figures, from paperB_final_complete.json (no simulation; run paperB_final.py --complete first).

Figure B1: gain of each uncertainty-directed rule over percentile sortition tuned to its best
P, as a share of that cell's static-over-random gap; rows are drift rates, columns rules, x the
four pools. Colour carries the one property the paper is about -- staleness-aware (blue) or not
(orange, UCB) -- with the rule named in each panel title, so identity is never colour alone.
Filled marker = Holm-significant; hollow = not. Grey tick = the +/-eps equivalence margin in
that pool's units. Palette: categorical slots 1-2, validated all-pairs (CVD dE 24.7).
"""
import json, os, sys
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "paper_figs")
C1, C2 = "#2a78d6", "#eb6834"                 # staleness-aware, lifetime-count UCB
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK3, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": "#e8e8e4", "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 200, "savefig.dpi": 300,
    "savefig.bbox": "tight", "lines.linewidth": 1.8,
})
EPS = 0.005
POOLS = ["Fixed", "Growing", "Shrinking", "Evolving"]
RULES = [("ucb", "UCB (lifetime count)", C2), ("ducb", "Discounted UCB", C1),
         ("swucb", "Sliding-window UCB", C1), ("ts", "Discounted Thompson", C1)]
THETAS = [("0.0", r"no drift"), ("0.02", r"drift $\theta=0.02$"), ("0.1", r"drift $\theta=0.1$")]

F = json.load(open("paperB_final_complete.json"))

fig, axes = plt.subplots(len(THETAS), len(RULES), figsize=(7.0, 5.2), sharey="row",
                         constrained_layout=True)
for r, (th, thlab) in enumerate(THETAS):
    for c, (fam, lab, col) in enumerate(RULES):
        ax = axes[r, c]
        ax.axhline(0, color=INK3, lw=0.8, zorder=1)
        for x, pool in enumerate(POOLS):
            cell = F.get(f"{pool}|{th}")
            if cell is None:
                continue
            g = 100 / cell["gap"]
            e = EPS * g
            ax.plot([x - 0.28, x + 0.28], [e, e], color="#c9c8c2", lw=1.0, zorder=1)
            ax.plot([x - 0.28, x + 0.28], [-e, -e], color="#c9c8c2", lw=1.0, zorder=1)
            k = cell["comparisons"][f"{fam} vs static"]
            sig = k["p_holm"] < 0.05
            ax.plot([x, x], [k["ci"][0] * g, k["ci"][1] * g], color=col, lw=1.6, zorder=3,
                    solid_capstyle="round")
            ax.plot([x], [k["diff"] * g], "o", ms=5.5, mew=1.5, mec=col,
                    mfc=col if sig else "white", zorder=4)
        ax.set_xticks(range(len(POOLS)))
        ax.set_xticklabels(POOLS if r == len(THETAS) - 1 else [], rotation=35, ha="right",
                           fontsize=7.5)
        ax.set_xlim(-0.6, len(POOLS) - 0.4)
        if r == 0:
            ax.set_title(lab, fontsize=8.5, color=INK)
        if c == 0:
            ax.set_ylabel(f"{thlab}\ngain vs tuned static\n(% of gap)", fontsize=8)
fig.text(0.5, -0.02, "Filled: Holm-significant.  Grey ticks: equivalence margin $\\pm\\varepsilon$.  "
         "Orange: bonus from a lifetime count; blue: uncertainty regrows with staleness.",
         ha="center", fontsize=7.5, color=INK2)
os.makedirs(OUT, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"figB1_rule_vs_static.{ext}"))
print("cells in figure:", sorted(F))
