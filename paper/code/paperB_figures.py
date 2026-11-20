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
# legend lives in the LaTeX caption
os.makedirs(OUT, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"figB1_rule_vs_static.{ext}"))
print("cells in figure:", sorted(F))


# ------------------------------------------------------------------ Figure B2
# Whitewashing defence, as the newcomer prior n0 grows (none = undefended D-UCB). One line per
# pool x drift cell: colour = pool, line style = drift. (a) a=20% adversarial identities and
# (b) honest pool, both as D-UCB's paired difference from percentile sortition (1e-3);
# (c) adversaries' share of the active set at a=20%, thin lines = percentile sortition.
A = json.load(open("paperB_attack_results.json"))
N0 = [0, 1, 3, 10, 30, 100]; XS = [0.33, 1, 3, 10, 30, 100]
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.45), constrained_layout=True)
for cell, col, ls, lab in (("Fixed|0.0", C1, "-", "Fixed, no drift"),
                           ("Fixed|0.02", C1, (0, (3, 2)), r"Fixed, $\theta=0.02$"),
                           ("Evolving|0.0", C2, "-", "Evolving, no drift"),
                           ("Evolving|0.02", C2, (0, (3, 2)), r"Evolving, $\theta=0.02$")):
    q = A[cell]["quality"]; sh = A[cell]["adv_share"]
    arm = lambda n: "ducb" if n == 0 else f"ducb+prior{n:g}"
    d = lambda n, f: 1000 * (np.mean(q[f"{arm(n)}|{f}"]) - np.mean(q[f"static|{f}"]))
    kw = dict(ls=ls, color=col, lw=1.4, marker="o", ms=2.8)
    axs[0].plot(XS, [d(n, "0.2") for n in N0], label=lab, **kw)
    axs[1].plot(XS, [d(n, "0.0") for n in N0], **kw)
    axs[2].plot(XS, [100 * np.mean(sh[f"{arm(n)}|0.2"]) for n in N0], **kw)
    axs[2].axhline(100 * np.mean(sh["static|0.2"]), ls=ls, color=col, lw=0.7, alpha=0.55)
for a_, t in zip(axs, ("(a) 20% whitewashers", "(b) honest pool", "(c) adversaries in active set")):
    a_.set_xscale("log"); a_.set_xticks(XS); a_.set_xticklabels(["none", "1", "3", "10", "30", "100"])
    a_.minorticks_off(); a_.set_xlabel(r"newcomer prior $n_0$"); a_.set_title(t, fontsize=8.5)
for a_ in axs[:2]:
    a_.axhline(0, color=INK3, lw=0.8)
axs[0].set_ylabel(r"D-UCB $-$ static ($10^{-3}$)"); axs[1].set_ylabel(r"D-UCB $-$ static ($10^{-3}$)")
axs[2].set_ylabel("share (%)")
axs[0].legend(loc="lower right", fontsize=6.5, handlelength=2.2)
fig.savefig(os.path.join(OUT, "figB2_whitewashing.pdf")); fig.savefig(os.path.join(OUT, "figB2_whitewashing.png"))
plt.close(fig)
print("wrote figB2_whitewashing")
