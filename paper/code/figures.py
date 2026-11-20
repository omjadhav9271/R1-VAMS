"""Publication figures.

Palette: validated categorical slots 1-3 (blue/orange/aqua) -- all-pairs CVD dE 9.2,
normal-vision dE 24.0 (validate_palette.js, light mode). Aqua carries a contrast WARN,
so every series is direct-labelled and line style doubles as a print-safe secondary
encoding. No dual axes: churn gets its own panel rather than a second y-scale.
"""
import json, os, sys
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "paper_figs")
os.makedirs(OUT, exist_ok=True)

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"   # M1 unbiased, M2 as-implemented, M3 latent
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

SC = ["Fixed", "Growing", "Shrinking", "Evolving"]
M = json.load(open("metrics_results.json"))
PG = sorted({float(k.split("|")[1]) for k in M})


def arr(sc, P, i):
    key = f"{sc}|{int(P) if float(P).is_integer() else P}"
    return np.array(M[key][i])


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"))
    plt.close(fig)


# ------------------------------------------------------------------ Figure 1
fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), constrained_layout=True)
for ax, sc in zip(axes.ravel(), SC):
    m1 = np.array([arr(sc, P, 0).mean() for P in PG])
    m2 = np.array([arr(sc, P, 1).mean() for P in PG])
    m3 = np.array([arr(sc, P, 2).mean() for P in PG])
    ax.axvspan(20, 40, color="#f0efe9", zorder=0, lw=0)
    ax.plot(PG, m2, color=C2, lw=2.2, zorder=3)
    ax.plot(PG, m1, color=C1, lw=1.8, zorder=4)
    ax.plot(PG, m3, color=C3, lw=1.4, ls=(0, (3, 2)), zorder=5)
    for v, c, dx, dy in ((m2, C2, 4, 7), (m1, C1, 5, -30)):
        k = int(np.argmax(v))
        ax.plot([PG[k]], [v[k]], "o", ms=5, mfc="white", mec=c, mew=1.6, zorder=6)
        ax.annotate("P*=%d%%" % PG[k], (PG[k], v[k]), textcoords="offset points",
                    xytext=(dx, dy), fontsize=7.5, color=c)
    ax.set_title(sc + " pool", fontsize=9, loc="left", color=INK)
    ax.set_xlim(0, 100)
    ax.set_ylim(0.195, 0.30)
    if sc in SC[2:]:
        ax.set_xlabel("replacement percentile $P$ (%)")
    if sc in ("Fixed", "Shrinking"):
        ax.set_ylabel("mean active-set quality")

axes[0, 0].annotate("$M_2$ (score selected set)", (24, 0.288), color=C2,
                    fontsize=7.5, fontweight="bold")
axes[0, 0].annotate("$M_1$ (score\nbefore selecting)", (47, 0.212), color=C1,
                    fontsize=7.5, fontweight="bold")
axes[0, 0].annotate("$M_3$ (latent, dashed)", (60, 0.199), color=C3,
                    fontsize=7.5, fontweight="bold")
axes[0, 1].annotate("shaded: $P\\in[20,40]\\%$\noptimum reported\nby Kruijssen et al.", (42, 0.202),
                    fontsize=7, color=INK2)
save(fig, "fig1_metric_divergence")
print("fig1 ok")

# ------------------------------------------------------------------ Figure 2
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)
MK = {"Fixed": "o", "Growing": "s", "Shrinking": "^", "Evolving": "D"}
xs, ys = [], []
for sc in SC:
    ch = np.array([arr(sc, P, 3).mean() for P in PG])
    bias = np.array([arr(sc, P, 1).mean() - arr(sc, P, 0).mean() for P in PG])
    xs += list(ch)
    ys += list(bias)
    a.plot(ch, bias, MK[sc], ms=4.5, mfc="white", mec=C1, mew=1.1, ls="none", label=sc)

xs, ys = np.array(xs), np.array(ys)
slope = float(xs @ ys / (xs @ xs))                      # fit through the origin
r2 = 1.0 - ((ys - slope * xs) ** 2).sum() / ((ys - ys.mean()) ** 2).sum()
grid = np.linspace(0, xs.max() * 1.03, 50)
a.plot(grid, slope * grid, color=INK3, lw=1.2, ls=(0, (4, 2)), zorder=0)
a.annotate("bias $=%.3f \\times$ churn\n$R^2=%.3f$" % (slope, r2), (0.04, 0.80),
           xycoords="axes fraction", fontsize=7.5, color=INK2)
a.set_xlabel("fraction of active set replaced per epoch")
a.set_ylabel("$M_2 - M_1$   (bias)")
a.set_title("Bias is proportional to churn", fontsize=9, loc="left")
a.legend(fontsize=7, loc="lower right", handletextpad=0.3)

for sc in SC:
    ch = np.array([arr(sc, P, 3).mean() for P in PG])
    b.plot(PG, ch, MK[sc] + "-", ms=3.5, lw=1.3, color=C1, mfc="white", mew=1.0, label=sc)
b.set_xlabel("replacement percentile $P$ (%)")
b.set_ylabel("churn per epoch")
b.set_title("Churn is driven by $P$", fontsize=9, loc="left")
b.legend(fontsize=7)
save(fig, "fig2_bias_churn")
print("fig2 ok  slope=%.4f  R2=%.4f" % (slope, r2))

# ------------------------------------------------------------------ Figure 3
F = json.load(open("final_results.json"))
MECHS = ["random", "static P=20", "static P=25", "static P=40", "static P=70",
         "VAMS [10,40]", "VAMS [55,85]"]
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True, sharey=True)
panels = [("m1", C1, "Scored on $M_1$ (score before selecting)"),
          ("m2", C2, "Scored on $M_2$ (score selected set)")]
for ax, (mk, col, ttl) in zip(axs, panels):
    y = np.arange(len(MECHS))
    ref = np.array(F["Evolving"]["static P=25"][mk]).mean()
    vals, errs = [], []
    for m in MECHS:
        v = np.array(F["Evolving"][m][mk])
        vals.append(v.mean() - ref)
        errs.append(1.96 * v.std(ddof=1) / np.sqrt(len(v)))
    ax.barh(y, vals, xerr=errs, color=col, height=0.62,
            error_kw=dict(lw=0.9, ecolor=INK2))
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(MECHS, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_title(ttl, fontsize=9, loc="left")
    ax.set_xlabel(r"$\Delta$ quality vs static $P{=}25\%$")
    ax.grid(axis="y", visible=False)
save(fig, "fig3_ranking_inversion")
print("fig3 ok")

# ------------------------------------------------------------------ Figure 4
# Top: each VAMS variant at its realised mean P, over the static-P curve (M1, relative to
# static P=25). Bottom: adaptation strength (sd of P_t) against the gain over static P at
# the variant's own mean P -- flat at zero means adapting adds nothing beyond the average.
if os.path.exists("vams_sweep_results.json"):
    VS = json.load(open("vams_sweep_results.json"))
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.9), constrained_layout=True, sharey="row")
    for j, sc in enumerate(SC):
        a, b = axes[0, j], axes[1, j]
        ref = arr(sc, 25, 0).mean()
        pg = [P for P in PG if 10 <= P <= 50]
        a.plot(pg, [arr(sc, P, 0).mean() - ref for P in pg], color=INK3, lw=1.3, zorder=1)
        b.axhspan(-0.005, 0.005, color="#f0efe9", lw=0, zorder=0)
        for lab, r in sorted(VS[sc].items(), key=lambda kv: kv[0].startswith("sigmoid")):
            sig = lab.startswith("sigmoid")
            sty = dict(marker="o" if sig else "D", ms=3.4 if sig else 4.0, ls="none",
                       mfc="white" if sig else C2, mec=C1 if sig else C2, mew=1.0, zorder=3)
            a.plot(r["P_mean"], r["vs25"][0], **sty)
            b.plot(r["P_sd"], r["vs_meanP"][1], **sty)
        for ax in (a, b):
            ax.axhline(0, color=INK, lw=0.7, zorder=2)
        a.set_title(sc, fontsize=8.5, loc="left")
        a.set_xlim(15, 45); a.set_ylim(-0.0125, 0.004); a.set_xlabel("realised mean $P_t$ (%)", fontsize=7.5)
        b.set_xlim(-0.5, 16); b.set_xlabel("sd of $P_t$ (adaptation strength)", fontsize=7)
    axes[0, 0].set_ylabel(r"$\Delta M_1$ vs static $P{=}25\%$", fontsize=7.5)
    axes[1, 0].set_ylabel(r"$\Delta M_1$ vs static $P$" "\n" "at the same mean", fontsize=7.5)
    axes[0, 0].annotate("static $P$", (40, arr("Fixed", 40, 0).mean() - arr("Fixed", 25, 0).mean()),
                        textcoords="offset points", xytext=(4, 5), fontsize=7, color=INK2)
    axes[1, 0].annotate(r"$\pm\varepsilon$", (14.5, 0.0035), fontsize=7, color=INK2, ha="right")
    axes[0, 2].plot([], [], "o", ms=3.4, mfc="white", mec=C1, label=r"sigmoid, $k\times W$")
    axes[0, 2].plot([], [], "D", ms=4.0, mfc=C2, mec=C2, label="other curves")
    axes[0, 2].legend(fontsize=6.5, loc="lower center", handletextpad=0.2)
    save(fig, "fig4_vams_meanP")
    print("fig4 ok")

print("\nwrote figures to " + OUT)
