"""
Paper A: analysis of pivotality_results.json (no simulation). Prints every number the text
quotes and writes paper_figs/fig5_pivotality.{pdf,png} and paperA/tables/{pivot,shift}.tex.
"""
import json, os, sys, numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding="utf-8")
from pivotality import OUT, CONDS, PGRID, CGRID, TUNE, EVAL, ckey
from engine_v import SCENARIOS

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "..", "paper_figs"); TAB = os.path.join(HERE, "..", "paperA", "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
R = json.load(open(os.path.join(HERE, OUT)))
NT = len(TUNE); POOLS = list(SCENARIOS)
NAME = dict(gauss="Gaussian", laplace="Laplace", uniform="uniform", t5="Student-$t_5$")
GRID = dict(static=PGRID, ucb=CGRID)

def arm(c, pool, mode, x):
    a = np.array(R[c][f"{pool}|{mode}|{x:g}"])     # rows: M1 M2 M3 churn pred audit; cols: seeds
    return a

# ---------------- 1. theorem: prediction vs measurement ----------------
rows = []
for n, V in CONDS:
    c = ckey(n, V)
    for pool in POOLS:
        for mode in GRID:
            for x in GRID[mode]:
                a = arm(c, pool, mode, x); b = a[1] - a[2]
                m, se = b.mean(), b.std(ddof=1) / np.sqrt(b.size)
                rows.append(dict(noise=n, V=V, pool=pool, mode=mode, x=x, meas=m, se=se,
                                 pred=a[4].mean(), audit=a[5].mean(), churn=a[3].mean()))
def summ(sel, label):
    meas = np.array([r['meas'] for r in sel]); pred = np.array([r['pred'] for r in sel])
    se = np.array([r['se'] for r in sel]); aud = np.array([r['audit'] for r in sel])
    inside = np.abs(pred - meas) <= 1.96 * se
    big = meas > 0.005
    print(f"{label:<22} n={len(sel):>3}  pred within 95% CI: {inside.mean()*100:5.1f}%  "
          f"max|pred-meas|={np.abs(pred-meas).max():.5f}  "
          f"rel.err (bias>0.005): median {np.median(np.abs(pred[big]/meas[big]-1))*100:.1f}% max {np.abs(pred[big]/meas[big]-1).max()*100:.1f}%  "
          f"audit/meas (bias>0.005): {np.min(aud[big]/meas[big]):.2f}-{np.max(aud[big]/meas[big]):.2f} median {np.median(aud[big]/meas[big]):.2f}")
    return inside.mean(), np.abs(pred - meas).max()
print("== Theorem: predicted vs measured E[M2-M3] ==")
summ(rows, "all")
for mode in GRID: summ([r for r in rows if r['mode'] == mode], f"  {mode}")
for n, V in CONDS: summ([r for r in rows if r['noise'] == n and r['V'] == V], f"  {n} V={V:g}")

# ---------------- 2. slope of bias on churn ----------------
print("\n== Bias per unit churn (fit through origin) ==")
slope = {}
for n, V in CONDS:
    for mode in GRID:
        sel = [r for r in rows if r['noise'] == n and r['V'] == V and r['mode'] == mode]
        x = np.array([r['churn'] for r in sel]); y = np.array([r['meas'] for r in sel])
        s = float(x @ y / (x @ x)); r2 = 1 - ((y - s * x) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        slope[(n, V, mode)] = (s, r2)
        print(f"  {n:<8} V={V:<5g} {mode:<6} slope={s:.4f}  slope/V={s/V:.3f}  R2={r2:.3f}")

# ---------------- 3. optimum shift and its cost ----------------
print("\n== Optimum under M1 vs M2 (picked on TUNE), cost on EVAL as % of gain over random ==")
shift = {}
for n, V in CONDS:
    c = ckey(n, V)
    for pool in POOLS:
        rnd = np.mean(R[c][f"{pool}|random"])
        for mode in GRID:
            A = {x: arm(c, pool, mode, x) for x in GRID[mode]}
            x1 = max(GRID[mode], key=lambda x: A[x][0][:NT].mean())
            x2 = max(GRID[mode], key=lambda x: A[x][1][:NT].mean())
            e1, e2 = A[x1][0][NT:].mean(), A[x2][0][NT:].mean()
            cost = 100 * (e1 - e2) / (e1 - rnd)
            shift[(n, V, pool, mode)] = (x1, x2, cost)
for mode in GRID:
    for n, V in CONDS:
        v = [shift[(n, V, p, mode)] for p in POOLS]
        print(f"  {mode:<6} {n:<8} V={V:<5g} " + "  ".join(f"{p[:3]}: {a:g}->{b:g} ({c:.0f}%)" for p, (a, b, c) in zip(POOLS, v)))

if __name__ == "__main__":
    # ---------------- tables ----------------
    L = [r"\begin{tabular}{llrrrr}", r"\toprule",
         r"Noise & $V$ & \multicolumn{2}{c}{percentile} & \multicolumn{2}{c}{UCB}\\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r" & & slope & in CI & slope & in CI\\", r"\midrule"]
    for n, V in CONDS:
        cells = []
        for mode in GRID:
            sel = [r for r in rows if r['noise'] == n and r['V'] == V and r['mode'] == mode]
            ins = np.mean([abs(r['pred'] - r['meas']) <= 1.96 * r['se'] for r in sel])
            cells += [f"${slope[(n, V, mode)][0]:.3f}$", f"${100*ins:.0f}\\%$"]
        L.append(f"{NAME[n]} & ${V:g}$ & " + " & ".join(cells) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "pivot.tex"), "w").write("\n".join(L) + "\n")

    L = [r"\begin{tabular}{ll" + "r" * 4 + "}", r"\toprule",
         r"Noise & $V$ & Fixed & Grow. & Shrink. & Evol.\\", r"\midrule"]
    for n, V in CONDS:
        L.append(f"{NAME[n].replace('Student-', '')} & ${V:g}$ & " + " & ".join(
            f"${shift[(n, V, p, 'static')][0]:g}{{\\to}}{shift[(n, V, p, 'static')][1]:g}$ ({f"{shift[(n, V, p, 'static')][2]:.0f}".replace('-', '$-$')})"
            for p in POOLS) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "shift.tex"), "w").write("\n".join(L) + "\n")

    # ---------------- figure ----------------
    C1, C2, INK, INK2, INK3 = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#8a8984"
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK3,
        "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": "#e8e8e4", "grid.linewidth": 0.6,
        "legend.frameon": False, "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight"})
    MK = dict(gauss="o", laplace="s", uniform="^", t5="D")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.9), constrained_layout=True)
    top = max(r['meas'] for r in rows) * 1.05
    a1.plot([0, top], [0, top], color=INK3, lw=0.8, zorder=1)
    for r in rows:
        a1.plot(r['pred'], r['meas'], MK[r['noise']], ms=2.6, mfc="none",
                mec=C1 if r['mode'] == 'static' else C2, mew=0.7, zorder=2)
    a1.set_xlabel(r"predicted $\frac{1}{N'}\sum_j g(u_j)$"); a1.set_ylabel(r"measured $M_2-M_3$")
    a1.text(0.03, 0.93, "percentile", color=C1, transform=a1.transAxes)
    a1.text(0.03, 0.84, "UCB", color=C2, transform=a1.transAxes)
    for k, (n, lab) in enumerate(NAME.items()):
        a1.plot([0.62], [0.30 - 0.08 * k], MK[n], ms=3.5, mfc="none", mec=INK2, transform=a1.transAxes)
        a1.text(0.66, 0.28 - 0.08 * k, lab.replace("$t_5$", "t$_5$").replace("$", ""), color=INK2, transform=a1.transAxes, fontsize=7.5)
    Vs = [V for n, V in CONDS if n == 'gauss']; sQ = 0.1
    for mode, col, lab, dy in (("static", C1, "percentile", 6), ("ucb", C2, "UCB", -12)):
        per = np.array([[shift[("gauss", V, p, mode)][2] for p in POOLS] for V in Vs])
        for k in range(len(POOLS)):
            a2.plot([V / sQ for V in Vs], per[:, k], color=col, lw=0.6, alpha=0.35)
        a2.plot([V / sQ for V in Vs], per.mean(1), "-o", color=col, ms=3.5, lw=1.8)
        a2.annotate(lab, (Vs[-1] / sQ, per.mean(1)[-1]), xytext=(-48, dy), textcoords="offset points", color=col)
    a2.axvline(2.0, color=INK3, lw=0.8, ls=(0, (3, 2)))
    a2.annotate("reference\nparameters", (2.0, 74), xytext=(-52, 0), textcoords="offset points", color=INK2, fontsize=7.5)
    a2.set_xscale("log"); a2.set_xticks([V / sQ for V in Vs]); a2.set_xticklabels([f"{V/sQ:g}" for V in Vs])
    a2.minorticks_off(); a2.set_ylim(-3, 85)
    a2.set_xlabel(r"noise-to-spread ratio $V/\sigma_Q$"); a2.set_ylabel(r"gain forfeited by tuning on $M_2$ (%)")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig5_pivotality.{ext}"))
    print("\nwrote fig5_pivotality, tables/pivot.tex, tables/shift.tex")
