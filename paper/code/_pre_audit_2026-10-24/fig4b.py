import json, os, re, sys
import numpy as np, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding="utf-8")
OUT = r"D:\Resume Projects\Research\R1-VAMS\paper_figs"
C1, INK, INK2, INK3 = "#2a78d6", "#0b0b0b", "#52514e", "#8a8984"
mpl.rcParams.update({"font.family":"serif","font.serif":["DejaVu Serif"],"font.size":8.5,
 "axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":INK3,
 "axes.labelcolor":INK,"text.color":INK,"xtick.color":INK2,"ytick.color":INK2,
 "axes.grid":True,"grid.color":"#e8e8e4","grid.linewidth":0.6,"legend.frameon":False,
 "figure.dpi":200,"savefig.dpi":300,"savefig.bbox":"tight","lines.linewidth":1.8})

# parse the M1 c-sweep from the run log
txt = open(r"C:\Users\omjad\AppData\Local\Temp\claude\D--Resume-Projects-Research-R1-VAMS\fb0929eb-cbf3-447f-b67d-e30d013cec30\tasks\benpknfuz.output", encoding="utf-8").read()
SC = ["Fixed","Growing","Shrinking","Evolving"]
sweep, best_static = {}, {}
for sc in SC:
    blk = txt.split("--- %s ---" % sc)[1].split("---")[0]
    sweep[sc] = {float(a): float(b) for a, b in re.findall(r"c=([\d.]+):([\d.]+)", blk)}
    best_static[sc] = max(float(b) for a, b in re.findall(r"P=(\d+):([\d.]+)", blk))

D = json.load(open("decisive_results.json"))
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)

y = np.arange(len(SC))
diff = [D[s]["diff"] for s in SC]
lo = [D[s]["diff"]-D[s]["ci"][0] for s in SC]; hi = [D[s]["ci"][1]-D[s]["diff"] for s in SC]
a.errorbar(diff, y, xerr=[lo,hi], fmt="o", ms=6, color=C1, mfc="white", mew=1.6,
           ecolor=C1, elinewidth=1.4, capsize=3, ls="none")
a.axvline(0, color=INK, lw=0.9)
a.axvline(0.005, color=INK3, lw=1.0, ls=(0,(3,2)))
a.annotate("SESOI 0.005", (0.0052, 3.35), fontsize=7, color=INK2, ha="left")
for i, s in enumerate(SC):
    a.annotate("best static $P{=}%d\%%$" % D[s]["bestP"], (diff[i], y[i]),
               textcoords="offset points", xytext=(0,10), fontsize=6.8, color=INK2, ha="center")
a.set_yticks(y); a.set_yticklabels(SC, fontsize=8); a.set_ylim(3.6,-0.9)
a.set_xlabel("paired $\Delta$:  UCB $-$ best static $P$"); a.set_xlim(-0.001,0.017)
a.set_title("UCB beats a correctly tuned baseline", fontsize=9, loc="left")
a.grid(axis="y", visible=False)

MK = {"Fixed":"o","Growing":"s","Shrinking":"^","Evolving":"D"}
cs = sorted(sweep["Fixed"])
for sc in SC:
    d = [sweep[sc][c]-best_static[sc] for c in cs]
    b.plot(cs, d, MK[sc]+"-", ms=4, lw=1.3, color=C1, mfc="white", mew=1.0, label=sc)
b.axhline(0, color=INK, lw=0.9)
b.axvspan(0.15, 0.3, color="#f0efe9", zorder=0, lw=0)
b.annotate("wins in all\nfour pools", (0.205, -0.0092), fontsize=6.8, color=INK2, ha="center")
b.set_xscale("log"); b.set_xlabel("exploration constant $c$  (log scale)")
b.set_ylabel("$\Delta$ vs best static $P$")
b.set_title("Sensitivity to $c$ (unbiased metric)", fontsize=9, loc="left")
b.set_ylim(-0.0125, 0.0155)
b.legend(fontsize=7, loc="lower left", handletextpad=0.3, ncol=2, columnspacing=1.0)
fig.savefig(os.path.join(OUT,"fig4_ucb_result.pdf")); fig.savefig(os.path.join(OUT,"fig4_ucb_result.png"))
print("fig4 regenerated on M1")
for sc in SC:
    print(f"  {sc:<10} best static={best_static[sc]:.4f}  " +
          "  ".join(f"c={c}:{sweep[sc][c]-best_static[sc]:+.4f}" for c in cs))
