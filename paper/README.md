# Reproducibility bundle

- `paperA/` — *Score Before You Select*: the selection-conditioned evaluation bias, what it
  invalidates, and the volatility-adaptive-P null. ACM `sigconf`, anonymous.
- `paperB/` — *Optimism Must Decay with Staleness*: uncertainty-directed merit sortition
  under drift, and the whitewashing threat model. Same template; draft.
- `code/` — simulation engine, every experiment script, and the raw per-seed results each
  one writes. Every number in the paper is regenerated from these.
- `../paper_figs/` — figures, rebuilt from the JSON results by `figures.py` (no simulation).
- `code/_pre_audit_2026-10-24/` — the code, results and draft as they stood before the
  2026-10-24 audit. Kept for the record; do not cite numbers from it.

## Build

```
cd paperA && tectonic main.tex
```
(or `pdflatex main && bibtex main && pdflatex main && pdflatex main`).

## Engine

`code/engine_v.py` is the single engine: a vectorized reimplementation of Kruijssen et al.
(2024). `code/ref_engine.py` is a verbatim transcription of the notebook engine
(`adaptive_vams_v2.ipynb`). `xcheck.py` compares them on 100 seeds per configuration,
4 pools x {static, adaptive} x {M1, M2}: all 16 comparisons are TOST-equivalent at
eps = 0.005 (largest |diff| = 0.0015).

`run()` returns six per-epoch series:

| index | name | meaning |
|---|---|---|
| 0 | `mean_T` | **M2** — scores the newly selected set with the observations that selected it. Biased. |
| 1 | `P_hist` | realised replacement percentile |
| 2 | `S_hist` | sigma_A, sd of the active set's observations |
| 3 | `lat_Q`  | **M3** — latent quality of the selected set (ground truth) |
| 4 | `churn`  | fraction of the active set replaced |
| 5 | `obs_A`  | **M1** — mean observed quality of the fielded set, before selection. Unbiased. |

**Use index 5 (M1) for any quality comparison.** Index 0 exists only to reproduce the bias.

Modes: `random`, `static`, `adaptive` (VAMS; `form=` one of six response curves), `ucb`,
`greedy`, `sched` (two-phase schedule), and the Paper B non-stationary bandit baselines
`ducb` (Discounted UCB, `gamma=`), `swucb` (sliding-window UCB, `sw_tau=`) and `ts`
(Gaussian Thompson sampling, `ts_scale=`, `gamma=`). Adding the baselines (2026-11-06) left
every pre-existing mode's output bitwise identical (80 seed x mode x pool hashes). `alpha_floor` sets the memory of the running-mean
estimator used by `ucb`/`greedy`. `ou_theta` adds mean-reverting drift to latent quality
with the stationary variance held at sigma_Q. The whitewashing threat model (2026-11-20)
adds `adv_frac=`, `adv_Q=`, `prior_n0=` and `stats=` (a dict that receives `adv_share`); at
their defaults every mode is bitwise identical to the previous engine (`xcheck_bitwise.py`:
144 hashes, 9 modes x 4 pools x 2 drift rates x 2 seeds). The previous engine and scripts are
in git commit 15ce5c6. Paper A's pivotality analysis (2026-11-20) adds `noise=` ('gauss'
default, 'laplace', 'uniform', 't5', all with sd V) and, when `stats={'pivot': True}` is
passed in static/ucb mode, per-epoch `pivot_bias` (the Theorem 1 prediction of M2-M3 from
each fielded participant's flip point) and `pivot_audit` (the same with latent quality
replaced by the participant's own estimate). Defaults remain bitwise identical (144/144).

## Experiments

Run each from `code/` with `WORKERS=4` (see below). Seeds are disjoint across scripts.

| script | what | used in | runtime @4 workers |
|---|---|---|---|
| `xcheck.py` | engine equivalence vs the notebook engine | A §2 | 4 min |
| `metrics.py` | M1/M2/M3 and churn over a P sweep | A Fig 1–2, Tab 2 | 10 min |
| `generality.py` | the same bias audit on UCB selection | A §3.4 | 7 min |
| `final.py` | 8 mechanisms x 2 metrics | A Fig 3, Tab 3 (VAMS vs P=25) | 11 min |
| `vams_sweep.py` | VAMS k x W grid and six response curves, on M1 | A §5, Fig 4 | ~30 min |
| `vams_drift.py` | VAMS under OU drift in latent quality | A §5 | 17 min |
| `noise_floor.py` | sigma_A fluctuation vs estimator noise floor | A §5, Tab 4 | 2 min |
| `headroom.py` | hindsight oracle over two-phase schedules for P | A §6, Tab 5 | ~80 min @6 |
| `pivotality.py` | Theorem 1 prediction and audit vs measured bias; optimum shift under 7 noise conditions (Gaussian V=0.05-0.4, Laplace, uniform, t5) x 4 pools x P and c grids, 200 seeds (tune/eval split) | A §3.2, §5, Fig 5, Tabs 3-4 | 21 min @20 |
| `pivotality_report.py` | prints every §5 number; writes `paperA/tables/{pivot,shift}.tex` and `fig5_pivotality` (no simulation) | A §5 | seconds |
| `ucb_heldout.py` | UCB vs tuned static, held-out tuning, 2x2 ablation | Paper B | 38 min |
| `memory_drift.py` | estimator memory x latent drift, on M1. Superseded: fixes P and c across drift and picks the floor in-sample | — | 16 min |
| `paperB_baselines.py` | UCB vs tuned static, D-UCB, SW-UCB, Thompson; every family tuned on held-out seeds, 4 pools x 3 drift levels | Paper B | ~3.5 h @6 |
| `paperB_edge.py` | extends each grid past every truncated edge pick (two rounds; round 1 kept as `paperB_edge_results.round1.json`) | Paper B | ~35 min per round @6 |
| `paperB_final.py` | merges the two above into the final per-cell table, 108 comparisons Holm-adjusted (no simulation) | Paper B | seconds |
| `paperB_robust.py` | one configuration per family fixed without knowing the drift: leave-one-cell-out (no simulation) | Paper B | seconds |
| `paperB_summary.py` | quick look at a partial `paperB_baselines_results.json` (no simulation) | — | seconds |
| `paperB_complete.py` | the single final grid (489 configs): union of the main grid, every edge extension, and the 2026-11-20 extensions to the natural limits (SW-UCB tau=1000 and c=0, TS scale=0 and gamma to 0.5, D-UCB gamma to 0.6, UCB floor to 1); no pick on a boundary. Reruns fill only missing configs | Paper B, all tables | ~25 min @20 (24-core) |
| `paperB_final.py --complete`, `paperB_robust.py --complete` | final table and leave-one-cell-out on the complete grid | Paper B | seconds |
| `paperB_attack.py` | whitewashing threat model: adversarial fraction x rule x newcomer prior n0 in {1,3,10,30,100} | Paper B §7 | 2 min @20 |
| `paperB_tables.py` | writes `paperB/tables/*.tex` and prints every number the prose quotes; Holm-adjusts the 144 attack comparisons (no simulation) | Paper B | seconds |
| `xcheck_bitwise.py` | new engine at defaults vs old engine (`engine_orig.py`), bitwise | — | seconds |
| `study.py` | legacy; superseded by the scripts above, not used in either paper | — | 8 min |

Runtimes are at 4 workers unless marked. `run_rest.ps1` (`vams_sweep`, `headroom`) and
`run_paperB.ps1` (`paperB_baselines`) run as detached, low-priority processes with 6
workers, for machines where background jobs get reclaimed under memory pressure. All three
scripts checkpoint per pool and resume. On Linux, `run_paperB_complete.sh` and
`run_paperB_rest.sh` do the same (`nice`, detached, default `WORKERS=20` for a 24-core / 62 GB
machine; memory use stays near 6 GB).

```
cd code && WORKERS=4 python ucb_heldout.py
python figures.py
```

**Worker count matters.** The development machine has 8 GB RAM; `os.cpu_count()-1`
workers exhaust it and the run dies partway as a `BrokenProcessPool`, not an obvious OOM.
Never edit `engine_v.py` while a pool is running: newly spawned workers re-import it.

## Statistics

`code/stats_util.py`: Welch t-test with Hedges-corrected d; paired t-test with d_z; TOST
equivalence (unpaired and paired); power and sample-size helpers. Every "no difference"
claim is a TOST result at eps = 0.005 (10% of the merit-over-random gap), never a failure
to reject. Hyperparameters compared in Paper B are tuned on seeds disjoint from those
they are evaluated on.

## Known limits

- Simulation only, in the model of Kruijssen et al. (latent quality fixed per participant
  unless `ou_theta > 0`).
- The headroom bound covers two-phase open-loop schedules on a 7-value grid, not all
  adaptive policies.
