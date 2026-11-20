# Reproducibility bundle

Manuscript: `main.tex`. Figures: `../paper_figs/`. Code and raw per-seed results: `code/`.

## Build

No LaTeX toolchain was available on the machine where this was drafted, so **`main.tex`
has not been compiled**. It is venue-neutral (`article`, `twocolumn`, no class-specific
commands), so retargeting should be a one-line `\documentclass` swap for `aamas.cls`,
`sigconf`, or `IEEEtran`. Build once before trusting the layout.

```
cd paper && pdflatex main && pdflatex main
```

## Engine

`code/engine_v.py` is the single simulation engine. It is a vectorized
reimplementation of Kruijssen et al. (2024), **validated against a direct transcription
of the reference notebook engine**: `code/xcheck.py` compares the two across all four
scenarios and both fixed/adaptive modes over 24 seeds and finds no difference
(Welch p > 0.56 in all 8 comparisons, means within 0.003).

It returns six series per run:

| index | name | meaning |
|---|---|---|
| 0 | `mean_T` | **M2** — the metric used in the original notebooks. Biased. |
| 1 | `P_hist` | realised replacement percentile per epoch |
| 2 | `S_hist` | sigma_A per epoch |
| 3 | `lat_Q`  | **M3** — latent quality of the post-selection set (ground truth) |
| 4 | `churn`  | fraction of the active set replaced |
| 5 | `obs_A`  | **M1** — mean(T_act), measured before selection consumes it. Unbiased. |

**Use index 5 (M1) for any quality comparison.** Index 0 is retained only to reproduce
the bias.

Modes: `random`, `static`, `adaptive` (VAMS), `ucb`, `greedy`, `sched`.
Non-stationarity is an Ornstein–Uhlenbeck process on latent quality (`ou_theta`), scaled
so the *stationary* variance stays at sigma_Q for every theta — otherwise drift is
confounded with a widening quality distribution.

## Experiments

| script | produces | runtime |
|---|---|---|
| `xcheck.py` | engine equivalence vs reference | ~6 min |
| `metrics.py` | 3 metrics x P sweep (Fig 1, Fig 2) | ~10 min |
| `final.py` | 8 mechanisms x 2 metrics (Fig 3) | ~11 min |
| `headroom.py` | oracle two-phase schedule bound | ~9 min |
| `decisive.py` / `decisive2.py` | UCB vs tuned static, paired (Fig 4) | ~14 min |
| `memory_drift.py` | estimator memory x drift surface | ~5 min |
| `study.py` | TOST equivalence, power table | ~6 min |

Run with `WORKERS=4`. **This matters**: the development machine has 8 GB RAM and
`os.cpu_count()-1` workers exhausts it, failing partway through as a confusing
`BrokenProcessPool` rather than an obvious OOM.

```
cd code && WORKERS=4 python metrics.py
```

Each script writes per-seed arrays to its `*_results.json`, which are included — the
figures rebuild from those without re-running the simulations:

```
python figures.py && python fig4b.py
```

## Statistics

`code/stats_util.py`: Welch's t-test with Hedges-corrected d, paired t-test with d_z,
TOST equivalence (unpaired and paired), and power/sample-size helpers.

Every "no difference" claim in the paper is a **TOST equivalence result** at
epsilon = 0.005 (10% of the merit-over-random gap), not a failure to reject. This is
load-bearing: at n = 10 seeds — the protocol in the original notebooks — a two-sample
test has 80% power only for d > 1.3, so non-significance there carries almost no
information. The paper uses n = 100, and n = 120 paired for the final comparison.

## Known gaps

- `main.tex` is uncompiled (no LaTeX toolchain available).
- Proposition 1 is proved in sketch for a stationary regime and verified numerically
  (M1 and M3 agree to 4 decimals at every P in every scenario); it is not proved in
  full generality.
- The regret framing cites Chen et al. (2013) by analogy; no bound specific to this
  mechanism is derived. The sortition setting differs in that the pool itself grows
  and shrinks.
- Simulation only. No deployed-network validation.
