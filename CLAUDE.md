# R1-VAMS — research project notes for Claude

Two conference papers from one simulation study of merit-based sortition
(Kruijssen et al. 2024). Everything that produces a number lives in `paper/code/`;
`paper/README.md` maps every script to the section it feeds.

## Goal and state (as of 2026-11-20)

- **Paper A** — `paper/paperA/main.tex`, *Score Before You Select*. **Complete draft**, compiles
  clean, no `\pending` left. Claim: scoring the newly selected set with the observations that
  selected it (M2) is biased upward in proportion to churn; volatility-adaptive P (VAMS) is
  equivalent to static P; a hindsight two-phase schedule gains ≤0.002. Remaining: final
  proofread, venue formatting.
- **Paper B** — `paper/paperB/main.tex`, *Optimism Must Decay with Staleness*. **Draft written;
  every number is a red `\pending{}` awaiting the runs below.** Framing is decided (the user
  delegated it): uncertainty-directed sortition beats tuned percentile sortition without drift;
  under drift, UCB (lifetime count) loses in 7/8 settings while Discounted UCB never loses, and
  with one configuration and unknown drift D-UCB is never worse than the best single P.
  New bib entries in `paper/paperB/refs.bib` are marked `% VERIFY` (written from memory).

## Resume here — next steps in order

1. **Finish the complete-grid run** (`paper/code/paperB_complete.py`). It died on the old PC
   after 1 of 12 cells (`Fixed|0.0`); it checkpoints per cell to
   `paperB_complete_results.json` and resumes. ~7 h at 6 workers (drift cells are slowest).
   - Windows: `powershell -File paper/code/run_paperB_complete.ps1` launched detached
     (`Start-Process ... -WindowStyle Hidden`, then set PriorityClass BelowNormal).
   - Linux/macOS: `cd paper/code && WORKERS=6 nohup python -u paperB_complete.py > logs/paperB_complete.log 2>&1 &`
2. **Engine patch for the whitewashing threat model**, only after step 1 has exited
   (never edit `engine_v.py` while a pool is live — workers re-import it):
   - First `python engine_regress.py . before.json` (snapshot on THIS machine).
   - Add to `run()`: `adv_frac=0.0, adv_Q=0.0, prior_n0=0.0, stats=None`. Adversaries are
     flagged at creation (draw from rng ONLY when adv_frac>0, so old outputs stay bitwise
     identical); an adversary that has been observed and is not selected leaves and rejoins at
     once as a fresh identity with the same latent quality (new lifetime, n_obs=0); in bandit
     modes a newcomer gets `bN=prior_n0, bS=prior_n0*target_inactive`; when `stats` is a dict,
     fill `stats['adv_share']` (adversary share of the selected set per epoch).
   - Then `python engine_regress.py . after.json` and `--diff before.json after.json` must
     report 0 differences.
3. **Run `paperB_attack.py`** (already written; reads picks from the complete-grid results).
   Needs a runner like `run_paperB_complete.ps1`. ~1 h.
4. **Fill Paper B**: `python paperB_final.py --complete`, `python paperB_robust.py --complete`,
   `python paperB_figures.py` (writes `paper_figs/figB1_rule_vs_static.*`), then replace every
   `\pending{}` in `paper/paperB/main.tex` and add the figure and tables. Re-check the claims
   in the abstract against the complete-grid numbers — they were drafted from the earlier
   (multi-round) results and some may shift.

## Rules that hold every number together

- **Quality = M1** (`run()[5]`), never index 0 (M2, biased) — this is Paper A's whole point.
- **Tune on TUNE seeds, report on disjoint EVAL seeds, paired.** Never pick and report on the
  same seeds. Holm across all comparisons; "equivalent" = TOST at eps=0.005.
- **No tuned pick on a grid boundary** (natural limits floor=0, gamma=1 excepted); extend the
  grid instead. This flipped the D-UCB-vs-Thompson result once.
- **Never cite** anything in `paper/code/_pre_audit_2026-10-24/` or `VAMS_paper_v2.docx`
  (pre-audit numbers, superseded). `memory_drift.py` is superseded too (in-sample picks).
- Engine changes: regression-hash before/after (`engine_regress.py`), 0 differences allowed.

## Machine constraints

- The original PC had 8 GB RAM with ~1 GB free; each worker ~35–90 MB, so `WORKERS=6`.
  Claude Code reclaimed background shells under memory pressure and killed long runs, hence
  the detached `run_*.ps1` launchers and per-cell checkpoints. Size WORKERS to the new PC.
- PowerShell `*>` writes UTF-16 logs: read them with `tr -d '\000'` (or `Get-Content`).

## Build

- `pip install -r requirements.txt` (Python 3.13).
- LaTeX: `tectonic main.tex` in `paper/paperA` or `paper/paperB` (tectonic is a single
  binary; download it — it was kept in a temp folder on the old PC, not in the repo).

## Working preference

The user wants Claude to reason critically, decide the highest-value next step toward the two
papers, and execute it — not pause for confirmation or run status-only checks. Report results
honestly, including when they overturn an earlier claim.
