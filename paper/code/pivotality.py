"""
Paper A: the pivotality formula for the selection-conditioned bias, its log-only audit, and the
robustness of the optimum shift to the noise law and noise level.

Theorem (pivotality): with every other draw held fixed, participant j's retention is a step in
its own noise eps_j at a flip point u_j, so its contribution to E[M2 - M3] is
g(u_j) = E[eps 1{eps > u_j}] (V*phi(u/V) for Gaussian noise; V^2 times the zero-bias density in
general). The engine records, per epoch, (1/N') sum_j g(u_j) with u_j measured against latent
quality ('pivot_bias', the prediction) and against the participant's own pre-epoch estimate
('pivot_audit', computable from logged scores, the rule and V alone).

Conditions: Gaussian noise at V in {0.05, 0.1, 0.2, 0.4}, and Laplace, uniform, Student-t(5)
at V = 0.2 (all with sd V). Mechanisms: percentile sortition over P, and UCB over c. Four pools.
200 seeds: P* / c* are picked on the first 100 (TUNE) and the cost of the M2 pick is scored on
the other 100 (EVAL), with random selection on EVAL as the reference. Checkpoints per condition.
"""
import numpy as np, sys, os, time, json, warnings
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore', category=RuntimeWarning)
from engine_v import run, SCENARIOS

TUNE = list(range(20000, 20100)); EVAL = list(range(20100, 20200)); SEEDS = TUNE + EVAL
OUT = 'pivotality_results.json'
CONDS = [('gauss', 0.05), ('gauss', 0.1), ('gauss', 0.2), ('gauss', 0.4),
         ('laplace', 0.2), ('uniform', 0.2), ('t5', 0.2)]
PGRID = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 85, 100]
CGRID = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
ARMS = [('static', dict(P_static=float(P)), P) for P in PGRID] + \
       [('ucb', dict(c_ucb=c), c) for c in CGRID]
FIELDS = ('M1', 'M2', 'M3', 'churn', 'pred', 'audit')

def _j(a):
    noise, V, pool, mode, kw, s = a
    st = {'pivot': True}
    r = run(**SCENARIOS[pool], mode=mode, V=V, noise=noise, seed=s, stats=st, **kw)
    if mode == 'random':
        return (float(np.nanmean(r[5])),)
    return (float(np.nanmean(r[5])), float(np.nanmean(r[0])), float(np.nanmean(r[3])),
            float(np.nanmean(r[4])), float(np.nanmean(st['pivot_bias'])),
            float(np.nanmean(st['pivot_audit'])))

def ckey(noise, V): return f"{noise}|{V:g}"

if __name__ == '__main__':
    t0 = time.time(); nw = int(os.environ.get('WORKERS', '4'))
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for noise, V in CONDS:
            ck = ckey(noise, V)
            if ck in out: print(f"{ck}: done, skipping"); continue
            tc = time.time(); D = {}
            for pool in SCENARIOS:
                for mode, kw, x in ARMS:
                    r = list(ex.map(_j, [(noise, V, pool, mode, kw, s) for s in SEEDS], chunksize=10))
                    D[f"{pool}|{mode}|{x:g}"] = np.array(r).T.tolist()
                r = list(ex.map(_j, [(noise, V, pool, 'random', {}, s) for s in EVAL], chunksize=10))
                D[f"{pool}|random"] = [x[0] for x in r]
            out[ck] = D
            json.dump(out, open(OUT, 'w'))
            print(f"{ck}  ({time.time()-tc:.0f}s)", flush=True)
    print(f"elapsed {time.time()-t0:.0f}s -> {OUT}", flush=True)
