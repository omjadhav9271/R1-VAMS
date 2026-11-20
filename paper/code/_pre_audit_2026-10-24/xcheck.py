"""Statistical equivalence check: vectorized engine vs original notebook engine."""
import numpy as np, sys, time
from scipy import stats
sys.path.insert(0,'.')
sys.stdout.reconfigure(encoding='utf-8')
from engine_v import run as run_v, SCENARIOS
from ucb import run as run_o          # original-structure engine

SEEDS = list(range(1, 25))
print("="*88)
print("ENGINE CROSS-CHECK  (24 seeds; original loop-based vs vectorized)")
print("="*88)
print(f"{'Scenario':<11} {'mode':<9} {'original':>18} {'vectorized':>18} {'Welch p':>9} {'verdict':>9}")
print("-"*88)
t0=time.time()
for name, sc in SCENARIOS.items():
    for mode, kw in [('static', dict(P_static=25.0)), ('adaptive', dict(k=20.0, window=50))]:
        o = np.array([np.nanmean(run_o(**sc, mode=mode, seed=s, **kw)[0]) for s in SEEDS])
        v = np.array([np.nanmean(run_v(**sc, mode=mode, seed=s, **kw)[0]) for s in SEEDS])
        t, p = stats.ttest_ind(o, v, equal_var=False)
        ok = 'agree' if p > 0.05 else 'DIFFER'
        print(f"{name:<11} {mode:<9} {o.mean():>10.4f}±{o.std():<7.4f} {v.mean():>10.4f}±{v.std():<7.4f} {p:>9.4f} {ok:>9}")
print("-"*88)
print(f"elapsed {time.time()-t0:.0f}s")
