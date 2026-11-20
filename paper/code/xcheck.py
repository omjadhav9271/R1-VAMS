"""
Engine cross-check: vectorized engine_v vs ref_engine (verbatim notebook engine).

The two consume randomness in a different order (tie-breaking, per-participant draws), so
they cannot agree bit-for-bit; they must agree in distribution. "Agree" is therefore a
TOST equivalence verdict at the paper's SESOI, on BOTH metrics -- not a failure to reject.
"""
import numpy as np, sys, os, time, json
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from engine_v import run as run_v, SCENARIOS
from ref_engine import run_sortition as run_o
from stats_util import welch, tost

NS=100; SEEDS=list(range(1, 1+NS)); EPS=0.005
MODES=[('static', dict(P_static=25.0)), ('adaptive', dict(k=20.0, window=50))]

def _j(a):
    eng,name,mode,kw,s=a
    if eng=='o':
        r=run_o(**SCENARIOS[name], mode=mode, seed=s, **kw); m2,m1=r[0],r[3]
    else:
        r=run_v(**SCENARIOS[name], mode=mode, seed=s, **kw); m2,m1=r[0],r[5]
    return float(np.nanmean(m1)), float(np.nanmean(m2))

if __name__=='__main__':
    t0=time.time(); nw=int(os.environ.get('WORKERS','4')); out={}
    print("="*104)
    print(f"ENGINE CROSS-CHECK ({NS} seeds; notebook engine vs vectorized; TOST eps={EPS})")
    print("="*104)
    print(f"{'Scenario':<10} {'mode':<9} {'metric':<4} {'notebook':>9} {'vector':>9} {'diff':>9} "
          f"{'95% CI':>22} {'Welch p':>8} {'TOST p':>8} {'verdict':>8}")
    print("-"*104)
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for name in SCENARIOS:
            for mode,kw in MODES:
                o=np.array(list(ex.map(_j,[('o',name,mode,kw,s) for s in SEEDS],chunksize=4)))
                v=np.array(list(ex.map(_j,[('v',name,mode,kw,s) for s in SEEDS],chunksize=4)))
                for mi,mn in enumerate(['M1','M2']):
                    a,b=v[:,mi],o[:,mi]
                    _,p,_=welch(a,b); tp,diff,lo,hi=tost(a,b,EPS)
                    ok='EQUIV' if tp<0.05 else 'FAIL'
                    print(f"{name:<10} {mode:<9} {mn:<4} {b.mean():>9.4f} {a.mean():>9.4f} {diff:>+9.5f} "
                          f"[{lo:+.5f},{hi:+.5f}] {p:>8.3f} {tp:>8.4f} {ok:>8}")
                    out[f"{name}|{mode}|{mn}"]=dict(notebook=b.tolist(),vector=a.tolist(),
                                                     tost=tp,welch_p=p,ci=[lo,hi])
    json.dump(out,open('xcheck_results.json','w'))
    print(f"elapsed {time.time()-t0:.0f}s   -> xcheck_results.json")
