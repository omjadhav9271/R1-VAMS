"""
Estimator memory x latent drift.

alpha_floor is a single knob spanning the two estimator designs:
    0.00 -> convergent sample mean (what UCB assumes; infinite memory)
    0.10 -> fixed-alpha EMA, i.e. Kruijssen et al.'s estimator (memory ~ 1/0.1 = 10)
If the EMA's forgetting is ever worth its variance cost, a larger alpha_floor must win
at a high enough drift. This maps that surface. Static P=25%+EMA is shown for reference.
"""
import numpy as np, sys, os, time, json
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from engine_v import run, SCENARIOS
from stats_util import welch

NS=100; SEEDS=list(range(3000,3000+NS))
SC=dict(SCENARIOS['Evolving'])
THETAS=[0.0, 0.005, 0.02, 0.05, 0.10, 0.20]
FLOORS=[0.0, 0.02, 0.05, 0.10, 0.20]

def _job(a):
    kind, th, fl, s = a
    if kind=='static':
        kw=dict(mode='static', P_static=25.0)
    else:
        kw=dict(mode='ucb', c_ucb=0.3, alpha_floor=fl)
    return float(np.nanmean(run(**SC, ou_theta=th, seed=s, **kw)[0]))

if __name__=='__main__':
    t0=time.time(); nw=max(1,(os.cpu_count() or 4)-1)
    with ProcessPoolExecutor(max_workers=nw) as ex:
        res={}
        for th in THETAS:
            res[('static',th)]=np.array(list(ex.map(_job,[('static',th,0.0,s) for s in SEEDS],chunksize=2)))
            for fl in FLOORS:
                res[(fl,th)]=np.array(list(ex.map(_job,[('ucb',th,fl,s) for s in SEEDS],chunksize=2)))
    print("="*104)
    print(f"ESTIMATOR MEMORY x DRIFT — Evolving Pool, {NS} seeds, UCB c=0.3")
    print("  cell = mean quality; (d) vs the static-P+EMA baseline in the same drift column")
    print("="*104)
    hdr="".join(f"{('th='+format(t,'.3f')):>16}" for t in THETAS)
    print(f"{'alpha_floor':>12}{hdr}")
    print(f"{'innov sd':>12}"+"".join(f"{(0.1*np.sqrt(max(2*t-t*t,0))):>16.4f}" for t in THETAS))
    print("-"*104)
    print(f"{'staticP+EMA':>12}"+"".join(f"{res[('static',t)].mean():>16.4f}" for t in THETAS))
    print("-"*104)
    best={}
    for fl in FLOORS:
        row=f"{fl:>12.2f}"
        for th in THETAS:
            m=res[(fl,th)].mean()
            _,p,d=welch(res[(fl,th)],res[('static',th)])
            row+=f"{m:>10.4f}({d:>+.2f})"
            best.setdefault(th,[]).append((m,fl))
        print(row)
    print("-"*104)
    print("best alpha_floor per drift level:")
    for th in THETAS:
        m,fl=max(best[th])
        st=res[('static',th)].mean()
        _,p,d=welch(res[(fl,th)],res[('static',th)])
        print(f"  theta={th:<6} innov_sd={0.1*np.sqrt(max(2*th-th*th,0)):.4f}  best floor={fl:<5} "
              f"q={m:.4f}  vs staticP+EMA {st:.4f}  d={d:+.3f} p={p:.5f}")
    json.dump({f"{k[0]}|{k[1]}":v.tolist() for k,v in res.items()}, open('memory_drift.json','w'))
    print(f"elapsed {time.time()-t0:.0f}s")
