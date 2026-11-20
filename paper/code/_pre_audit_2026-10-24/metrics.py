"""
Three metrics for the same simulation, swept over P:

 M1 obs_A  = mean(T_act): quality delivered by the set active DURING the epoch, measured
             before selection consumes it. Unbiased. Matches Kruijssen et al. sec 3.4.
 M2 mean_T = the notebooks' metric: post-selection set, reusing the selection-driving T
             for retained participants and a synthetic fresh draw for promoted ones.
 M3 lat_Q  = mean latent Q of the post-selection set. Bias-free ground truth.

If M2 is the artefact, M1 and M3 should peak at low P (recovering the paper's 20-40%)
while only M2 climbs with P.
"""
import numpy as np, sys, os, time, json
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from engine_v import run, SCENARIOS
from stats_util import welch

NS=50; SEEDS=list(range(7000,7000+NS))
PGRID=[1,3,5,10,15,20,25,30,40,50,60,70,85,100]

def _j(a):
    name,P,s=a
    q,Ph,S,L,C,OA = run(**SCENARIOS[name], mode='static', P_static=float(P), seed=s)
    return (float(np.nanmean(OA)), float(np.nanmean(q)), float(np.nanmean(L)), float(np.nanmean(C)))

if __name__=='__main__':
    t0=time.time(); nw=int(os.environ.get('WORKERS','4')); res={}
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for n in SCENARIOS:
            for P in PGRID:
                r=list(ex.map(_j,[(n,P,s) for s in SEEDS],chunksize=8))
                res[(n,P)]=tuple(np.array([x[i] for x in r]) for i in range(4))
    print("="*112)
    print(f"THREE METRICS vs P  ({NS} seeds).  M1=mean(T_act) unbiased | M2=notebooks' mean_T | M3=latent | ch=churn")
    print("="*112)
    for n in SCENARIOS:
        print(f"\n--- {n} Pool ---")
        print(f"{'P%':>5} {'M1 (unbiased)':>15} {'M2 (notebooks)':>16} {'M3 (latent)':>13} {'churn':>8}")
        for P in PGRID:
            m1,m2,m3,c=res[(n,P)]
            print(f"{P:>5} {m1.mean():>15.4f} {m2.mean():>16.4f} {m3.mean():>13.4f} {c.mean():>8.3f}")
        b1=max(PGRID,key=lambda P:res[(n,P)][0].mean())
        b2=max(PGRID,key=lambda P:res[(n,P)][1].mean())
        b3=max(PGRID,key=lambda P:res[(n,P)][2].mean())
        print(f"  argmax ->  M1: P={b1}%   M2: P={b2}%   M3: P={b3}%    "
              f"[paper reports 20-40%, P=25% recommended]")
    json.dump({f"{k[0]}|{k[1]}":[v[i].tolist() for i in range(4)] for k,v in res.items()},
              open('metrics_results.json','w'))
    print(f"\nelapsed {time.time()-t0:.0f}s")
