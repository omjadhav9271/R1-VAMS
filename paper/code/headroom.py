"""
Headroom bound on two-phase schedules for P:  P_t = p1 (t<t0), p2 (t>=t0).

The schedule grid CONTAINS every constant P (p1 == p2), so the oracle can never do worse
than the best constant on the seeds it is selected on. (The previous version's grid
excluded the constants, so its "oracle" lost to a constant in the Evolving pool and bounded
nothing.)

Two numbers per scenario, both on M1:
  oracle   -- best schedule minus best constant, both picked AND scored on TUNE seeds.
              Selection over many noisy candidates only inflates this, so it is an upper
              bound on what a schedule of this form can gain.
  held-out -- the same two picks re-scored on disjoint EVAL seeds (paired): an unbiased
              estimate of the gain actually achievable by tuning a schedule.
Scope: this bounds two-phase open-loop schedules on this grid -- not every adaptive rule.
"""
import numpy as np, sys, os, time, json, itertools
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from engine_v import run, SCENARIOS
from stats_util import paired, tost_paired

TUNE=list(range(9000,9050)); EVAL=list(range(9500,9600)); EPS=0.005
PS=[10,15,20,25,30,40,60]; T0=[50,150,400]
SCHED=[(p,p,0) for p in PS]+[(a,b,t) for a,b in itertools.permutations(PS,2) for t in T0]

def _j(a):
    name,sch,s=a
    kw=dict(mode='static',P_static=float(sch[0])) if sch[0]==sch[1] else dict(mode='sched',sched=sch)
    return float(np.nanmean(run(**SCENARIOS[name], seed=s, **kw)[5]))   # M1

if __name__=='__main__':
    t0_=time.time(); nw=int(os.environ.get('WORKERS','4'))
    # resumable: each finished pool is saved immediately and skipped on restart
    out=json.load(open('headroom_results.json')) if os.path.exists('headroom_results.json') else {}
    with ProcessPoolExecutor(max_workers=nw) as ex:
        def go(name,sch,seeds):
            return np.array(list(ex.map(_j,[(name,sch,s) for s in seeds],chunksize=10)))
        for name in SCENARIOS:
            if name in out: print(f"{name}: done, skipping"); continue
            print("="*96); print(f"HEADROOM — {name}: {len(SCHED)} schedules incl. {len(PS)} constants, M1")
            tune={s:go(name,s,TUNE) for s in SCHED}
            const=[s for s in SCHED if s[0]==s[1]]
            bC=max(const,key=lambda s:tune[s].mean()); bS=max(SCHED,key=lambda s:tune[s].mean())
            orc=tune[bS].mean()-tune[bC].mean()
            print("  constant P (tune): "+"  ".join(f"{s[0]}:{tune[s].mean():.4f}" for s in const))
            print(f"  best constant P={bC[0]}  best schedule {bS}   ORACLE gain (in-sample) {orc:+.5f}")
            print("  top 5: "+"  ".join(f"{s}:{tune[s].mean()-tune[bC].mean():+.5f}"
                                         for s in sorted(SCHED,key=lambda s:-tune[s].mean())[:5]))
            eS,eC=go(name,bS,EVAL),go(name,bC,EVAL)
            if bS==bC:
                ho=dict(diff=0.0,ci=[0.0,0.0],p=1.0,tost=0.0)
            else:
                _,p,dz=paired(eS,eC); tp,m,lo,hi=tost_paired(eS,eC,EPS)
                ho=dict(diff=m,ci=[lo,hi],p=p,tost=tp,dz=dz)
            print(f"  HELD-OUT gain {ho['diff']:+.5f} CI[{ho['ci'][0]:+.5f},{ho['ci'][1]:+.5f}] "
                  f"p={ho['p']:.4f} TOST={ho['tost']:.4f}")
            out[name]=dict(best_const=list(bC),best_sched=list(bS),oracle_gain=orc,heldout=ho,
                           tune={str(s):v.tolist() for s,v in tune.items()},
                           eval_sched=eS.tolist(),eval_const=eC.tolist())
            json.dump(out,open('headroom_results.json','w'))
    print(f"elapsed {time.time()-t0_:.0f}s   -> headroom_results.json")
