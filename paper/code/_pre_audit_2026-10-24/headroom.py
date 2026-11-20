"""
Headroom bound on time-varying P.

Searches two-phase schedules P_t = p1 (t<t0), p2 (t>=t0) with an ORACLE -- the best
schedule is picked using the same data it is evaluated on, so this OVERSTATES what any
real adaptive rule could achieve. Scored on the unbiased metric M1.

If best_schedule - best_constant is small, then no causal, online rule for P can help,
which bounds the whole family VAMS belongs to instead of refuting one rule at a time.
"""
import numpy as np, sys, os, time, json, itertools
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from engine_v import run, SCENARIOS
from stats_util import welch

NS=50; SEEDS=list(range(9000,9000+NS))
SCN=['Fixed','Evolving']
PCONST=[5,10,15,20,25,30,40]
P1=[20,50,80]; P2=[5,15,25]; T0=[100,250]

def _j(a):
    name,kw,s=a
    r=run(**SCENARIOS[name], seed=s, **kw)
    return float(np.nanmean(r[5]))   # M1 = obs_A, unbiased

if __name__=='__main__':
    t0_=time.time(); nw=int(os.environ.get('WORKERS','4')); out={}
    with ProcessPoolExecutor(max_workers=nw) as ex:
        def go(name,kw):
            return np.array(list(ex.map(_j,[(name,kw,s) for s in SEEDS],chunksize=8)))
        for name in SCN:
            print("="*96); print(f"HEADROOM FOR TIME-VARYING P — {name} Pool, {NS} seeds, unbiased metric M1")
            print("="*96)
            const={P:go(name,dict(mode='static',P_static=float(P))) for P in PCONST}
            bP=max(PCONST,key=lambda P:const[P].mean())
            print("  constant P:  " + "  ".join(f"P={P}:{const[P].mean():.4f}" for P in PCONST))
            print(f"  best constant: P={bP}%  q={const[bP].mean():.4f}")
            sch={}
            for p1,p2,tt in itertools.product(P1,P2,T0):
                sch[(p1,p2,tt)]=go(name,dict(mode='sched',sched=(p1,p2,tt)))
            bS=max(sch,key=lambda k:sch[k].mean())
            _,p,d=welch(sch[bS],const[bP])
            print(f"  best 2-phase schedule: p1={bS[0]}% -> p2={bS[1]}% at t={bS[2]}   q={sch[bS].mean():.4f}")
            print(f"  ORACLE headroom over best constant: {sch[bS].mean()-const[bP].mean():+.5f}  "
                  f"d={d:+.3f}  p={p:.4f}")
            print("  top 5 schedules:")
            for k in sorted(sch,key=lambda k:-sch[k].mean())[:5]:
                print(f"    p1={k[0]:>3} p2={k[1]:>3} t0={k[2]:>4}  q={sch[k].mean():.4f} "
                      f"({sch[k].mean()-const[bP].mean():+.5f} vs best const)")
            out[name]=dict(best_const=[bP,const[bP].mean()],
                           best_sched=[list(bS),sch[bS].mean()],
                           const={str(k):v.tolist() for k,v in const.items()},
                           sched={str(k):v.tolist() for k,v in sch.items()})
            print()
    json.dump(out,open('headroom_results.json','w'))
    print(f"elapsed {time.time()-t0_:.0f}s   -> headroom_results.json")
