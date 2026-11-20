"""
Decisive test: does uncertainty-directed selection (UCB) beat a CORRECTLY TUNED static
baseline on the CORRECT (unbiased) metric?

Both earlier verdicts on UCB were wrong for the same reason: the baseline was tuned on
the biased metric M2, which puts P* at 65-95%. On M1 the optimum is P ~ 20-30%. This
run sweeps static P finely around that optimum and UCB over c, all on identical seeds,
and compares the best of each with a PAIRED test (same seeds -> far higher power).
"""
import numpy as np, sys, os, time, json
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from engine_v import run, SCENARIOS
from stats_util import paired, tost_paired, welch

NS=120; SEEDS=list(range(11000,11000+NS)); EPS=0.005
STATICS=[20,25,30]
CS=[0.02,0.05,0.1,0.15,0.2,0.3]   # extended down: c=0.2 was at the grid edge

def _j(a):
    name,kw,s=a
    return float(np.nanmean(run(**SCENARIOS[name], seed=s, **kw)[5]))   # M1

if __name__=='__main__':
    t0=time.time(); nw=int(os.environ.get('WORKERS','4')); out={}
    with ProcessPoolExecutor(max_workers=nw) as ex:
        def go(name,kw):
            return np.array(list(ex.map(_j,[(name,kw,s) for s in SEEDS],chunksize=8)))
        print("="*100)
        print(f"DECISIVE: UCB vs best-tuned static P, unbiased metric M1, {NS} paired seeds")
        print("="*100)
        for name in SCENARIOS:
            st={P:go(name,dict(mode='static',P_static=float(P))) for P in STATICS}
            uc={c:go(name,dict(mode='ucb',c_ucb=c)) for c in CS}
            bP=max(STATICS,key=lambda P:st[P].mean()); bC=max(CS,key=lambda c:uc[c].mean())
            print(f"\n--- {name} ---")
            print("  static: "+"  ".join(f"P={P}:{st[P].mean():.4f}" for P in STATICS))
            print("  ucb   : "+"  ".join(f"c={c}:{uc[c].mean():.4f}" for c in CS))
            a,b=uc[bC],st[bP]
            t,p,dz=paired(a,b); tp,m,lo,hi=tost_paired(a,b,EPS)
            _,pw,dw=welch(a,b)
            verdict=("UCB WINS" if (p<0.05 and m>0) else
                     "static wins" if (p<0.05 and m<0) else
                     "EQUIVALENT" if tp<0.05 else "inconclusive")
            print(f"  best static P={bP}% -> {b.mean():.4f}   best UCB c={bC} -> {a.mean():.4f}")
            print(f"  paired diff = {m:+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]  d_z={dz:+.3f}  "
                  f"p_paired={p:.5f}  (welch d={dw:+.3f} p={pw:.5f})")
            print(f"  TOST p={tp:.5f}   VERDICT: {verdict}")
            out[name]=dict(bestP=bP,bestC=bC,static=b.tolist(),ucb=a.tolist(),
                           diff=m,ci=[lo,hi],p_paired=p,tost=tp,verdict=verdict)
    json.dump(out,open('decisive2_results.json','w'))
    print(f"\nelapsed {time.time()-t0:.0f}s   -> decisive_results.json")
