"""
Robustness of the VAMS null, on the unbiased metric M1 (the notebooks ran these sweeps on
the biased M2 with 8-10 seeds).

 (a) sigmoid gain k x window W. At k=20 a noise-level deviation of sigma_A (~0.02) moves P
     by only ~+-3 points, so the null could be an artefact of a timid gain. Large k turns
     the rule into bang-bang control between P_min and P_max.
 (b) the six response curves of VAMS_Final_Comparison.ipynb at their default parameters.

Each config is compared with static P=25 (paired, same seeds) and with static P at the
config's own realised mean P_t: if the second gap is ~0 while the first is not, the formula
changes outcomes only by shifting the average P, not by adapting.
"""
import numpy as np, sys, os, time, json
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from engine_v import run, SCENARIOS, VAMS_FORMS
from stats_util import paired, tost_paired

NS=100; SEEDS=list(range(13000,13000+NS)); EPS=0.005
KS=[5,20,50,100,200,1000]; WS=[10,50,200]
CONFIGS={f"sigmoid|k={k}|W={w}":dict(mode='adaptive',form='sigmoid',k=float(k),window=w)
         for k in KS for w in WS}
for f in VAMS_FORMS:
    if f!='sigmoid': CONFIGS[f"{f}|default|W=50"]=dict(mode='adaptive',form=f,window=50)

def _j(a):
    name,kw,s=a
    r=run(**SCENARIOS[name], seed=s, **kw)
    return float(np.nanmean(r[5])), float(np.nanmean(r[1])), float(np.nanstd(r[1]))

if __name__=='__main__':
    t0=time.time(); nw=int(os.environ.get('WORKERS','4'))
    # resumable: each finished pool is saved immediately and skipped on restart
    out=json.load(open('vams_sweep_results.json')) if os.path.exists('vams_sweep_results.json') else {}
    with ProcessPoolExecutor(max_workers=nw) as ex:
        def go(name,kw):
            return np.array(list(ex.map(_j,[(name,kw,s) for s in SEEDS],chunksize=8)))
        static={}
        def st(name,P):
            P=int(round(P))
            if (name,P) not in static: static[(name,P)]=go(name,dict(mode='static',P_static=float(P)))[:,0]
            return static[(name,P)]
        for name in SCENARIOS:
            if name in out: print(f"{name}: done, skipping"); continue
            print("="*112); print(f"VAMS ROBUSTNESS — {name}, {NS} paired seeds, M1, TOST eps={EPS}")
            print(f"{'config':<26} {'M1':>7} {'P mean':>7} {'P sd':>6} | {'vs P=25':>9} {'TOST':>7} | "
                  f"{'vs P=meanP':>10} {'TOST':>7}"); print("-"*112)
            base=st(name,25); res={}
            for lab,kw in CONFIGS.items():
                r=go(name,kw); q=r[:,0]; pm=r[:,1].mean(); ps=r[:,2].mean()
                tp1,m1,_,_=tost_paired(q,base,EPS); tp2,m2,_,_=tost_paired(q,st(name,pm),EPS)
                print(f"{lab:<26} {q.mean():>7.4f} {pm:>7.2f} {ps:>6.2f} | {m1:>+9.5f} {tp1:>7.4f} | "
                      f"{m2:>+10.5f} {tp2:>7.4f}")
                res[lab]=dict(m1=q.tolist(),P_mean=pm,P_sd=ps,vs25=[m1,tp1],vs_meanP=[int(round(pm)),m2,tp2])
            out[name]=res
            out.setdefault('static',{}).update({f"{k[0]}|{k[1]}":v.tolist() for k,v in static.items()})
            json.dump(out,open('vams_sweep_results.json','w'))
    print(f"elapsed {time.time()-t0:.0f}s   -> vams_sweep_results.json")
