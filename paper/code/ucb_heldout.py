"""
UCB vs a tuned static baseline, with HELD-OUT tuning and a 2x2 ablation.
Replaces decisive.py / decisive2.py, which picked best-P and best-c on the very seeds they
were scored on -- the selection-conditioned bias this paper is about.

Protocol
  - every hyperparameter (static P, UCB c) is chosen on TUNE seeds
  - the chosen configurations are compared on disjoint EVAL seeds, paired
  - the whole grid is also run on EVAL seeds, for the sensitivity figure and to measure how
    much in-sample selection would have inflated the result

Ablation: UCB differs from static sortition in two ways at once -- the estimator
(running mean, no imputation for inactive participants) and the exploration bonus.
    estimator / bonus |  none          UCB
    running mean      |  greedy        ucb
    EMA (alpha=0.1)   |  greedy_ema    ucb_ema      (alpha_floor=0.1, updated only when active)
'greedy' still promotes never-observed participants first (n_j=0 -> +inf); that is the
only exploration it has.
Scored on M1 throughout.
"""
import numpy as np, sys, os, time, json
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from engine_v import run, SCENARIOS
from stats_util import paired, tost_paired, welch

TUNE=list(range(11000,11060)); EVAL=list(range(12000,12120)); EPS=0.005
STATICS=[10,15,20,25,30,35,40]
CS=[0.02,0.05,0.1,0.15,0.2,0.3,0.4]
CONFIGS={}
for P in STATICS: CONFIGS[f"static|{P}"]=dict(mode='static',P_static=float(P))
for c in CS:
    CONFIGS[f"ucb|{c}"]=dict(mode='ucb',c_ucb=c)
    CONFIGS[f"ucb_ema|{c}"]=dict(mode='ucb',c_ucb=c,alpha_floor=0.1)
CONFIGS["greedy|-"]=dict(mode='greedy')
CONFIGS["greedy_ema|-"]=dict(mode='greedy',alpha_floor=0.1)
FAMILIES=['static','ucb','ucb_ema','greedy','greedy_ema']

def _j(a):
    name,kw,s=a
    return float(np.nanmean(run(**SCENARIOS[name], seed=s, **kw)[5]))   # M1

def cmp(a,b):
    _,p,dz=paired(a,b); tp,m,lo,hi=tost_paired(a,b,EPS); _,_,dw=welch(a,b)
    v=("A WINS" if (p<0.05 and m>0) else "B wins" if (p<0.05 and m<0) else
       "EQUIVALENT" if tp<0.05 else "inconclusive")
    return dict(diff=m,ci=[lo,hi],p=p,tost=tp,dz=dz,welch_d=dw,verdict=v)

if __name__=='__main__':
    t0=time.time(); nw=int(os.environ.get('WORKERS','4')); out={}
    with ProcessPoolExecutor(max_workers=nw) as ex:
        def go(name,kw,seeds):
            return np.array(list(ex.map(_j,[(name,kw,s) for s in seeds],chunksize=8)))
        print("="*104)
        print(f"UCB vs tuned static, M1. tune n={len(TUNE)}, eval n={len(EVAL)} (disjoint, paired)")
        print("="*104)
        for name in SCENARIOS:
            tune={k:go(name,kw,TUNE) for k,kw in CONFIGS.items()}
            ev  ={k:go(name,kw,EVAL) for k,kw in CONFIGS.items()}
            best={f:max((k for k in CONFIGS if k.split('|')[0]==f),key=lambda k:tune[k].mean())
                  for f in FAMILIES}
            print(f"\n--- {name} ---")
            for f in FAMILIES:
                ks=[k for k in CONFIGS if k.split('|')[0]==f]
                print(f"  {f:<11} tune: "+"  ".join(f"{k.split('|')[1]}:{tune[k].mean():.4f}" for k in ks))
                print(f"  {'':<11} eval: "+"  ".join(f"{k.split('|')[1]}:{ev[k].mean():.4f}" for k in ks))
            print("  tuned on TUNE seeds: "+"  ".join(f"{f}={best[f].split('|')[1]}" for f in FAMILIES))
            S=ev[best['static']]
            C={'ucb vs static':        cmp(ev[best['ucb']],S),
               'greedy vs static':     cmp(ev[best['greedy']],S),
               'ucb vs greedy (bonus, mean est.)': cmp(ev[best['ucb']],ev[best['greedy']]),
               'ucb_ema vs greedy_ema (bonus, EMA est.)': cmp(ev[best['ucb_ema']],ev[best['greedy_ema']]),
               'ucb vs ucb_ema (estimator, with bonus)':  cmp(ev[best['ucb']],ev[best['ucb_ema']]),
               'greedy vs greedy_ema (estimator, no bonus)': cmp(ev[best['greedy']],ev[best['greedy_ema']]),
               'ucb_ema vs static':    cmp(ev[best['ucb_ema']],S)}
            for lab,r in C.items():
                print(f"  {lab:<44} diff={r['diff']:+.5f} CI[{r['ci'][0]:+.5f},{r['ci'][1]:+.5f}] "
                      f"dz={r['dz']:+.2f} p={r['p']:.4g} TOST={r['tost']:.4f}  {r['verdict']}")
            # how much would choosing on the eval seeds have inflated the UCB-vs-static gap?
            ins=max(ev[k].mean() for k in CONFIGS if k.startswith('ucb|'))-\
                max(ev[k].mean() for k in CONFIGS if k.startswith('static|'))
            print(f"  in-sample-selected gap on EVAL (biased): {ins:+.5f}  vs held-out: {C['ucb vs static']['diff']:+.5f}")
            out[name]=dict(best=best,comparisons=C,insample_gap=ins,
                           tune={k:v.tolist() for k,v in tune.items()},
                           eval={k:v.tolist() for k,v in ev.items()})
    json.dump(out,open('ucb_heldout_results.json','w'))
    print(f"\nelapsed {time.time()-t0:.0f}s   -> ucb_heldout_results.json")
