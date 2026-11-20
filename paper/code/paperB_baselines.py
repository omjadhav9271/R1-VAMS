"""
Paper B main experiment: uncertainty-directed sortition against tuned static sortition AND
the standard non-stationary bandit baselines, every family tuned on held-out seeds, in every
pool x latent-drift cell.

Replaces memory_drift.py as evidence. That script fixed static P=25 and UCB c=0.2 across
drift levels and picked the best alpha_floor on the seeds it reported -- the selection bias
Paper A is about. Here, per cell:
  - every configuration of every family is run on TUNE seeds; each family's best is picked
  - the picks are re-run on disjoint EVAL seeds and compared paired
  - random sortition is run on EVAL seeds to express gains as a share of merit-over-random

Families (grid sizes in brackets; the baselines get grids about as large as UCB's):
  static   P                                     [9]   Kruijssen et al. percentile sortition
  ucb      c x alpha_floor (0 = running mean)     [35]  the proposed rule
  ucb_mean ucb restricted to alpha_floor=0        [7]   no forgetting (subset, no extra tune runs)
  ducb     c x gamma                              [36]  Discounted UCB
  swucb    c x window tau                         [30]  Sliding-window UCB
  ts       ts_scale x gamma (1 = no discount)     [30]  Gaussian Thompson sampling
Scored on M1. Checkpoints after every cell; rerun resumes.
"""
import numpy as np, sys, os, time, json, itertools
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from engine_v import run, SCENARIOS
from stats_util import paired, tost_paired

TUNE=list(range(13000,13050)); EVAL=list(range(14000,14100)); EPS=0.005
THETAS=[0.0, 0.02, 0.10]
OUT='paperB_baselines_results.json'

# Grids reach far toward fast turnover / fast forgetting: a 2-seed pilot at theta=0.1 put
# every family's pick on the edge of a narrower grid, which would favour the widest grid.
GRID={}
for P in [10,15,20,25,30,35,40,50,60]:
    GRID[f"static|P={P}"]=dict(mode='static',P_static=float(P))
for c,f in itertools.product([0.05,0.1,0.15,0.2,0.3,0.4,0.6],[0.0,0.02,0.05,0.1,0.2]):
    GRID[f"ucb|c={c},floor={f}"]=dict(mode='ucb',c_ucb=c,alpha_floor=f)
for c,g in itertools.product([0.05,0.1,0.2,0.3,0.4,0.6],[0.9,0.95,0.98,0.99,0.995,0.999]):
    GRID[f"ducb|c={c},gamma={g}"]=dict(mode='ducb',c_ucb=c,gamma=g)
for c,t in itertools.product([0.1,0.2,0.3,0.4,0.6],[10,25,50,100,200,400]):
    GRID[f"swucb|c={c},tau={t}"]=dict(mode='swucb',c_ucb=c,sw_tau=t)
for s,g in itertools.product([0.05,0.1,0.2,0.3,0.4],[0.95,0.98,0.99,0.995,0.999,1.0]):
    GRID[f"ts|scale={s},gamma={g}"]=dict(mode='ts',ts_scale=s,gamma=g)
FAMILIES={'static':  lambda k:k.startswith('static|'),
          'ucb':     lambda k:k.startswith('ucb|'),
          'ucb_mean':lambda k:k.startswith('ucb|') and k.endswith('floor=0.0'),
          'ducb':    lambda k:k.startswith('ducb|'),
          'swucb':   lambda k:k.startswith('swucb|'),
          'ts':      lambda k:k.startswith('ts|')}

def _j(a):
    name,th,kw,s=a
    return float(np.nanmean(run(**SCENARIOS[name], ou_theta=th, seed=s, **kw)[5]))   # M1

def on_edge(key,fam):
    """True if the pick sits at the boundary of its family's grid in any parameter that
    the family actually varies (ucb_mean's fixed floor=0 does not count)."""
    params=key.split('|')[1]; ks=[k for k in GRID if FAMILIES[fam](k)]
    for i,(pn,pv) in enumerate(p.split('=') for p in params.split(',')):
        vals=sorted({float(k.split('|')[1].split(',')[i].split('=')[1]) for k in ks})
        # floor=0 and gamma=1 mean 'no forgetting': natural limits, not grid truncation
        edges={vals[0],vals[-1]}-{v for n,v in (('floor',0.0),('gamma',1.0)) if pn==n}
        if len(vals)>1 and float(pv) in edges: return True
    return False

def cmp(a,b):
    _,p,dz=paired(a,b); tp,m,lo,hi=tost_paired(a,b,EPS)
    return dict(diff=m,ci=[lo,hi],p=p,tost=tp,dz=dz)

if __name__=='__main__':
    t0=time.time(); nw=int(os.environ.get('WORKERS','4'))
    out=json.load(open(OUT)) if os.path.exists(OUT) else {}
    print(f"{len(GRID)} configs; tune n={len(TUNE)}, eval n={len(EVAL)}; cells={len(SCENARIOS)*len(THETAS)}")
    with ProcessPoolExecutor(max_workers=nw) as ex:
        def go(name,th,kw,seeds):
            return np.array(list(ex.map(_j,[(name,th,kw,s) for s in seeds],chunksize=10)))
        for th in THETAS:
            for name in SCENARIOS:
                cell=f"{name}|{th}"
                if cell in out: print(f"{cell}: done, skipping"); continue
                tc=time.time()
                tune={k:go(name,th,kw,TUNE) for k,kw in GRID.items()}
                best={f:max((k for k in GRID if sel(k)),key=lambda k:tune[k].mean())
                      for f,sel in FAMILIES.items()}
                ev={k:go(name,th,GRID[k],EVAL) for k in set(best.values())}
                rnd=go(name,th,dict(mode='random'),EVAL)
                S=ev[best['static']]; gap=S.mean()-rnd.mean()
                C={f:cmp(ev[best[f]],S) for f in FAMILIES if f!='static'}
                C.update({f"ucb vs {f}":cmp(ev[best['ucb']],ev[best[f]]) for f in ('ducb','swucb','ts')})
                print("="*100); print(f"{cell}   static-over-random gap {gap:.4f}   ({time.time()-tc:.0f}s)")
                for f in FAMILIES:
                    k=best[f]; print(f"  {f:<9} pick {k.split('|')[1]:<22}{' [EDGE]' if on_edge(k,f) else '       '}"
                                     f" tune {tune[k].mean():.4f}  eval {ev[k].mean():.4f}")
                for lab,r in C.items():
                    lab=lab if lab.startswith('ucb vs') else f"{lab} vs static"
                    print(f"  {lab:<18} diff={r['diff']:+.5f} CI[{r['ci'][0]:+.5f},{r['ci'][1]:+.5f}] "
                          f"({100*r['diff']/gap:+5.1f}% of gap) dz={r['dz']:+.2f} p={r['p']:.3g} TOST={r['tost']:.3f}")
                out[cell]=dict(best=best,edge={f:on_edge(k,f) for f,k in best.items()},gap=gap,
                               comparisons=C,tune={k:v.tolist() for k,v in tune.items()},
                               eval={k:v.tolist() for k,v in ev.items()},random=rnd.tolist())
                json.dump(out,open(OUT,'w'))
    print(f"elapsed {time.time()-t0:.0f}s   -> {OUT}")
