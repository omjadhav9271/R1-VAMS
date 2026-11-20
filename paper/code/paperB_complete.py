"""
Paper B, final grid: one complete grid per family, run in every pool x drift cell.

paperB_baselines.py plus two edge-extension rounds (paperB_edge.py) left each cell searched
on a different grid, and the leave-one-cell-out analysis could only use the original grid,
which truncates Thompson sampling at gamma=0.95. This script runs the UNION of every value
either round needed, in all 12 cells, so every Paper B number comes from one grid.

Configurations already run by paperB_baselines.py are reused (same TUNE seeds, same engine);
only the new ones are simulated. Each family's pick is re-run on EVAL seeds unless that exact
configuration's EVAL scores already exist. Output has the same schema as
paperB_baselines_results.json, so paperB_final.py / paperB_robust.py can read it.
Checkpoints per cell. A rerun after GRID has grown fills in only the missing configurations of
a finished cell (from its own checkpoint) and re-picks; nothing already run is repeated.

SW-UCB includes tau=1000=Nepochs: a window that never drops an observation, and whose bonus
log(min(t,tau)) is then log(t) throughout -- the 'no forgetting' limit, like gamma=1 and
floor=0, so a pick there is not grid truncation. Added 2026-11-20 after tau=800 (the old
largest window) was the edge pick in Fixed|0.0 and Growing|0.0. Thompson scale extended
down to 0 the same day, after scale=0.02 (the smallest) was the pick in Fixed|0.02; SW-UCB c
down to 0 and TS gamma down to 0.5 after Fixed|0.1 picked c=0.02 and gamma=0.7; UCB floor up
to 1 and D-UCB gamma down to 0.6 after Growing|0.1 picked floor=0.3 and gamma=0.8.
"""
import numpy as np, sys, os, time, json, itertools
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from engine_v import run, SCENARIOS
from stats_util import paired, tost_paired
import paperB_baselines as B

OUT='paperB_complete_results.json'
GRID=dict(B.GRID)
def add(fam,kwfn,keyfn,*axes):
    for v in itertools.product(*axes):
        GRID.setdefault(keyfn(*v),kwfn(*v))
add('static',lambda P:dict(mode='static',P_static=float(P)),lambda P:f"static|P={P}",
    [10,15,20,25,30,35,40,50,60,70,80])
add('ucb',lambda c,f:dict(mode='ucb',c_ucb=c,alpha_floor=f),lambda c,f:f"ucb|c={c},floor={f}",
    [0.05,0.1,0.15,0.2,0.3,0.4,0.6,0.8,1.0,1.5,2.0,3.0],[0.0,0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.7,1.0])
add('ducb',lambda c,g:dict(mode='ducb',c_ucb=c,gamma=g),lambda c,g:f"ducb|c={c},gamma={g}",
    [0.01,0.02,0.03,0.05,0.1,0.2,0.3,0.4,0.6],[0.6,0.7,0.75,0.8,0.85,0.9,0.95,0.98,0.99,0.995,0.999,1.0])
add('swucb',lambda c,t:dict(mode='swucb',c_ucb=c,sw_tau=t),lambda c,t:f"swucb|c={c},tau={t}",
    [0.0,0.005,0.01,0.02,0.03,0.05,0.1,0.2,0.3,0.4,0.6],[3,5,10,25,50,100,200,400,800,1000])
add('ts',lambda s,g:dict(mode='ts',ts_scale=s,gamma=g),lambda s,g:f"ts|scale={s},gamma={g}",
    [0.0,0.005,0.01,0.02,0.03,0.05,0.1,0.2,0.3,0.4],[0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.98,0.99,0.995,0.999,1.0])
FAMILIES=B.FAMILIES
# natural limits, not truncation: no forgetting (floor=0, gamma=1, tau=Nepochs), total
# forgetting (floor=1: the estimate is the last observation) and no
# exploration (TS scale=0 / SW-UCB c=0: greedy on the discounted or windowed mean, unseen
# identities still first)
NATURAL=(('floor',0.0),('floor',1.0),('gamma',1.0),('tau',1000.0),('scale',0.0),('c',0.0))

def on_edge(key,fam):
    """Pick on the boundary of its family's grid in a parameter the family varies
    (floor=0 and gamma=1 are natural limits, not truncation)."""
    params=key.split('|')[1]; ks=[k for k in GRID if FAMILIES[fam](k)]
    for i,(pn,pv) in enumerate(p.split('=') for p in params.split(',')):
        vals=sorted({float(k.split('|')[1].split(',')[i].split('=')[1]) for k in ks})
        edges={vals[0],vals[-1]}-{v for n,v in NATURAL if pn==n}
        if len(vals)>1 and float(pv) in edges: return True
    return False

def cmp(a,b):
    _,p,dz=paired(a,b); tp,m,lo,hi=tost_paired(a,b,B.EPS)
    return dict(diff=m,ci=[lo,hi],p=p,tost=tp,dz=dz)

if __name__=='__main__':
    t0=time.time(); nw=int(os.environ.get('WORKERS','4'))
    old=json.load(open(B.OUT)); out=json.load(open(OUT)) if os.path.exists(OUT) else {}
    new_keys=[k for k in GRID if k not in B.GRID]
    print(f"{len(GRID)} configs ({len(new_keys)} new) x {len(B.TUNE)} tune seeds x {len(old)} cells")
    with ProcessPoolExecutor(max_workers=nw) as ex:
        def go(name,th,kw,seeds):
            return np.array(list(ex.map(B._j,[(name,th,kw,s) for s in seeds],chunksize=10)))
        for th in B.THETAS:
            for name in SCENARIOS:
                cell=f"{name}|{th}"
                d=out.get(cell) or old[cell]
                missing=[k for k in GRID if k not in d['tune']]
                if cell in out and not missing: print(f"{cell}: done, skipping"); continue
                tc=time.time()
                tune={k:np.array(v) for k,v in d['tune'].items()}
                for k in missing: tune[k]=go(name,th,GRID[k],B.TUNE)
                best={f:max((k for k in GRID if sel(k)),key=lambda k:tune[k].mean())
                      for f,sel in FAMILIES.items()}
                ev={k:np.array(v) for k,v in d['eval'].items()}
                for k in set(best.values())-set(ev): ev[k]=go(name,th,GRID[k],B.EVAL)
                rnd=np.array(d['random']); S=ev[best['static']]; gap=S.mean()-rnd.mean()
                C={f:cmp(ev[best[f]],S) for f in FAMILIES if f!='static'}
                C.update({f"ucb vs {f}":cmp(ev[best['ucb']],ev[best[f]]) for f in ('ducb','swucb','ts')})
                C.update({f"ducb vs {f}":cmp(ev[best['ducb']],ev[best[f]]) for f in ('swucb','ts')})
                edge={f:on_edge(k,f) for f,k in best.items()}
                print(f"{cell}  ({time.time()-tc:.0f}s)  "+"  ".join(
                    f"{f}={best[f].split('|')[1]}{'[EDGE]' if edge[f] else ''}" for f in FAMILIES), flush=True)
                out[cell]=dict(best=best,edge=edge,gap=gap,comparisons=C,
                               tune={k:v.tolist() for k,v in tune.items()},
                               eval={k:v.tolist() for k,v in ev.items()},random=rnd.tolist())
                json.dump(out,open(OUT,'w'))
    print(f"elapsed {time.time()-t0:.0f}s   -> {OUT}", flush=True)
