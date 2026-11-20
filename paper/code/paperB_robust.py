"""
Deployment without knowing the drift rate (no simulation; reads paperB_baselines_results.json).

An operator must fix one configuration per family before seeing the pool or its drift.
Leave-one-cell-out: for each held-out cell, each family's configuration is the one with the
lowest mean regret on the other 11 cells; it is then scored on the held-out cell. Regret is
measured against the best configuration of ANY family in that cell, as a share of that cell's
static-over-random gap, so cells with different scales weigh equally.

All scores are on TUNE seeds (the only seeds every configuration was run on). The held-out
cell's seeds never influence the configuration it is scored with, so the comparison is not
selection-biased. Paired comparisons use the held-out cell's 50 seeds; Holm across cells.
Original grids only, since the edge extensions were run in some cells and not others.
"""
import json, sys, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
from stats_util import paired, tost_paired
import paperB_baselines as B
from paperB_final import holm

# --complete: the single final grid (paperB_complete.py), which the paper reports
if '--complete' in sys.argv:
    import paperB_complete as PC
    R=json.load(open(PC.OUT)); GRID=PC.GRID; PICKS_OUT='paperB_robust_picks_complete.json'
else:
    R=json.load(open(B.OUT)); GRID=B.GRID; PICKS_OUT='paperB_robust_picks.json'
CELLS=sorted(R,key=lambda c:(float(c.split('|')[1]),c))
FAMS=['static','ucb','ducb','swucb','ts']
M={c:{k:np.array(v) for k,v in R[c]['tune'].items()} for c in CELLS}
best={c:max(v.mean() for v in M[c].values()) for c in CELLS}
regret={c:{k:100*(best[c]-v.mean())/R[c]['gap'] for k,v in M[c].items()} for c in CELLS}

pick={}; rows=[]
for h in CELLS:
    others=[c for c in CELLS if c!=h]
    for f in FAMS:
        ks=[k for k in GRID if k.startswith(f+'|')]
        pick[(h,f)]=min(ks,key=lambda k:np.mean([regret[c][k] for c in others]))
labs=[('ducb','static'),('ts','static'),('swucb','static'),('ucb','static'),('ducb','ts')]
res={}
for h in CELLS:
    for a,b in labs:
        x,y=M[h][pick[(h,a)]],M[h][pick[(h,b)]]
        _,p,_=paired(x,y); tp,m,lo,hi=tost_paired(x,y,B.EPS)
        res[(h,a,b)]=dict(diff=m,ci=[lo,hi],p=p,tost=tp)
adj=dict(zip(res,holm(np.array([r['p'] for r in res.values()]))))

if __name__=='__main__':
    print("Picks (leave-one-cell-out; most common across the 12 folds):")
    for f in FAMS:
        from collections import Counter
        print(f"  {f:<7}", Counter(pick[(h,f)].split('|')[1] for h in CELLS).most_common(3))
    print("\nRegret on the held-out cell, % of its static-over-random gap (0 = best of anything):")
    print(f"{'cell':<16}"+"".join(f"{f:>9}" for f in FAMS))
    for h in CELLS:
        print(f"{h:<16}"+"".join(f"{regret[h][pick[(h,f)]]:>9.1f}" for f in FAMS))
    print(f"{'mean':<16}"+"".join(f"{np.mean([regret[h][pick[(h,f)]] for h in CELLS]):>9.1f}" for f in FAMS))
    print(f"{'worst':<16}"+"".join(f"{max(regret[h][pick[(h,f)]] for h in CELLS):>9.1f}" for f in FAMS))
    print("\nPaired on the held-out cell's 50 seeds, Holm across all:")
    for a,b in labs:
        v=[]
        for h in CELLS:
            r=res[(h,a,b)]; pa=adj[(h,a,b)]
            v.append('WIN' if pa<0.05 and r['diff']>0 else 'LOSS' if pa<0.05 else 'eq' if r['tost']<0.05 else '?')
        print(f"  {a} vs {b:<7} "+" ".join(f"{x:<4}" for x in v)+f"   W/L/eq/? = {v.count('WIN')}/{v.count('LOSS')}/{v.count('eq')}/{v.count('?')}")
    print("  (cell order: "+", ".join(CELLS)+")")
    json.dump({f"{h}|{f}":pick[(h,f)] for h in CELLS for f in FAMS},open(PICKS_OUT,'w'),indent=0)
