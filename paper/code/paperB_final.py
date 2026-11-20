"""
Final Paper B table: merge paperB_baselines_results.json with paperB_edge_results.json
(no simulation). Each family's pick is the edge follow-up's pick where one ran, else the
original. All comparisons are recomputed on the EVAL seeds, paired, and Holm-adjusted
together. 'edge' marks a final pick still on the boundary of its EXTENDED grid (natural
limits floor=0, gamma=1, tau=1000=Nepochs excluded), i.e. the extension list ran out.
"""
import json, sys, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
from stats_util import paired, tost_paired
import paperB_baselines as B, paperB_edge as E

def holm(ps):
    o=np.argsort(ps); m=len(ps); adj=np.empty(m); run=0.0
    for r,i in enumerate(o):
        run=max(run,min(1.0,(m-r)*ps[i])); adj[i]=run
    return adj

# --complete: read the single-grid run (paperB_complete.py) instead of merging the main run
# with the edge rounds. That is the version the paper reports.
COMPLETE='--complete' in sys.argv
if COMPLETE:
    import paperB_complete as PC
    R=json.load(open(PC.OUT)); X={}; FINAL_OUT='paperB_final_complete.json'
else:
    R=json.load(open(B.OUT)); X=json.load(open(E.OUT)); FINAL_OUT='paperB_final.json'
FAMS=['static','ucb','ducb','swucb','ts']

def final(cell,fam):
    d=R[cell]; rec=X.get(f"{cell}|{fam}")
    if rec and rec['pick']!=rec['orig']:
        keys=list(rec['tune_mean']); return rec['pick'],np.array(rec['eval']),keys
    keys=[k for k in d['tune'] if k.startswith(fam+'|')]
    if rec: keys=list(rec['tune_mean'])
    k=d['best'][fam]; return k,np.array(d['eval'][k]),keys

def still_edge(pick,keys):
    fam,pp=E.parse(pick); ps=[E.parse(k)[1] for k in keys]
    for a,v in pp.items():
        if (a,v) in (('floor',0.0),('floor',1.0),('gamma',1.0),('tau',1000.0),('scale',0.0),('c',0.0)): continue   # natural limits
        vals=sorted({p[a] for p in ps})
        if len(vals)>1 and v in (vals[0],vals[-1]): return True
    return False

def cmp(a,b):
    _,p,dz=paired(a,b); tp,m,lo,hi=tost_paired(a,b,B.EPS)
    return dict(diff=m,ci=[lo,hi],p=p,tost=tp,dz=dz)

rows={}
for cell in R:
    F={f:final(cell,f) for f in FAMS}; S=F['static'][1]
    C={f"{f} vs static":cmp(F[f][1],S) for f in FAMS if f!='static'}
    C.update({f"ucb vs {f}":cmp(F['ucb'][1],F[f][1]) for f in ('ducb','swucb','ts')})
    C["ducb vs swucb"]=cmp(F['ducb'][1],F['swucb'][1]); C["ducb vs ts"]=cmp(F['ducb'][1],F['ts'][1])
    rows[cell]=dict(F=F,C=C,gap=R[cell]['gap'])
keys=[(c,l) for c in rows for l in rows[c]['C']]
adj=dict(zip(keys,holm(np.array([rows[c]['C'][l]['p'] for c,l in keys]))))

def v(r,pa): return ('WIN' if r['diff']>0 else 'LOSS') if pa<0.05 else ('equiv' if r['tost']<0.05 else 'inconc')

if __name__=='__main__':
    out={}
    for cell in sorted(rows,key=lambda c:(float(c.split('|')[1]),c)):
        r=rows[cell]; gap=r['gap']; print(f"\n{cell:<16} gap {gap:.4f}")
        for f,(k,_,ks) in r['F'].items():
            print(f"   {f:<7} {k.split('|')[1]:<22}{' STILL-EDGE' if still_edge(k,ks) else ''}")
        for lab,c in r['C'].items():
            pa=adj[(cell,lab)]
            print(f"   {lab:<16} {c['diff']:+.4f} [{c['ci'][0]:+.4f},{c['ci'][1]:+.4f}] {100*c['diff']/gap:+6.1f}%gap p_holm={pa:.2g} {v(c,pa)}")
        out[cell]=dict(gap=gap,picks={f:k for f,(k,_,_) in r['F'].items()},
                       still_edge={f:still_edge(k,ks) for f,(k,_,ks) in r['F'].items()},
                       comparisons={l:dict(c,p_holm=float(adj[(cell,l)]),verdict=v(c,adj[(cell,l)])) for l,c in r['C'].items()})
    print(f"\n{len(keys)} comparisons Holm-adjusted together")
    json.dump(out,open(FINAL_OUT,'w'),indent=1)
