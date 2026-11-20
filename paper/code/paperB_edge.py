"""
Follow-up to paperB_baselines.py: extend the grid past every truncated edge pick.

For each cell and family whose held-out pick sits on a grid edge (natural limits floor=0 and
gamma=1 excluded), add values beyond that edge in the offending parameter, crossed with the
family's full grid in its other parameter, and run them on the same TUNE seeds. If the
extended grid yields a better tune score, the new pick is re-run on the same EVAL seeds and
compared, paired, with that cell's static and UCB picks. Repeat until no pick is on an edge
or the extension list is exhausted. Never changes paperB_baselines_results.json.
"""
import numpy as np, sys, os, time, json, itertools
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from engine_v import run, SCENARIOS
from stats_util import paired, tost_paired
import paperB_baselines as B

OUT='paperB_edge_results.json'
# values beyond each end of each parameter, tried nearest-first
BEYOND={('static','P'):     dict(hi=[70,80]),
        ('ucb','c'):        dict(lo=[0.02,0.03], hi=[0.8,1.0,1.5,2.0,3.0]),
        ('ucb','floor'):    dict(hi=[0.3,0.4]),
        ('ducb','c'):       dict(lo=[0.01,0.02,0.03], hi=[0.8,1.0]),
        ('ducb','gamma'):   dict(lo=[0.8,0.85], hi=[0.9995,1.0]),
        ('swucb','c'):      dict(lo=[0.01,0.02,0.03,0.05], hi=[0.8,1.0]),
        ('swucb','tau'):    dict(lo=[2,3,5], hi=[600,800]),
        ('ts','scale'):     dict(lo=[0.02,0.03], hi=[0.6,0.8]),
        ('ts','gamma'):     dict(lo=[0.75,0.8,0.85,0.9])}
MODE={'static':'static','ucb':'ucb','ducb':'ducb','swucb':'swucb','ts':'ts'}
KW={'P':'P_static','c':'c_ucb','floor':'alpha_floor','gamma':'gamma','tau':'sw_tau','scale':'ts_scale'}

def parse(key):
    fam,ps=key.split('|'); return fam,{a:float(b) for a,b in (p.split('=') for p in ps.split(','))}

def make(fam,params):
    key=f"{fam}|"+",".join(f"{a}={int(v) if a in ('P','tau') else v}" for a,v in params.items())
    kw=dict(mode=MODE[fam]); kw.update({KW[a]:(float(v) if a!='tau' else int(v)) for a,v in params.items()})
    return key,kw

def grid_vals(fam):
    ks=[parse(k)[1] for k in B.GRID if k.startswith(fam+'|')]
    return {a:sorted({p[a] for p in ks}) for a in ks[0]}

def _j(a):
    name,th,kw,s=a
    return float(np.nanmean(run(**SCENARIOS[name], ou_theta=th, seed=s, **kw)[5]))

def cmp(a,b):
    _,p,dz=paired(a,b); tp,m,lo,hi=tost_paired(a,b,B.EPS)
    return dict(diff=m,ci=[lo,hi],p=p,tost=tp,dz=dz)

if __name__=='__main__':
    t0=time.time(); nw=int(os.environ.get('WORKERS','4'))
    R=json.load(open(B.OUT)); out=json.load(open(OUT)) if os.path.exists(OUT) else {}
    with ProcessPoolExecutor(max_workers=nw) as ex:
        def go(name,th,kw,seeds):
            return np.array(list(ex.map(_j,[(name,th,kw,s) for s in seeds],chunksize=10)))
        for cell,d in R.items():
            name,th=cell.split('|')[0],float(cell.split('|')[1])
            for fam in ('static','ucb','ducb','swucb','ts'):
                tag=f"{cell}|{fam}"
                if tag in out or not d['edge'][fam]: continue
                tune={k:np.array(v) for k,v in d['tune'].items() if k.startswith(fam+'|')}
                vals=grid_vals(fam); tried=set(); pick=d['best'][fam]
                while True:
                    f_,pp=parse(pick); new=[]
                    for a,v in pp.items():
                        lo,hi=vals[a][0],vals[a][-1]
                        if (a,v)==('floor',0.0) or (a,v)==('gamma',1.0): continue
                        ext=BEYOND.get((fam,a),{}).get('lo' if v==lo else 'hi' if v==hi else '',[])
                        ext=[e for e in ext if (a,e) not in tried]
                        if not ext: continue
                        e=ext[0] if v==hi else ext[-1]          # nearest value beyond the edge
                        tried.add((a,e)); vals[a]=sorted(vals[a]+[e])
                        others=[b for b in pp if b!=a]
                        for combo in itertools.product(*[vals[b] for b in others]):
                            p2=dict(pp); p2[a]=e; p2.update(dict(zip(others,combo)))
                            new.append(make(fam,{k:p2[k] for k in pp}))
                    if not new: break
                    for k,kw in new:
                        if k not in tune: tune[k]=go(name,th,kw,B.TUNE)
                    best=max(tune,key=lambda k:tune[k].mean())
                    print(f"{tag}: added {len(new)}; pick {pick} -> {best}")
                    if best==pick: break
                    pick=best
                rec=dict(orig=d['best'][fam],pick=pick,tune_mean={k:float(v.mean()) for k,v in tune.items()})
                if pick!=d['best'][fam]:
                    ev=go(name,th,make(*parse(pick))[1],B.EVAL)
                    S=np.array(d['eval'][d['best']['static']]); U=np.array(d['eval'][d['best']['ucb']])
                    O=np.array(d['eval'][d['best'][fam]])
                    rec.update(eval=ev.tolist(),vs_static=cmp(ev,S),ucb_vs_new=cmp(U,ev),new_vs_orig=cmp(ev,O))
                    print(f"  eval: new-orig {rec['new_vs_orig']['diff']:+.5f}  ucb-new {rec['ucb_vs_new']['diff']:+.5f} "
                          f"CI[{rec['ucb_vs_new']['ci'][0]:+.5f},{rec['ucb_vs_new']['ci'][1]:+.5f}]")
                out[tag]=rec; json.dump(out,open(OUT,'w'))
    print(f"elapsed {time.time()-t0:.0f}s -> {OUT}")
