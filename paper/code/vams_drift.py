"""
VAMS under non-stationary latent quality -- the regime where adapting P should help most.

In the base model latent quality is fixed, so one could object that VAMS had nothing to
adapt to. Here latent quality follows a mean-reverting (OU) process with stationary sd held
at sigma_Q, at three drift rates. Compared, paired on M1: the default sigmoid VAMS and the
most aggressive form (bang-bang threshold) against static P at the VAMS rule's own realised
mean P, and against static P=25. The static grid is also reported so the drift-optimal P
is visible.
"""
import numpy as np, sys, os, time, json
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from engine_v import run, SCENARIOS
from stats_util import paired, tost_paired

NS=100; SEEDS=list(range(16000,16000+NS)); EPS=0.005
THETAS=[0.005,0.02,0.1]; POOLS=['Fixed','Evolving']; STAT=[15,20,25,30,40]
VAMS={'sigmoid':dict(mode='adaptive',form='sigmoid',window=50),
      'threshold':dict(mode='adaptive',form='threshold',window=50)}

def _j(a):
    name,th,kw,s=a
    r=run(**SCENARIOS[name], ou_theta=th, seed=s, **kw)
    return float(np.nanmean(r[5])), float(np.nanmean(r[1]))

if __name__=='__main__':
    t0=time.time(); nw=int(os.environ.get('WORKERS','4')); out={}
    with ProcessPoolExecutor(max_workers=nw) as ex:
        def go(name,th,kw):
            return np.array(list(ex.map(_j,[(name,th,kw,s) for s in SEEDS],chunksize=5)))
        for name in POOLS:
            for th in THETAS:
                st={P:go(name,th,dict(mode='static',P_static=float(P)))[:,0] for P in STAT}
                print(f"--- {name}, theta={th} ---  static: "+"  ".join(f"{P}:{st[P].mean():.4f}" for P in STAT))
                rec=dict(static={str(P):v.tolist() for P,v in st.items()})
                for lab,kw in VAMS.items():
                    r=go(name,th,kw); q=r[:,0]; pm=r[:,1].mean()
                    Pm=int(round(pm))
                    if Pm not in st: st[Pm]=go(name,th,dict(mode='static',P_static=float(Pm)))[:,0]
                    t1,m1,lo1,hi1=tost_paired(q,st[25],EPS); t2,m2,lo2,hi2=tost_paired(q,st[Pm],EPS)
                    print(f"  {lab:<9} M1={q.mean():.4f} meanP={pm:5.1f} | vs P=25 {m1:+.5f} "
                          f"[{lo1:+.4f},{hi1:+.4f}] TOST={t1:.3f} | vs P={Pm} {m2:+.5f} "
                          f"[{lo2:+.4f},{hi2:+.4f}] TOST={t2:.3f}")
                    rec[lab]=dict(m1=q.tolist(),P_mean=pm,vs25=[m1,lo1,hi1,t1],
                                  vs_meanP=[Pm,m2,lo2,hi2,t2])
                out[f"{name}|{th}"]=rec
    json.dump(out,open('vams_drift_results.json','w'))
    print(f"elapsed {time.time()-t0:.0f}s   -> vams_drift_results.json")
