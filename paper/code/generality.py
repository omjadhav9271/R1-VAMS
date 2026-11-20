"""
Generality of the bias: the same audit on a different mechanism class.

UCB selection ranks participants by running-mean score + c*sqrt(log t / n_j) and imputes
nothing for inactive participants. Raising T_ij raises only j's own score, so retention is
monotone in T_ij and Proposition 1 applies unchanged. The exploration constant c plays the
role P plays in percentile sortition: it drives turnover. Prediction: M2 - M1 grows with
churn, and the c that maximises M2 is larger than the c that maximises M1.
"""
import numpy as np, sys, os, time, json
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from engine_v import run, SCENARIOS

NS=50; SEEDS=list(range(15000,15000+NS))
CS=[0.02,0.05,0.1,0.2,0.3,0.5,1.0,2.0]

def _j(a):
    name,c,s=a
    q,_,_,L,C,OA=run(**SCENARIOS[name], mode='ucb', c_ucb=c, seed=s)
    return float(np.nanmean(OA)), float(np.nanmean(q)), float(np.nanmean(L)), float(np.nanmean(C))

if __name__=='__main__':
    t0=time.time(); nw=int(os.environ.get('WORKERS','4')); out={}; xs=[]; ys=[]
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for name in SCENARIOS:
            print(f"--- {name}: UCB, {NS} seeds ---")
            print(f"{'c':>6} {'M1':>8} {'M2':>8} {'M3':>8} {'churn':>7} {'M2-M1':>8}")
            res={}
            for c in CS:
                r=np.array(list(ex.map(_j,[(name,c,s) for s in SEEDS],chunksize=5)))
                m=r.mean(axis=0); res[c]=r
                print(f"{c:>6} {m[0]:>8.4f} {m[1]:>8.4f} {m[2]:>8.4f} {m[3]:>7.3f} {m[1]-m[0]:>+8.4f}")
                xs.append(m[3]); ys.append(m[1]-m[0])
            a1=max(CS,key=lambda c:res[c][:,0].mean()); a2=max(CS,key=lambda c:res[c][:,1].mean())
            a3=max(CS,key=lambda c:res[c][:,2].mean())
            loss=res[a1][:,0].mean()-res[a2][:,0].mean()
            print(f"  argmax c:  M1={a1}  M2={a2}  M3={a3}   M1 lost by tuning c on M2: {loss:.4f}")
            out[name]={str(c):v.tolist() for c,v in res.items()}
            out[name+'|argmax']=dict(M1=a1,M2=a2,M3=a3,loss=loss)
    xs,ys=np.array(xs),np.array(ys)
    sl=float(xs@ys/(xs@xs)); r2=1-((ys-sl*xs)**2).sum()/((ys-ys.mean())**2).sum()
    print(f"\nbias vs churn (through origin), all pools and c: slope={sl:.4f}  R2={r2:.3f}  "
          f"corr={np.corrcoef(xs,ys)[0,1]:.3f}  min bias={ys.min():+.4f}")
    out['fit']=dict(slope=sl,r2=r2)
    json.dump(out,open('generality_results.json','w'))
    print(f"elapsed {time.time()-t0:.0f}s   -> generality_results.json")
