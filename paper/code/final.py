"""
Definitive comparison: every mechanism, scored on BOTH the unbiased metric M1 and the
notebooks' biased metric M2. The point is that the metric choice reverses the ranking.
"""
import numpy as np, sys, os, time, json
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from engine_v import run, SCENARIOS
from stats_util import welch, tost

NS=100; SEEDS=list(range(8000,8000+NS)); EPS=0.005
MECHS=[('random',      dict(mode='random')),
       ('static P=20', dict(mode='static',  P_static=20.0)),
       ('static P=25', dict(mode='static',  P_static=25.0)),
       ('static P=40', dict(mode='static',  P_static=40.0)),
       ('static P=70', dict(mode='static',  P_static=70.0)),
       ('VAMS [10,40]',dict(mode='adaptive',P_min=10.,P_max=40.,k=20.,window=50)),
       ('VAMS [55,85]',dict(mode='adaptive',P_min=55.,P_max=85.,k=20.,window=50)),
       ('UCB c=0.3',   dict(mode='ucb',     c_ucb=0.3))]

def _j(a):
    name,kw,s=a
    q,Ph,S,L,C,OA = run(**SCENARIOS[name], seed=s, **kw)
    return float(np.nanmean(OA)), float(np.nanmean(q)), float(np.nanmean(L)), float(np.nanmean(C))

if __name__=='__main__':
    t0=time.time(); nw=int(os.environ.get('WORKERS','4')); R={}
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for n in SCENARIOS:
            for mn,kw in MECHS:
                r=list(ex.map(_j,[(n,kw,s) for s in SEEDS],chunksize=8))
                R[(n,mn)]=tuple(np.array([x[i] for x in r]) for i in range(4))
    print("="*118)
    print(f"DEFINITIVE MECHANISM COMPARISON ({NS} seeds). Reference = static P=25%.")
    print("  M1 = unbiased mean(T_act)   M2 = notebooks' mean_T   d/TOST computed on each metric")
    print("="*118)
    out={}
    for n in SCENARIOS:
        ref1,ref2=R[(n,'static P=25')][0],R[(n,'static P=25')][1]
        print(f"\n--- {n} Pool ---")
        print(f"{'mechanism':<14} | {'M1':>8} {'d1':>7} {'p1':>9} {'TOST1':>7} | "
              f"{'M2':>8} {'d2':>7} {'p2':>9} {'TOST2':>7} | {'churn':>6}")
        print("-"*118)
        for mn,_ in MECHS:
            m1,m2,m3,c=R[(n,mn)]
            _,p1,d1=welch(m1,ref1); t1,*_=tost(m1,ref1,EPS)
            _,p2,d2=welch(m2,ref2); t2,*_=tost(m2,ref2,EPS)
            e1='EQUIV' if t1<0.05 else '-'; e2='EQUIV' if t2<0.05 else '-'
            print(f"{mn:<14} | {m1.mean():>8.4f} {d1:>+7.3f} {p1:>9.5f} {e1:>7} | "
                  f"{m2.mean():>8.4f} {d2:>+7.3f} {p2:>9.5f} {e2:>7} | {c.mean():>6.3f}")
            out.setdefault(n,{})[mn]=dict(m1=m1.tolist(),m2=m2.tolist(),m3=m3.tolist(),churn=c.tolist())
        b1=max(MECHS,key=lambda x:R[(n,x[0])][0].mean())[0]
        b2=max(MECHS,key=lambda x:R[(n,x[0])][1].mean())[0]
        print(f"  BEST on M1 (unbiased): {b1:<14}   BEST on M2 (notebooks'): {b2}")
    json.dump(out,open('final_results.json','w'))
    print(f"\nelapsed {time.time()-t0:.0f}s   -> final_results.json")
