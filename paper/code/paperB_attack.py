"""
Paper B threat model: whitewashing.

A fraction adv_frac of identities (initial pool and arrivals) are adversarial, with latent
quality adv_Q = mu_Q - 2 sigma_Q. Whenever an adversarial participant has been observed and
is then not selected, it abandons the identity and rejoins at once as a fresh participant
with the same latent quality (worst case: rejoining is free and instantaneous).

Rules that give a never-observed identity infinite optimism (ucb, ducb, swucb, ts) re-select
a fresh identity immediately; percentile sortition seeds it at the P-th percentile. The
defence: a newcomer prior of prior_n0 pseudo-observations at that same percentile
(bandit modes only), which D-UCB then discounts like any other observation.

Each rule uses its held-out pick from paperB_complete_results.json for that pool and drift
level, chosen on HONEST pools: the defender does not tune against the attack. prior_n0 is
swept rather than tuned, to show the price paid on an honest pool. M1 and adversary share of
the active set, 100 paired seeds per arm. Checkpoints per (pool, theta).
"""
import numpy as np, sys, os, time, json
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from engine_v import run, SCENARIOS
from stats_util import paired, tost_paired
import paperB_complete as C

SEEDS=list(range(15000,15100)); EPS=0.005; OUT='paperB_attack_results.json'
POOLS=['Fixed','Evolving']; THETAS=[0.0,0.02]; FRACS=[0.0,0.05,0.10,0.20]
# 30/100 added 2026-11-20: a 5-seed pilot showed n0=3 barely dents the attack, so {1,3,10}
# could end before the prior has any effect
N0S=[1.0,3.0,10.0,30.0,100.0]; ADV_Q=0.2-2*0.1

def _j(a):
    name,th,kw,frac,s=a; st={}
    r=run(**SCENARIOS[name],ou_theta=th,seed=s,adv_frac=frac,adv_Q=ADV_Q,stats=st,**kw)
    return float(np.nanmean(r[5])), float(np.nanmean(st['adv_share']))

def arms(cell):
    best=json.load(open(C.OUT))[cell]['best']
    A={f:C.GRID[best[f]] for f in ('static','ucb','ducb','swucb','ts')}
    for n0 in N0S: A[f"ducb+prior{n0:g}"]=dict(A['ducb'],prior_n0=n0)
    return A,best

if __name__=='__main__':
    t0=time.time(); nw=int(os.environ.get('WORKERS','4'))
    out=json.load(open(OUT)) if os.path.exists(OUT) else {}
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for th in THETAS:
            for name in POOLS:
                cell=f"{name}|{th}"
                if cell in out: print(f"{cell}: done, skipping"); continue
                A,best=arms(cell); tc=time.time(); q={}; sh={}
                for frac in FRACS:
                    for a,kw in A.items():
                        r=np.array(list(ex.map(_j,[(name,th,kw,frac,s) for s in SEEDS],chunksize=5)))
                        q[f"{a}|{frac}"]=r[:,0]; sh[f"{a}|{frac}"]=r[:,1]
                print(f"\n{cell}  ({time.time()-tc:.0f}s)  picks: "+", ".join(f"{f}={best[f].split('|')[1]}" for f in ('static','ducb','ts')))
                print(f"  {'rule':<16}"+"".join(f"{'a='+format(f,'.2f'):>18}" for f in FRACS))
                for a in A:
                    print(f"  {a:<16}"+"".join(f"{q[f'{a}|{f}'].mean():>9.4f} ({100*sh[f'{a}|{f}'].mean():>4.1f}%)" for f in FRACS))
                cmpd={}
                for frac in FRACS:
                    S=q[f"static|{frac}"]
                    for a in A:
                        if a=='static': continue
                        _,p,dz=paired(q[f"{a}|{frac}"],S); tp,m,lo,hi=tost_paired(q[f"{a}|{frac}"],S,EPS)
                        cmpd[f"{a} vs static|{frac}"]=dict(diff=m,ci=[lo,hi],p=p,tost=tp,dz=dz)
                out[cell]=dict(picks={a:kw for a,kw in A.items()},quality={k:v.tolist() for k,v in q.items()},
                               adv_share={k:v.tolist() for k,v in sh.items()},comparisons=cmpd)
                json.dump(out,open(OUT,'w'))
    print(f"elapsed {time.time()-t0:.0f}s -> {OUT}", flush=True)
