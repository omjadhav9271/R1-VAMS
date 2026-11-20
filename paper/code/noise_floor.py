"""
Is the VAMS control signal sigma_A informative, or estimator noise?

For a sample of n draws with standard deviation s, the sampling sd of the (ddof=0) sample
sd is ~ s / sqrt(2n). If sigma_A's epoch-to-epoch fluctuation matches that floor, and its
lag-1 autocorrelation is ~0, it carries no state the controller could track.
Measured under static P=25 (the operating point of VAMS[10,40]) after a burn-in.

Also reports the latent spread of the active set vs the noise V, which is why the signal
is noise-dominated: sigma_A^2 ~ V^2 + Var(latent | active).
"""
import numpy as np, sys, os, time, json
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from engine_v import run, SCENARIOS

NS=100; SEEDS=list(range(14000,14000+NS)); BURN=200

def _j(a):
    name,s=a
    _,_,S,_,_,_=run(**SCENARIOS[name], mode='static', P_static=25.0, seed=s)
    x=S[BURN:]; x=x[~np.isnan(x)]
    n=SCENARIOS[name]['Nact']
    ac=np.corrcoef(x[:-1],x[1:])[0,1]
    return float(x.mean()), float(x.std(ddof=1)), float(x.mean()/np.sqrt(2*n)), float(ac)

if __name__=='__main__':
    t0=time.time(); nw=int(os.environ.get('WORKERS','4')); out={}
    print(f"sigma_A NOISE FLOOR (static P=25, {NS} seeds, epochs {BURN}-999)")
    print(f"{'Scenario':<10} {'mean sigma_A':>12} {'sd(sigma_A)':>12} {'floor s/sqrt(2n)':>17} "
          f"{'ratio':>6} {'lag-1 ac':>9} {'implied latent sd':>17}")
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for name in SCENARIOS:
            r=np.array(list(ex.map(_j,[(name,s) for s in SEEDS],chunksize=8)))
            m,sd,fl,ac=r.mean(axis=0)
            n=SCENARIOS[name]['Nact']        # ddof=0 sd: E[s^2] = (n-1)/n sigma^2
            lat=np.sqrt(max(m**2*n/(n-1)-0.2**2,0.0))
            print(f"{name:<10} {m:>12.4f} {sd:>12.4f} {fl:>17.4f} {sd/fl:>6.3f} {ac:>+9.3f} {lat:>17.3f}")
            out[name]=dict(per_seed=r.tolist(),mean_sigma=m,sd_sigma=sd,floor=fl,lag1=ac,latent_sd=lat)
    json.dump(out,open('noise_floor_results.json','w'))
    print(f"elapsed {time.time()-t0:.0f}s   -> noise_floor_results.json")
