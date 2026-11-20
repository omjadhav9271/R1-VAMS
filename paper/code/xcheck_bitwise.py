# bitwise check: new engine with defaults == original engine, every mode x pool x drift
import hashlib, numpy as np, itertools, sys
from concurrent.futures import ProcessPoolExecutor
import engine_v as N, engine_orig as O
MODES=[dict(mode='random'),dict(mode='static'),dict(mode='adaptive'),dict(mode='ucb',alpha_floor=0.05),
       dict(mode='greedy'),dict(mode='sched',sched=(10,40,300)),dict(mode='ducb',gamma=0.95),
       dict(mode='swucb',sw_tau=25),dict(mode='ts',gamma=0.98)]
def h(r): return hashlib.md5(b''.join(np.ascontiguousarray(x).tobytes() for x in r)).hexdigest()
def j(a):
    m,p,th,s=a; kw=dict(**N.SCENARIOS[p],ou_theta=th,seed=s,**m)
    return h(N.run(**kw))==h(O.run(**kw))
if __name__=='__main__':
    jobs=list(itertools.product(MODES,N.SCENARIOS,[0.0,0.1],[1,2]))
    with ProcessPoolExecutor(int(__import__("os").environ.get("WORKERS","4"))) as ex: r=list(ex.map(j,jobs))
    print(f"{sum(r)}/{len(r)} bitwise identical"); sys.exit(0 if all(r) else 1)
