"""
Bitwise regression check for engine_v.run, used before and after any engine change.

    python engine_regress.py . before.json      # before editing engine_v.py
    python engine_regress.py . after.json       # after
    python engine_regress.py --diff before.json after.json

Every pre-existing mode must hash identically, so a new feature cannot move any number a
paper already reports. Take the 'before' snapshot on the machine that will make the edit:
hashes are only guaranteed stable for the same numpy version.
"""
import sys, json, hashlib, numpy as np
if sys.argv[1]=='--diff':
    a,b=(json.load(open(f)) for f in sys.argv[2:4]); bad=[k for k in a if a[k]!=b.get(k)]
    print(f"{len(bad)} of {len(a)} differ", bad[:10]); sys.exit(1 if bad else 0)
sys.path.insert(0, sys.argv[1]); from engine_v import run, SCENARIOS
CASES=[dict(mode='random'),dict(mode='static',P_static=25.0),dict(mode='adaptive'),
       dict(mode='adaptive',form='tanh',k=5.0,window=10),dict(mode='ucb',c_ucb=0.2),
       dict(mode='ucb',c_ucb=0.15,alpha_floor=0.1),dict(mode='greedy'),
       dict(mode='sched',sched=(40,20,50)),dict(mode='static',P_static=25.0,ou_theta=0.05),
       dict(mode='ucb',c_ucb=0.2,alpha_floor=0.05,ou_theta=0.1),
       dict(mode='ducb',c_ucb=0.05,gamma=0.9,ou_theta=0.02),dict(mode='swucb',c_ucb=0.1,sw_tau=25),
       dict(mode='ts',ts_scale=0.1,gamma=0.8,ou_theta=0.1)]
h={}
for name,sc in SCENARIOS.items():
    for i,kw in enumerate(CASES):
        for s in (1,77):
            out=run(**sc,Nepochs=300,seed=s,**kw)
            h[f"{name}|{i}|{s}"]=hashlib.sha256(b"".join(np.ascontiguousarray(a).tobytes() for a in out)).hexdigest()
json.dump(h,open(sys.argv[2],'w'),indent=0); print(len(h),"hashes")
