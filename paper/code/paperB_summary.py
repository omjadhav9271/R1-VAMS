"""
Summarise paperB_baselines_results.json (no simulation). Works on a partial file.
Per cell: each family's held-out pick and its gain over tuned static (as a share of the
static-over-random gap), and UCB against each bandit baseline. p-values are Holm-adjusted
across every comparison in the file, since the paper reads them as one family of claims.
Verdicts at eps=0.005: WIN/LOSS need Holm p<0.05; EQUIV needs TOST p<0.05 (intersection-
union, no adjustment needed); otherwise inconclusive.
"""
import json, sys, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
R=json.load(open('paperB_baselines_results.json'))

def holm(ps):
    o=np.argsort(ps); m=len(ps); adj=np.empty(m); run=0.0
    for r,i in enumerate(o):
        run=max(run,min(1.0,(m-r)*ps[i])); adj[i]=run
    return adj

keys=[(c,lab) for c in R for lab in R[c]['comparisons']]
adj=dict(zip(keys,holm(np.array([R[c]['comparisons'][l]['p'] for c,l in keys]))))

def verdict(r,pa):
    if pa<0.05: return 'WIN ' if r['diff']>0 else 'LOSS'
    return 'EQUIV' if r['tost']<0.05 else 'inconc'

for cell in sorted(R,key=lambda c:(float(c.split('|')[1]),c)):
    d=R[cell]; gap=d['gap']
    print(f"\n{cell:<16} static-over-random gap {gap:.4f}")
    for f,k in d['best'].items():
        print(f"   {f:<9} {k.split('|')[1]:<22}{' EDGE' if d['edge'][f] else ''}")
    for lab,r in d['comparisons'].items():
        name=lab if lab.startswith('ucb vs') else f"{lab} vs static"
        pa=adj[(cell,lab)]
        print(f"   {name:<18} {r['diff']:+.4f} [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] "
              f"{100*r['diff']/gap:+6.1f}% gap  p_holm={pa:.2g}  {verdict(r,pa)}")
print(f"\n{len(R)}/12 cells, {len(keys)} comparisons Holm-adjusted together")
