"""
Paper B tables (no simulation). Reads paperB_final_complete.json, paperB_robust.py --complete
(recomputed in-process) and paperB_attack_results.json; writes ../paperB/tables/*.tex and
prints every number the prose quotes, so the text can be checked against this output.

Attack comparisons are Holm-adjusted together here (paperB_attack.py stores raw p).
"""
import json, os, sys, numpy as np
sys.argv = [sys.argv[0], '--complete']            # paperB_robust reads the complete grid
sys.stdout.reconfigure(encoding='utf-8')
from paperB_final import holm
import paperB_robust as R
import paperB_complete as PC

TAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'paperB', 'tables')
os.makedirs(TAB, exist_ok=True)
POOLS = ['Fixed', 'Growing', 'Shrinking', 'Evolving']
THETAS = ['0.0', '0.02', '0.1']
RULES = [('ucb', 'UCB'), ('ducb', 'D-UCB'), ('swucb', 'SW-UCB'), ('ts', 'TS')]
F = json.load(open('paperB_final_complete.json'))

def mark(pct, verdict):
    s = f"{pct:+.1f}".replace('-', '$-$')
    return {'WIN': r'\textbf{' + s + '}', 'LOSS': r'\underline{' + s + '}',
            'equiv': s, 'inconc': s + r'$^?$'}[verdict]

# ---------------- Table: every rule vs tuned static, per cell ----------------
L = [r'\begin{tabular}{l' + 'rrrr' * 3 + '}', r'\toprule',
     ' & ' + ' & '.join(rf'\multicolumn{{4}}{{c}}{{$\theta={t}$}}' for t in ('0', '0.02', '0.1')) + r'\\',
     r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}',
     'Rule & ' + ' & '.join(p[:2] if p != 'Shrinking' else 'Sh' for p in POOLS) * 0 +
     ' & '.join(['Fix', 'Gro', 'Shr', 'Evo'] * 3) + r'\\', r'\midrule',
     r'static $P^\star$ & ' + ' & '.join(F[f"{p}|{t}"]['picks']['static'].split('=')[1] + r'\%'
                                        for t in THETAS for p in POOLS) + r'\\', r'\midrule']
for f, name in RULES:
    row = []
    for t in THETAS:
        for p in POOLS:
            d = F[f"{p}|{t}"]; c = d['comparisons'][f"{f} vs static"]
            row.append(mark(100 * c['diff'] / d['gap'], c['verdict']))
    L.append(name + ' & ' + ' & '.join(row) + r'\\')
L.append(r'\midrule')
for a, b in (('ucb', 'ducb'), ('ducb', 'swucb'), ('ducb', 'ts')):
    lab = f"{a} vs {b}"; row = []
    for t in THETAS:
        for p in POOLS:
            d = F[f"{p}|{t}"]; c = d['comparisons'][lab]
            row.append(mark(100 * c['diff'] / d['gap'], c['verdict']))
    L.append(dict(RULES)[a] + ' $-$ ' + dict(RULES)[b] + ' & ' + ' & '.join(row) + r'\\')
L += [r'\bottomrule', r'\end{tabular}']
open(os.path.join(TAB, 'cells.tex'), 'w').write('\n'.join(L) + '\n')

# ---------------- Table: picks per cell ----------------
def short(k):
    f, ps = k.split('|'); kv = dict(p.split('=') for p in ps.split(','))
    sym = dict(c='c', floor=r'\alpha_f', gamma=r'\gamma', tau=r'\tau', scale='s', P='P')
    return '$' + ',\\,'.join(f"{sym[n]}{{=}}{float(v):g}" for n, v in kv.items()) + '$'
L = [r'\begin{tabular}{ll' + 'l' * 4 + '}', r'\toprule',
     r'$\theta$ & Pool & ' + ' & '.join(n for _, n in RULES) + r'\\', r'\midrule']
for t in THETAS:
    for i, p in enumerate(POOLS):
        pk = F[f"{p}|{t}"]['picks']
        L.append((f"${float(t):g}$" if i == 0 else '') + f" & {p} & " +
                 ' & '.join(short(pk[f]) for f, _ in RULES) + r'\\')
    if t != THETAS[-1]: L.append(r'\midrule')
L += [r'\bottomrule', r'\end{tabular}']
open(os.path.join(TAB, 'picks.tex'), 'w').write('\n'.join(L) + '\n')

# ---------------- Table: deployment without knowing the drift ----------------
FAMS = ['static', 'ucb', 'ducb', 'swucb', 'ts']; NAME = dict(RULES, static='static')
reg = {f: [R.regret[h][R.pick[(h, f)]] for h in R.CELLS] for f in FAMS}
vs = {}
for a, b in R.labs:
    if b != 'static': continue
    v = []
    for h in R.CELLS:
        r = R.res[(h, a, b)]; pa = R.adj[(h, a, b)]
        v.append('W' if pa < 0.05 and r['diff'] > 0 else 'L' if pa < 0.05 else 'E' if r['tost'] < 0.05 else '?')
    vs[a] = v
L = [r'\begin{tabular}{lrrcccc}', r'\toprule',
     r'Rule & mean & worst & win & loss & equiv. & inconcl.\\', r'\midrule']
for f in FAMS:
    v = vs.get(f)
    cnt = ' & '.join(str(v.count(x)) for x in 'WLE?') if v else r'\multicolumn{4}{c}{(reference)}'
    L.append(f"{NAME[f]} & ${np.mean(reg[f]):.1f}$ & ${max(reg[f]):.1f}$ & {cnt}" + r'\\')
L += [r'\bottomrule', r'\end{tabular}']
open(os.path.join(TAB, 'deploy.tex'), 'w').write('\n'.join(L) + '\n')

# ---------------- Table: whitewashing ----------------
A = json.load(open('paperB_attack_results.json'))
ACELLS = ['Fixed|0.0', 'Evolving|0.0', 'Fixed|0.02', 'Evolving|0.02']
keys = [(c, k) for c in ACELLS for k in A[c]['comparisons']]
aadj = dict(zip(keys, holm(np.array([A[c]['comparisons'][k]['p'] for c, k in keys]))))
def averdict(c, k):
    r = A[c]['comparisons'][k]; pa = aadj[(c, k)]
    return 'WIN' if pa < 0.05 and r['diff'] > 0 else 'LOSS' if pa < 0.05 else 'equiv' if r['tost'] < 0.05 else 'inconc'
ARMS = [('static', 'static'), ('ucb', 'UCB'), ('swucb', 'SW-UCB'), ('ts', 'TS'), ('ducb', 'D-UCB')] + \
       [(f"ducb+prior{n:g}", rf'\quad $+\,n_0{{=}}{n:g}$') for n in (1, 3, 10, 30, 100)]
def cellfmt(c, arm, frac):
    q = np.array(A[c]['quality'][f"{arm}|{frac}"]); S = np.array(A[c]['quality'][f"static|{frac}"])
    if arm == 'static': return f"{q.mean():.3f}"
    k = f"{arm} vs static|{frac}"; d = 1000 * (q.mean() - S.mean())
    return mark(d, averdict(c, k))
L = [r'\begin{tabular}{l' + 'rrr' * 4 + '}', r'\toprule',
     ' & ' + ' & '.join(rf"\multicolumn{{3}}{{c}}{{{c.split('|')[0]}, $\theta={float(c.split('|')[1]):g}$}}" for c in ACELLS) + r'\\',
     r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}',
     'Rule & ' + ' & '.join([r'$a{=}0$', r'$a{=}20\%$', 'adv.'] * 4) + r'\\', r'\midrule']
for arm, name in ARMS:
    row = []
    for c in ACELLS:
        sh = 100 * np.mean(A[c]['adv_share'][f"{arm}|0.2"])
        row += [cellfmt(c, arm, '0.0'), cellfmt(c, arm, '0.2'), f"{sh:.1f}"]
    L.append(name + ' & ' + ' & '.join(row) + r'\\')
    if arm == 'static': L.append(r'\midrule')
L += [r'\bottomrule', r'\end{tabular}']
open(os.path.join(TAB, 'attack.tex'), 'w').write('\n'.join(L) + '\n')

# ---------------- numbers quoted in the prose ----------------
if __name__ == '__main__':
    def pct(c, lab): d = F[c]; return 100 * d['comparisons'][lab]['diff'] / d['gap']
    st = [pct(f"{p}|0.0", f"{f} vs static") for p in POOLS for f, _ in RULES]
    print(f"stationary gain range: {min(st):.1f}..{max(st):.1f}%; all WIN:",
          all(F[f"{p}|0.0"]['comparisons'][f"{f} vs static"]['verdict'] == 'WIN' for p in POOLS for f, _ in RULES))
    print("stationary ucb-vs-other verdicts:", [F[f"{p}|0.0"]['comparisons'][f"ucb vs {g}"]['verdict'] for p in POOLS for g in ('ducb', 'swucb', 'ts')])
    for t in ('0.02', '0.1'):
        print(f"theta={t} static P*:", [F[f"{p}|{t}"]['picks']['static'] for p in POOLS])
    dr = [(c, F[c]) for c in F if c.split('|')[1] != '0.0']
    for f, _ in RULES:
        v = [d['comparisons'][f"{f} vs static"]['verdict'] for _, d in dr]
        p_ = [pct(c, f"{f} vs static") for c, _ in dr]
        print(f"drift {f}: W/L/eq/? = {v.count('WIN')}/{v.count('LOSS')}/{v.count('equiv')}/{v.count('inconc')}"
              f"  loss range {min([x for x, vv in zip(p_, v) if vv == 'LOSS'], default=0):.1f}..{max([x for x, vv in zip(p_, v) if vv == 'LOSS'], default=0):.1f}")
    print("drift UCB picks:", [d['picks']['ucb'] for _, d in dr])
    print(f"Holm comparisons: {sum(len(d['comparisons']) for d in F.values())}")
    print("deploy:", {f: (round(np.mean(reg[f]), 1), round(max(reg[f]), 1)) for f in FAMS},
          {a: ''.join(v) for a, v in vs.items()})
    print(f"attack: {len(keys)} comparisons Holm-adjusted")
    for c in ACELLS:
        print(" ", c, {k: averdict(c, k) for k in A[c]['comparisons'] if k.endswith('|0.2') or k.endswith('|0.0')})
