"""
Full study: 100 seeds, equivalence-tested nulls, OU-drift crossover.
Run:  python study.py
"""
import numpy as np, sys, os, time, json
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from engine_v import run, SCENARIOS
from stats_util import welch, tost, power_for, n_needed

NSEED  = 100
SEEDS  = list(range(1000, 1000 + NSEED))
C_UCB  = 0.3            # single un-tuned value, used everywhere
EPS    = 0.005          # SESOI: 10% of the merit-over-random gap (~0.05)
MODEKW = {
    'random':   dict(mode='random'),
    'static':   dict(mode='static', P_static=25.0),
    'vams':     dict(mode='adaptive', P_min=10.0, P_max=40.0, k=20.0, window=50),
    'ucb':      dict(mode='ucb', c_ucb=C_UCB),
}

def _one(job):
    sc, kw, seed = job
    q, P, S, L, C = run(**sc, seed=seed, **kw)
    return float(np.nanmean(q)), float(np.nanmean(P)), float(np.nanmean(S))

def batch(ex, sc, kw):
    jobs = [(sc, kw, s) for s in SEEDS]
    res = list(ex.map(_one, jobs, chunksize=4))
    return np.array([r[0] for r in res])

def main():
    nw = max(1, (os.cpu_count() or 4) - 1)
    out = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=nw) as ex:
        # ---------- Experiment 1: four scenarios, four mechanisms ----------
        print("="*100)
        print(f"EXPERIMENT 1 — mechanism comparison   ({NSEED} seeds, un-tuned c={C_UCB}, SESOI={EPS})")
        print("="*100)
        print(f"{'Scenario':<10} {'Random':>8} {'StaticP':>8} {'VAMS':>8} {'UCB':>8} |"
              f" {'VAMS-vs-Static':>26} | {'UCB-vs-Static':>26}")
        print(f"{'':<10} {'':>8} {'':>8} {'':>8} {'':>8} |"
              f" {'d':>6} {'p':>8} {'TOST':>9} | {'d':>6} {'p':>8} {'TOST':>9}")
        print("-"*100)
        ex1 = {}
        for name, sc in SCENARIOS.items():
            per = {m: batch(ex, sc, kw) for m, kw in MODEKW.items()}
            ex1[name] = {m: v.tolist() for m, v in per.items()}
            _, pv, dv = welch(per['vams'], per['static'])
            _, pu, du = welch(per['ucb'],  per['static'])
            tv, *_ = tost(per['vams'], per['static'], EPS)
            tu, *_ = tost(per['ucb'],  per['static'], EPS)
            fv = 'EQUIV' if tv < 0.05 else 'incon.'
            fu = 'EQUIV' if tu < 0.05 else 'incon.'
            print(f"{name:<10} {per['random'].mean():>8.4f} {per['static'].mean():>8.4f} "
                  f"{per['vams'].mean():>8.4f} {per['ucb'].mean():>8.4f} |"
                  f" {dv:>6.3f} {pv:>8.5f} {fv:>9} | {du:>6.3f} {pu:>8.5f} {fu:>9}")
        out['exp1'] = ex1

        # ---------- Experiment 2: OU drift crossover ----------
        print("\n" + "="*100)
        print(f"EXPERIMENT 2 — OU drift crossover, stationary sd held at sigma_Q=0.1   ({NSEED} seeds)")
        print("="*100)
        print(f"{'theta':>8} {'innov sd':>9} {'StaticP+EMA':>12} {'UCB+mean':>10} {'Delta':>9} "
              f"{'d':>7} {'p':>9} {'winner':>12}")
        print("-"*100)
        thetas = [0.0, 1e-4, 5e-4, 1.25e-3, 5e-3, 2e-2]
        ex2 = {}
        for th in thetas:
            sc = dict(SCENARIOS['Evolving'])
            st = batch(ex, sc, dict(MODEKW['static'], ou_theta=th))
            uc = batch(ex, sc, dict(MODEKW['ucb'],    ou_theta=th))
            _, p, d = welch(uc, st)
            innov = 0.1*np.sqrt(max(2*th - th**2, 0.0))
            w = 'UCB+mean' if uc.mean() > st.mean() else 'StaticP+EMA'
            star = '*' if p < 0.05 else ' '
            print(f"{th:>8.5f} {innov:>9.4f} {st.mean():>12.4f} {uc.mean():>10.4f} "
                  f"{uc.mean()-st.mean():>+9.5f} {d:>7.3f} {p:>8.5f}{star} {w:>12}")
            ex2[str(th)] = dict(static=st.tolist(), ucb=uc.tolist(), innov=innov)
        out['exp2'] = ex2

        # ---------- Experiment 3: c-robustness (not cherry-picked) ----------
        print("\n" + "="*100)
        print(f"EXPERIMENT 3 — UCB robustness to c (Evolving Pool, {NSEED} seeds)")
        print("="*100)
        base = np.array(ex1['Evolving']['static'])
        print(f"{'c':>7} {'UCB mean':>10} {'Delta':>9} {'d':>7} {'p':>9}")
        print("-"*100)
        ex3 = {}
        for c in [0.1, 0.2, 0.3, 0.4, 0.6, 1.0]:
            uc = batch(ex, dict(SCENARIOS['Evolving']), dict(mode='ucb', c_ucb=c))
            _, p, d = welch(uc, base)
            print(f"{c:>7.2f} {uc.mean():>10.4f} {uc.mean()-base.mean():>+9.5f} {d:>7.3f} {p:>8.5f}"
                  f"{'*' if p<0.05 else ''}")
            ex3[str(c)] = uc.tolist()
        out['exp3'] = ex3

    # ---------- power table ----------
    print("\n" + "="*100)
    print("POWER — seeds needed for 80% power at two-sample alpha=0.05")
    print("="*100)
    for d in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        print(f"  d={d:<4}  achieved power at n={NSEED}: {power_for(d, NSEED):.3f}   "
              f"n for 80%: {n_needed(d)}")

    json.dump(out, open('study_results.json', 'w'))
    print(f"\nelapsed {time.time()-t0:.0f}s   workers={nw}   -> study_results.json")

if __name__ == '__main__':
    main()
