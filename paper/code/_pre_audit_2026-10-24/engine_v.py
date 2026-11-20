"""
Vectorized re-implementation of the Kruijssen et al. (2024) merit-based sortition
engine, plus the adaptive (VAMS) and uncertainty-directed (UCB) variants.

Semantics preserved from the original notebook engine:
  - instantaneous T drawn only for ACTIVE+ALIVE participants
  - Eq.3 imputation: inactive participants receive the P-th percentile of active T
  - selection: top-N_act by estimator, with random tie-breaking at the cutoff
  - attrition via geometric lifetime; growth via Poisson arrivals
  - new arrivals seeded with Q_est = target_inactive

Latent-quality non-stationarity is modelled as a mean-reverting (OU) process so that
the STATIONARY variance stays sigma_Q^2 for every theta -- this isolates the effect of
drift from the effect of a widening quality distribution.
"""
import numpy as np
from scipy.special import expit
from collections import deque

MODES = ('random', 'static', 'adaptive', 'ucb', 'greedy', 'sched')
# alpha_floor interpolates the estimator's memory:
#   0.0  -> a_t = 1/n   : convergent sample mean (infinite memory)
#   f>0  -> a_t = max(1/n, f) : sample mean early, then fixed-alpha EMA (memory ~1/f)
# so one scalar spans 'remember everything' .. 'forget at rate f'.


def run(Ninit, Nact, Nepochs=1000, mu_Q=0.2, sigma_Q=0.1, V=0.2, alpha=0.1,
        pgrowth=1e-10, pattr=1e-10, mode='static', P_static=25.0,
        P_min=10.0, P_max=40.0, k=20.0, window=50,
        c_ucb=0.3, ou_theta=0.0, alpha_floor=0.0, sched=None, seed=42):

    rng = np.random.default_rng(seed)
    p_safe = max(pattr, 1e-15)

    # OU innovation scale chosen so stationary sd == sigma_Q for any theta
    ou_step = sigma_Q * np.sqrt(max(2.0 * ou_theta - ou_theta ** 2, 0.0)) if ou_theta > 0 else 0.0

    n = Ninit
    Q_med = rng.normal(mu_Q, sigma_Q, n)
    Q_est = np.full(n, np.nan)               # EMA (static/adaptive) or sample mean (ucb/greedy)
    n_obs = np.zeros(n, dtype=np.int64)
    alive = np.ones(n, dtype=bool)
    active = np.zeros(n, dtype=bool)
    lifetime = (rng.geometric(p_safe, n) if pattr > 0 else np.full(n, 10 ** 8)).astype(np.int64)

    use_mean = mode in ('ucb', 'greedy')
    mean_T = np.full(Nepochs, np.nan)
    P_hist = np.full(Nepochs, np.nan)
    S_hist = np.full(Nepochs, np.nan)
    # bias-free metrics: latent quality of the selected set, and turnover
    lat_Q  = np.full(Nepochs, np.nan)
    churn  = np.full(Nepochs, np.nan)
    # M1: quality DELIVERED by the set active during this epoch, measured before any
    # selection consumes it -> unbiased. This is the paper's "mean instantaneous
    # quality metric across the active set" (Kruijssen et al. 2024, sec 3.4).
    obs_A  = np.full(Nepochs, np.nan)
    sbuf = deque(maxlen=window)

    ia = np.flatnonzero(alive)
    active[rng.choice(ia, size=min(Nact, ia.size), replace=False)] = True

    for i in range(Nepochs):
        if ou_theta > 0.0:
            Q_med += ou_theta * (mu_Q - Q_med) + ou_step * rng.standard_normal(Q_med.size)

        ia = np.flatnonzero(alive)
        if ia.size == 0:
            continue

        am = active & alive
        act_idx = np.flatnonzero(am)
        if act_idx.size == 0:
            continue

        T_act = rng.normal(Q_med[act_idx], V)
        std_T = float(T_act.std()) if T_act.size > 1 else 0.0
        S_hist[i] = std_T
        obs_A[i]  = float(T_act.mean())
        lat_A     = float(Q_med[act_idx].mean())

        if mode == 'random':
            P_t = 25.0
        elif mode == 'sched':
            # two-phase schedule (p_early, p_late, switch_epoch): an ORACLE-tuned
            # time-varying policy, used to bound the headroom available to ANY
            # time-varying rule for P. If the best schedule ties the best constant P,
            # no adaptive rule of this form can help.
            p1, p2, t0 = sched
            P_t = float(p1) if i < t0 else float(p2)
        elif mode in ('static', 'ucb', 'greedy'):
            P_t = P_static
        else:                                   # adaptive / VAMS
            sbuf.append(std_T)
            P_t = P_min + (P_max - P_min) * float(expit(k * (std_T - float(np.mean(sbuf)))))
        P_hist[i] = P_t
        target_inactive = float(np.percentile(T_act, P_t))

        # ---------------- estimator update ----------------
        if mode != 'random':
            if use_mean:
                # Robbins-Monro step with a floor: spans sample mean .. fixed-alpha EMA
                n_obs[act_idx] += 1
                prev = Q_est[act_idx]
                step = np.maximum(1.0 / n_obs[act_idx], alpha_floor)
                upd = np.where(np.isnan(prev), T_act, prev + step * (T_act - prev))
                Q_est[act_idx] = upd
            else:
                hat = np.full(ia.size, target_inactive)
                is_act = am[ia]
                hat[is_act] = T_act
                prev = Q_est[ia]
                Q_est[ia] = np.where(np.isnan(prev), hat, alpha * hat + (1 - alpha) * prev)
                n_obs[act_idx] += 1

        # ---------------- selection ----------------
        take = min(Nact, ia.size)
        order = rng.permutation(ia)             # fair random tie-breaking
        if mode == 'random':
            new_active = order[:take]
        elif mode in ('ucb', 'greedy'):
            est = Q_est[order]
            cnt = n_obs[order]
            if mode == 'ucb':
                bonus = c_ucb * np.sqrt(np.log(i + 2.0) / np.maximum(cnt, 1))
                score = est + bonus
            else:
                score = est.copy()
            score = np.where((cnt == 0) | np.isnan(est), np.inf, score)
            new_active = order[np.argsort(-score, kind='stable')[:take]]
        else:
            est = Q_est[order]
            valid = ~np.isnan(est)
            if valid.sum() <= take:
                sel = list(order[valid])
                rest = order[~valid]
                need = take - len(sel)
                if need > 0 and rest.size:
                    sel += list(rest[:need])
                new_active = np.array(sel, dtype=np.int64)
            else:
                v = order[valid]
                vv = est[valid]
                new_active = v[np.argsort(-vv, kind='stable')[:take]]

        active[:] = False
        active[new_active] = True

        # realised quality of the NEW active set (fresh draw for newly promoted)
        obs = np.empty(new_active.size, dtype=float)
        was = am[new_active]
        if was.any():
            # act_idx is ascending, so searchsorted recovers each member's slot in T_act
            obs[was] = T_act[np.searchsorted(act_idx, new_active[was])]
        nw = ~was
        if nw.any():
            obs[nw] = rng.normal(Q_med[new_active[nw]], V)
        mean_T[i] = float(obs.mean())
        # latent quality of the selected set: no measurement noise, no selection bias
        lat_Q[i] = float(Q_med[new_active].mean())
        churn[i] = float((~was).sum() / max(new_active.size, 1))

        # ---------------- attrition ----------------
        lifetime[ia] -= 1
        dead = ia[lifetime[ia] <= 0]
        if dead.size:
            alive[dead] = False
            active[dead] = False

        # ---------------- growth ----------------
        n_new = rng.poisson(pgrowth)
        if n_new > 0:
            Q_med = np.concatenate([Q_med, rng.normal(mu_Q, sigma_Q, n_new)])
            Q_est = np.concatenate([Q_est, np.full(n_new, target_inactive)])
            n_obs = np.concatenate([n_obs, np.zeros(n_new, dtype=np.int64)])
            alive = np.concatenate([alive, np.ones(n_new, dtype=bool)])
            active = np.concatenate([active, np.zeros(n_new, dtype=bool)])
            nl = rng.geometric(p_safe, n_new) if pattr > 0 else np.full(n_new, 10 ** 8)
            lifetime = np.concatenate([lifetime, nl.astype(np.int64)])

    return mean_T, P_hist, S_hist, lat_Q, churn, obs_A


SCENARIOS = {
    'Fixed':     dict(Ninit=80,  Nact=50, pgrowth=1e-10, pattr=1e-10),
    'Growing':   dict(Ninit=8,   Nact=32, pgrowth=0.1,   pattr=1e-10),
    'Shrinking': dict(Ninit=100, Nact=32, pgrowth=1e-10, pattr=2e-3),
    'Evolving':  dict(Ninit=100, Nact=32, pgrowth=0.1,   pattr=2e-3),
}
