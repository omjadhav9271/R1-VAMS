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

MODES = ('random', 'static', 'adaptive', 'ucb', 'greedy', 'sched', 'ducb', 'swucb', 'ts')
# alpha_floor interpolates the estimator's memory:
#   0.0  -> a_t = 1/n   : convergent sample mean (infinite memory)
#   f>0  -> a_t = max(1/n, f) : sample mean early, then fixed-alpha EMA (memory ~1/f)
# so one scalar spans 'remember everything' .. 'forget at rate f'.
# Non-stationary bandit baselines (Paper B), top-N_act selection, one arm per participant:
#   ducb  -- Discounted UCB (Garivier & Moulines 2011): counts AND sums decay by gamma every
#            epoch, so an unobserved participant's bonus grows back (unlike alpha_floor, which
#            forgets in the estimate only). score = S/N + c*sqrt(log(sum N)/N).
#   swucb -- Sliding-window UCB: statistics over the last sw_tau epochs only.
#            score = S_w/N_w + c*sqrt(log(min(t,tau))/N_w); N_w==0 -> +inf.
#   ts    -- Gaussian Thompson sampling on the discounted statistics (gamma=1: plain mean):
#            score = S/N + ts_scale*z/sqrt(N), z~N(0,1). ts_scale ~ noise sd V is the
#            flat-prior posterior; it is tuned like c.

# The six VAMS response curves from VAMS_Final_Comparison.ipynb, map (sigma_A, sigma_ref) -> P_t.
# 'k' is each form's own gain parameter; None uses that notebook's default.
VAMS_FORMS = ('sigmoid', 'linear', 'exponential', 'power', 'threshold', 'tanh')
_FORM_K = dict(sigmoid=20.0, linear=100.0, exponential=10.0, power=2.0, threshold=0.0, tanh=15.0)


def vams_P(form, s, ref, P_min, P_max, P_static, k=None):
    k = _FORM_K[form] if k is None else k
    if form == 'sigmoid':
        return P_min + (P_max - P_min) * float(expit(k * (s - ref)))
    if form == 'linear':
        return float(np.clip(P_static + k * (s - ref), P_min, P_max))
    if form == 'exponential':
        return float(np.clip(P_min + (P_max - P_min) * (1.0 - np.exp(-k * s)), P_min, P_max))
    if form == 'power':
        ratio = (s / ref) ** k if ref > 1e-6 else 1.0
        return float(np.clip(P_min + (P_max - P_min) * ratio, P_min, P_max))
    if form == 'threshold':                     # k is the margin here
        return P_max if s > ref + k else P_min
    if form == 'tanh':
        return float(np.clip((P_min + P_max) / 2 + (P_max - P_min) / 2 * np.tanh(k * (s - ref)),
                             P_min, P_max))
    raise ValueError(form)


def run(Ninit, Nact, Nepochs=1000, mu_Q=0.2, sigma_Q=0.1, V=0.2, alpha=0.1,
        pgrowth=1e-10, pattr=1e-10, mode='static', P_static=25.0,
        P_min=10.0, P_max=40.0, k=None, window=50,
        c_ucb=0.3, ou_theta=0.0, alpha_floor=0.0, sched=None, form='sigmoid', seed=42,
        gamma=1.0, sw_tau=100, ts_scale=0.2):

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
    use_bandit = mode in ('ducb', 'swucb', 'ts')
    bN = np.zeros(n)                         # discounted / windowed observation count
    bS = np.zeros(n)                         # discounted / windowed sum of observations
    swbuf = deque()                          # swucb: (act_idx, T_act) of the last sw_tau epochs
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
        elif mode in ('static', 'ucb', 'greedy') or use_bandit:
            P_t = P_static
        else:                                   # adaptive / VAMS
            sbuf.append(std_T)
            P_t = vams_P(form, std_T, float(np.mean(sbuf)), P_min, P_max, P_static, k)
        P_hist[i] = P_t
        target_inactive = float(np.percentile(T_act, P_t))

        # ---------------- estimator update ----------------
        if use_bandit:
            if mode == 'swucb':
                swbuf.append((act_idx, T_act))
                bN[act_idx] += 1.0
                bS[act_idx] += T_act
                if len(swbuf) > sw_tau:
                    old_idx, old_T = swbuf.popleft()
                    bN[old_idx] -= 1.0
                    bS[old_idx] -= old_T
            else:
                bN *= gamma
                bS *= gamma
                bN[act_idx] += 1.0
                bS[act_idx] += T_act
            n_obs[act_idx] += 1
        elif mode != 'random':
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
        elif use_bandit:
            Nj = bN[order]
            # swucb counts are exact integers held in floats; < 0.5 means no obs in window
            seen = Nj > (0.5 if mode == 'swucb' else 0.0)
            Ns = np.where(seen, Nj, 1.0)
            est = bS[order] / Ns
            if mode == 'ducb':
                score = est + c_ucb * np.sqrt(np.log(max(float(bN.sum()), np.e)) / Ns)
            elif mode == 'swucb':
                score = est + c_ucb * np.sqrt(np.log(min(i + 2.0, sw_tau + 1.0)) / Ns)
            else:
                score = est + ts_scale * rng.standard_normal(order.size) / np.sqrt(Ns)
            score = np.where(seen, score, np.inf)
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
            bN = np.concatenate([bN, np.zeros(n_new)])
            bS = np.concatenate([bS, np.zeros(n_new)])
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
