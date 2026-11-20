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
# Whitewashing threat model (Paper B, paperB_attack.py):
#   adv_frac  -- each identity (initial pool and arrivals) is adversarial with this probability,
#                drawn from a separate stream so adv_frac=0 leaves every run bitwise unchanged.
#                Adversaries have latent quality pinned at adv_Q (no OU drift).
#   An adversary that has been observed and is then not selected abandons the identity and
#   rejoins at once as a fresh one (same latent quality, same remaining lifetime): free,
#   instantaneous whitewashing, the worst case for the defender.
#   prior_n0  -- bandit modes (ducb, swucb, ts): every identity that joins after epoch 0
#                (honest arrival or rejoin) starts with prior_n0 pseudo-observations at the
#                epoch's imputed percentile, instead of zero (infinite optimism). D-UCB/TS
#                discount them like real ones; under SW-UCB they never leave the window.
#   stats     -- optional dict; filled with 'adv_share', the adversarial share of each
#                epoch's selected set.

# Observation noise (Paper A, pivotality): noise= 'gauss' (default), 'laplace', 'uniform', 't5',
# each mean zero with sd V. g(u) = E[eps 1{eps > u}] is the covariance a participant with flip
# point u contributes to the selection bias (Stein / zero-bias identity; V*phi(u/V) if Gaussian).
NOISES = ('gauss', 'laplace', 'uniform', 't5')


def noise_g(u, noise, V):
    u = np.asarray(u, dtype=float)
    if noise == 'gauss':
        return V * np.exp(-0.5 * (u / V) ** 2) / np.sqrt(2 * np.pi)
    if noise == 'laplace':
        b = V / np.sqrt(2.0)
        return 0.5 * (np.abs(u) + b) * np.exp(-np.abs(u) / b)
    if noise == 'uniform':
        a = V * np.sqrt(3.0)
        return np.where(np.abs(u) < a, (a * a - u * u) / (4 * a), 0.0)
    if noise == 't5':
        from scipy.stats import t as _t
        sc = V * np.sqrt(3.0 / 5.0); x = u / sc
        return sc * (5.0 + x * x) / 4.0 * _t.pdf(x, 5)
    raise ValueError(noise)


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
        gamma=1.0, sw_tau=100, ts_scale=0.2,
        adv_frac=0.0, adv_Q=0.0, prior_n0=0.0, stats=None, noise='gauss'):

    rng = np.random.default_rng(seed)
    if noise == 'gauss':
        draw = lambda loc: rng.normal(loc, V)
    elif noise == 'laplace':
        draw = lambda loc: rng.laplace(loc, V / np.sqrt(2.0))
    elif noise == 'uniform':
        draw = lambda loc: rng.uniform(loc - V * np.sqrt(3.0), loc + V * np.sqrt(3.0))
    elif noise == 't5':
        draw = lambda loc: loc + V * np.sqrt(3.0 / 5.0) * rng.standard_t(5, np.shape(loc))
    else:
        raise ValueError(noise)
    # pivotality diagnostic (static / ucb only): predicted selection bias from flip points
    want_pivot = stats is not None and stats.get('pivot') is True and mode in ('static', 'ucb')
    piv_bias = np.full(Nepochs, np.nan)          # flip points scored against latent quality
    piv_audit = np.full(Nepochs, np.nan)         # ... against each participant's own estimate
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
    attack = adv_frac > 0.0
    adv_rng = np.random.default_rng([seed, 1]) if attack else None
    adv = adv_rng.random(n) < adv_frac if attack else np.zeros(n, dtype=bool)
    Q_med[adv] = adv_Q
    adv_share = np.full(Nepochs, np.nan)

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

    def join(Q_new, adv_new, life_new, seed_est):
        # append fresh identities; bandit statistics start empty or at the newcomer prior
        nonlocal Q_med, Q_est, n_obs, bN, bS, alive, active, lifetime, adv
        k = Q_new.size
        n0 = prior_n0 if use_bandit else 0.0
        Q_med = np.concatenate([Q_med, Q_new])
        Q_est = np.concatenate([Q_est, np.full(k, seed_est)])
        n_obs = np.concatenate([n_obs, np.zeros(k, dtype=np.int64)])
        bN = np.concatenate([bN, np.full(k, n0)])
        bS = np.concatenate([bS, np.full(k, n0 * seed_est)])
        alive = np.concatenate([alive, np.ones(k, dtype=bool)])
        active = np.concatenate([active, np.zeros(k, dtype=bool)])
        lifetime = np.concatenate([lifetime, life_new])
        adv = np.concatenate([adv, adv_new])

    for i in range(Nepochs):
        if ou_theta > 0.0:
            Q_med += ou_theta * (mu_Q - Q_med) + ou_step * rng.standard_normal(Q_med.size)
            if attack:
                Q_med[adv] = adv_Q

        ia = np.flatnonzero(alive)
        if ia.size == 0:
            continue

        am = active & alive
        act_idx = np.flatnonzero(am)
        if act_idx.size == 0:
            continue

        T_act = draw(Q_med[act_idx])
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

        if want_pivot:
            pre_est = Q_est.copy()
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

        if want_pivot:
            piv_bias[i], piv_audit[i] = _pivot(mode, Q_est, pre_est, n_obs, ia, act_idx, Q_med, take,
                                               new_active, alpha, alpha_floor, c_ucb, i, noise, V)
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
            obs[nw] = draw(Q_med[new_active[nw]])
        mean_T[i] = float(obs.mean())
        # latent quality of the selected set: no measurement noise, no selection bias
        lat_Q[i] = float(Q_med[new_active].mean())
        churn[i] = float((~was).sum() / max(new_active.size, 1))
        adv_share[i] = float(adv[new_active].mean()) if new_active.size else np.nan

        # ---------------- whitewashing ----------------
        if attack:
            # the abandoned identity is not decremented below, so carry this epoch's tick over;
            # a participant whose lifetime ends now leaves instead of rejoining
            ww = np.flatnonzero(adv & alive & ~active & (n_obs > 0))
            if ww.size:
                rest = lifetime[ww] - 1
                alive[ww] = False
                lifetime[ww] = 0
                ww = ww[rest > 0]; rest = rest[rest > 0]
                if ww.size:
                    join(np.full(ww.size, adv_Q), np.ones(ww.size, dtype=bool), rest, target_inactive)

        # ---------------- attrition ----------------
        lifetime[ia] -= 1
        dead = ia[lifetime[ia] <= 0]
        if dead.size:
            alive[dead] = False
            active[dead] = False

        # ---------------- growth ----------------
        n_new = rng.poisson(pgrowth)
        if n_new > 0:
            Q_new = rng.normal(mu_Q, sigma_Q, n_new)
            a_new = adv_rng.random(n_new) < adv_frac if attack else np.zeros(n_new, dtype=bool)
            Q_new[a_new] = adv_Q
            nl = rng.geometric(p_safe, n_new) if pattr > 0 else np.full(n_new, 10 ** 8)
            join(Q_new, a_new, nl.astype(np.int64), target_inactive)

    if stats is not None:
        stats['adv_share'] = adv_share
        if want_pivot:
            stats['pivot_bias'] = piv_bias
            stats['pivot_audit'] = piv_audit
    return mean_T, P_hist, S_hist, lat_Q, churn, obs_A


def _pivot(mode, Q_est, pre_est, n_obs, ia, act_idx, Q_med, take, new_active, alpha,
           alpha_floor, c_ucb, i, noise, V):
    """Predicted E[M2 - M3 | history] = (1/N') sum_j g(u_j) over the fielded set, where u_j is the
    noise draw at which j's retention flips with every other draw held fixed. Exact for ucb (own
    score only); for static it ignores the 1/N_act effect of T_j on the imputed percentile.
    Returns (prediction from latent quality, audit from the participant's pre-epoch estimate,
    which needs only the scores, the rule and V)."""
    if mode == 'ucb':
        cnt = n_obs[ia]
        score = Q_est[ia] + c_ucb * np.sqrt(np.log(i + 2.0) / np.maximum(cnt, 1))
        score = np.where((cnt == 0) | np.isnan(Q_est[ia]), np.inf, score)
    else:
        if np.isnan(Q_est[ia]).any():
            return np.nan, np.nan
        score = Q_est[ia].copy()
    sv = np.sort(score)[::-1]
    if take >= ia.size:
        return 0.0, 0.0                             # everyone is kept: nobody is pivotal
    kept = np.isin(act_idx, new_active)
    c = np.where(kept, sv[take], sv[take - 1])      # best competitor for the last slot
    pre = pre_est[act_idx]
    if mode == 'ucb':
        bonus = c_ucb * np.sqrt(np.log(i + 2.0) / np.maximum(n_obs[act_idx], 1))
        step = np.maximum(1.0 / np.maximum(n_obs[act_idx], 1), alpha_floor)
        T_star = np.where(np.isnan(pre), c - bonus, pre + (c - bonus - pre) / step)
    else:
        T_star = np.where(np.isnan(pre), c, (c - (1 - alpha) * pre) / alpha)
    T_star = np.where(np.isinf(c), np.inf, T_star)
    qh = pre_est[act_idx]
    fill = np.nanmean(qh) if np.isfinite(qh).any() else 0.0
    out = []
    for q in (Q_med[act_idx], np.where(np.isnan(qh), fill, qh)):
        u = T_star - q
        g = np.where(np.isinf(u), 0.0, noise_g(np.where(np.isinf(u), 0.0, u), noise, V))
        out.append(float(g.sum() / max(new_active.size, 1)))
    return tuple(out)


SCENARIOS = {
    'Fixed':     dict(Ninit=80,  Nact=50, pgrowth=1e-10, pattr=1e-10),
    'Growing':   dict(Ninit=8,   Nact=32, pgrowth=0.1,   pattr=1e-10),
    'Shrinking': dict(Ninit=100, Nact=32, pgrowth=1e-10, pattr=2e-3),
    'Evolving':  dict(Ninit=100, Nact=32, pgrowth=0.1,   pattr=2e-3),
}
