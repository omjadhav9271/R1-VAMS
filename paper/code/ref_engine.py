"""
Reference engine: verbatim transcription of run_sortition() from adaptive_vams_v2.ipynb,
the loop-based engine every notebook result was produced with.

The only change is an extra return value, obs_A = mean(T_act) per epoch (M1). It is
computed from T_act after it is drawn and consumes no randomness, so the RNG stream --
and therefore mean_T (M2) -- is identical to the notebook's.
"""
import numpy as np
from scipy.special import expit
from collections import deque


def run_sortition(
    Ninit, Nact, Nepochs=1000,
    mu_Q=0.2, sigma_Q=0.1, V=0.2, alpha=0.1,
    pgrowth=1e-10, pattr=1e-10,
    mode='static',
    P_static=25.0,
    P_min=10.0, P_max=40.0, k=20.0, window=50,
    seed=42,
):
    rng        = np.random.default_rng(seed)
    lambda_pen = 2.0

    n        = Ninit
    Q_med    = rng.normal(mu_Q, sigma_Q, n)
    Q_ema    = np.full(n, np.nan)
    active   = np.zeros(n, dtype=bool)
    alive    = np.ones(n,  dtype=bool)
    p_safe   = max(pattr, 1e-15)
    lifetime = rng.geometric(p_safe, n) if pattr > 0 else np.full(n, 10**8)

    mean_T        = []
    P_history     = []
    sigma_history = []
    obs_A         = []
    sigma_buf     = deque(maxlen=window)

    idx_alive = np.where(alive)[0]
    active[rng.choice(idx_alive, size=min(Nact, len(idx_alive)), replace=False)] = True

    for i in range(Nepochs):
        idx_alive = np.where(alive)[0]
        if len(idx_alive) == 0:
            mean_T.append(np.nan); P_history.append(np.nan); sigma_history.append(np.nan)
            obs_A.append(np.nan)
            continue

        T = np.full(n, np.nan)
        for j in idx_alive:
            if active[j]:
                T[j] = rng.normal(Q_med[j], V)

        T_act = T[active & alive]
        T_act = T_act[~np.isnan(T_act)]
        if len(T_act) == 0:
            mean_T.append(np.nan); P_history.append(np.nan); sigma_history.append(np.nan)
            obs_A.append(np.nan)
            continue
        obs_A.append(float(np.mean(T_act)))

        min_T = float(np.min(T_act))
        std_T = float(np.std(T_act)) if len(T_act) > 1 else 0.0
        sigma_history.append(std_T)

        if mode == 'random':
            P_t = 25.0
        elif mode == 'static':
            P_t = P_static
        elif mode == 'adaptive':
            sigma_buf.append(std_T)
            sigma_ref = float(np.mean(sigma_buf))
            P_t = P_min + (P_max - P_min) * float(expit(k * (std_T - sigma_ref)))

        P_history.append(P_t)
        target_inactive = float(np.percentile(T_act, P_t))

        for j in idx_alive:
            if mode != 'random':
                if active[j]:
                    hat_Q = (min_T - lambda_pen * std_T) if np.isnan(T[j]) else T[j]
                else:
                    hat_Q = target_inactive
                Q_ema[j] = hat_Q if np.isnan(Q_ema[j]) else alpha * hat_Q + (1-alpha) * Q_ema[j]

        if mode == 'random':
            new_active = set(rng.choice(idx_alive, size=min(Nact, len(idx_alive)), replace=False).tolist())
        else:
            ema_v = Q_ema[idx_alive]; valid = ~np.isnan(ema_v); v_idx = idx_alive[valid]
            if len(v_idx) <= Nact:
                new_active = set(v_idx.tolist())
                no_e = idx_alive[~valid].tolist()
                need = Nact - len(new_active)
                if need > 0 and no_e:
                    new_active.update(rng.choice(no_e, size=min(need, len(no_e)), replace=False).tolist())
            else:
                sd  = v_idx[np.argsort(ema_v[valid])[::-1]]
                cut = Q_ema[sd[Nact-1]]
                ab  = [j for j in sd[:Nact] if Q_ema[j] > cut]
                ti  = [j for j in v_idx if Q_ema[j] == cut]
                nd  = Nact - len(ab)
                if nd > 0 and ti:
                    ab += rng.choice(ti, size=min(nd, len(ti)), replace=False).tolist()
                new_active = set(ab)

        active[:] = False
        for j in new_active: active[j] = True

        T_rec = [T[j] if not np.isnan(T[j]) else rng.normal(Q_med[j], V) for j in new_active]
        mean_T.append(float(np.mean(T_rec)) if T_rec else np.nan)

        for j in idx_alive:
            lifetime[j] -= 1
            if lifetime[j] <= 0: alive[j] = False; active[j] = False

        n_new = rng.poisson(pgrowth)
        if n_new > 0:
            Q_med    = np.concatenate([Q_med,    rng.normal(mu_Q, sigma_Q, n_new)])
            Q_ema    = np.concatenate([Q_ema,    np.full(n_new, target_inactive)])
            alive    = np.concatenate([alive,    np.ones(n_new,  dtype=bool)])
            active   = np.concatenate([active,   np.zeros(n_new, dtype=bool)])
            lifetime = np.concatenate([lifetime, rng.geometric(p_safe, n_new) if pattr > 0 else np.full(n_new, 10**8)])
            n = len(Q_med)

    return np.array(mean_T), np.array(P_history), np.array(sigma_history), np.array(obs_A)
