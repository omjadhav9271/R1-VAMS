import numpy as np
from scipy import stats

def welch(a, b):
    """Welch t-test + Hedges-corrected Cohen's d of a vs b."""
    na, nb = len(a), len(b)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    sp = np.sqrt(((na-1)*a.var(ddof=1) + (nb-1)*b.var(ddof=1)) / (na+nb-2))
    d = (a.mean() - b.mean()) / sp
    J = 1 - 3/(4*(na+nb) - 9)                     # Hedges' g small-sample correction
    return float(t), float(p), float(d*J)

def tost(a, b, eps):
    """Two one-sided tests. H0: |mu_a - mu_b| >= eps.  p_tost<0.05 => equivalent."""
    na, nb = len(a), len(b)
    diff = a.mean() - b.mean()
    va, vb = a.var(ddof=1)/na, b.var(ddof=1)/nb
    se = np.sqrt(va + vb)
    df = (va+vb)**2 / (va**2/(na-1) + vb**2/(nb-1))
    p1 = stats.t.sf((diff + eps)/se, df)          # H0: diff <= -eps
    p2 = stats.t.cdf((diff - eps)/se, df)         # H0: diff >= +eps
    tc = stats.t.ppf(0.975, df)
    return float(max(p1, p2)), float(diff), float(diff - tc*se), float(diff + tc*se)

def power_for(d, n, alpha=0.05):
    """Achieved power of a two-sample t-test at effect size d, n per group."""
    from scipy.stats import nct, t as tdist
    df = 2*n - 2
    ncp = d * np.sqrt(n/2)
    crit = tdist.ppf(1 - alpha/2, df)
    return float(nct.sf(crit, df, ncp) + nct.cdf(-crit, df, ncp))

def n_needed(d, alpha=0.05, target=0.80):
    for n in range(5, 5001):
        if power_for(d, n, alpha) >= target:
            return n
    return None

def paired(a, b):
    """Paired t-test + paired Cohen's d (same seeds => much higher power)."""
    from scipy import stats as _s
    d_ = a - b
    t, p = _s.ttest_rel(a, b)
    dz = d_.mean() / d_.std(ddof=1)
    return float(t), float(p), float(dz)

def tost_paired(a, b, eps):
    """Paired TOST. H0: |mean difference| >= eps."""
    from scipy import stats as _s
    d_ = a - b; n = len(d_)
    m = d_.mean(); se = d_.std(ddof=1)/np.sqrt(n); df = n-1
    p1 = _s.t.sf((m + eps)/se, df); p2 = _s.t.cdf((m - eps)/se, df)
    tc = _s.t.ppf(0.975, df)
    return float(max(p1, p2)), float(m), float(m - tc*se), float(m + tc*se)
