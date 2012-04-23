from __future__ import division
import mpmath
import numpy as np
from scipy.optimize import bisect
from macroeco_distributions import *
from mete_distributions import *
from math import exp

def xsquare_pdf(x, dist, *pars):
    """Calculates the pdf for x**2, given the distribution of variable X 
    
    and a given value x (i.e., f_Y(x ** 2)). 
    
    """
    return 1 / x * dist.pdf(x, *pars) / 2 

def xsquare_ppf(q, dist, *pars):
    """Calculate the ppf for x**2, given the distribution of variable X 
    
    and a given quantile q.
    
    """
    return dist.ppf(q, *pars) ** 2
    
def pred_rank_dbh2_alt(dat, dist, outfile, *pars):
    """Returns predicted rank-dbh**2 given the assumed distribution of dbh."""
    dbh_raw = dat[dat.dtype.names[1]]
    dbh2_scale = sorted(dbh_raw / min(dbh_raw)) ** 2
    dbh2_pred = []
    for i in range(1, len(dbh2_scale) + 1):
        dbh2_pred.append(dist.ppf((i - 0.5) / len(dbh2_scale), *pars))
    out = np.zeros((len(ind_pred), ), dtype = [('pred', 'f8'), ('obs', 'f8')])
    out['pred'] = np.array(dbh2_pred)
    out['obs'] = dbh2_scale
    return out

def AICc(k, L, n):
    """Computes the corrected Akaike Information Criterion. 
    
    Keyword arguments:
    L  --  log likelihood value of given distribution.
    k  --  number of fitted parameters.
    n  --  number of observations.
       
    """
    AICc = 2 * k - 2 * L + 2 * k * (k + 1) / (n - k - 1)
    return AICc

def aic_weight_multiple(n, *AICc):
    """Computes Akaike weight for one model relative to another
    
    Based on information from Burnham and Anderson (2002).
    
    Keyword arguments:
    AICc_dist1  --  AICc for primary model of interest
    AICc_dist2  --  AICc for alternative model
    n           --  number of observations.
    cutoff      --  minimum number of observations required to generate a weight.
    
    """
    AICc_min = min(AICc)
    weight = []
    for AICc_single in AICc:
        weight.append(np.exp(-(AICc_single - min(AICc)) / 2))
    weight = np.array(weight) / sum(np.array(weight))
    return weight

def weights(dat):
    ind_em = dat[dat.dtype.names[1]]
    ind_em = ind_em / min(ind_em)
    E0 = sum(ind_em ** 2)
    N0 = len(ind_em)
    S0 = len(set(dat[dat.dtype.names[0]]))
    dist_list = [expon_pdf, pareto_pdf, weibull_pdf]
    dist_solver_list = [trunc_expon_solver, trunc_pareto_solver, trunc_weibull_solver]
    k_list = [2, 2, 3]
    AICc_list = []
    for i, dist in enumerate(dist_list):
        ll = 0
        obj = dist_square(ind_em, 1, dist, dist_solver_list[i])
        for ind_value in ind_em:
            ll += log(obj.pdf(ind_value))
        AICc_dist = AICc(k_list[i], ll, len(ind_em))
        AICc_list.append(AICc_dist)
    psi = psi_epsilon(S0, N0, E0)
    ll_psi = 0
    for ind_value in ind_em:
        ll_psi += log(psi.pdf(ind_value ** 2))
    AICc_psi = AICc(3, ll_psi, N0)
    AICc_list.append(AICc_psi)
    return aic_weight_multiple(N0, *AICc_list)
            