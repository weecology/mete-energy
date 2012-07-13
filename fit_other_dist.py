from __future__ import division
import mpmath
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect
from macroeco_distributions import *
from mete_distributions import *
from math import exp

def xsquare_pdf(x, dist, *pars):
    """Calculates the pdf for x, given the distribution of variable Y = sqrt(X) 
    
    and a given value x. 
    
    """
    x = np.array(x)
    return 1 / x ** 0.5 * dist.pdf(x ** 0.5, *pars) / 2 

def xsquare_ppf(q, dist, *pars):
    """Calculate the ppf for x, given the distribution of variable Y = sqrt(X) 
    
    and a given quantile q.
    
    """
    q = np.array(q)
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
    n           --  number of observations.
    AICc        --  AICc values to be compared. 
    
    """
    AICc_min = min(AICc)
    weight = []
    for AICc_single in AICc:
        weight.append(np.exp(-(AICc_single - min(AICc)) / 2))
    weight = np.array(weight) / sum(np.array(weight))
    return weight

def weights(dat, expon_par, pareto_par, weibull_k, weibull_lmd):
    """Calculates the AICc weights for the four distributions:
    
    truncated expontial, truncated Pareto, truncated Weibull, METE.
    
    """
    dbh_raw = dat[dat.dtype.names[2]]
    dbh_scale = np.array(sorted(dbh_raw / min(dbh_raw)))
    dbh2_scale = dbh_scale ** 2

    N0 = len(dbh2_scale)
    E0 = sum(dbh2_scale)
    S0 = len(set(dat[dat.dtype.names[1]]))
    ll_expon = sum(np.log(xsquare_pdf(dbh2_scale, trunc_expon, expon_par, 1)))
    ll_pareto = sum(np.log(xsquare_pdf(dbh2_scale, trunc_pareto, pareto_par, 1)))
    ll_weibull = sum(np.log(xsquare_pdf(dbh2_scale, trunc_weibull, weibull_k, weibull_lmd, 1)))
    ll_list = [ll_expon, ll_pareto, ll_weibull]
    k_list = [2, 2, 3]
    AICc_list = []
    for i, ll in enumerate(ll_list):
        AICc_dist = AICc(k_list[i], ll, N0)
        AICc_list.append(AICc_dist)
    psi = psi_epsilon(S0, N0, E0)
    ll_psi = 0
    for dbh2 in dbh2_scale:
        ll_psi += log(psi.pdf(dbh2))
    AICc_psi = AICc(3, ll_psi, N0)
    AICc_list.append(AICc_psi)
    return aic_weight_multiple(N0, *AICc_list)

def plot_ind_hist(dat, expon_par, pareto_par, weibull_k, weibull_lmd, title, plot_obj = None, legend = False):
    """Plots the histogram of observed dbh**2, with predicted pdf curves on top.""" 
    dbh_raw = dat[dat.dtype.names[2]]
    dbh_scale = np.array(sorted(dbh_raw / min(dbh_raw)))
    dbh2_scale = dbh_scale ** 2
    E0 = sum(dbh2_scale)
    N0 = len(dbh2_scale)
    S0 = len(set(dat[dat.dtype.names[1]]))

    num_bin = int(ceil(log(max(dbh2_scale)) / log(2))) #Set up log(2) 
    emp_pdf = []
    for i in range(num_bin):
        count = len(dbh2_scale[(dbh2_scale < 2 ** (i + 1)) & (dbh2_scale >= 2 ** i)])
        emp_pdf.append(count / N0 / 2 ** i)
    psi = psi_epsilon(S0, N0, E0)
    psi_pdf = []
    x_array = np.arange(1, ceil(max(dbh2_scale)) + 1)
    for x in x_array:
        psi_pdf.append(psi.pdf(x))
    
    if plot_obj == None:
        plot_obj = plt.figure()
    plot_obj.loglog(x_array, xsquare_pdf(x_array, trunc_expon, expon_par, 1), '#FF4040', 
                    linewidth = 2, label = 'Truncated exponential')
    plot_obj.loglog(x_array, xsquare_pdf(x_array, trunc_pareto, pareto_par, 1), '#00BFFF', 
                    linewidth = 2, label = 'Truncated Pareto')
    plot_obj.loglog(x_array, xsquare_pdf(x_array, trunc_weibull, weibull_k, weibull_lmd, 1), 
                    '#000000', linewidth = 3, label = 'Truncated Weibull')
    plot_obj.loglog(x_array, psi_pdf, '#9400D3', linewidth = 2, label = 'METE')
    if legend:
        plot_obj.legend(loc = 1, prop = {'size': 8})
    plot_obj.bar(2 ** np.array(range(num_bin)), emp_pdf, color = '#d6d6d6', 
            width = 0.4 * 2 ** np.array(range(num_bin)))
    #plot_obj.xlabel(r'$DBH^2$')
    #plot_obj.ylabel('Probability Density')
    return plot_obj