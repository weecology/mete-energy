"""Compare METE energy prediction to data"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from mete_distributions import *
from EM_dist import *

def pred_rank(S0, N0, E0):
    """Returns the predicted metabolic rate for each individual"""
    ind_mr = []
    psi_epsilon_obj = psi_epsilon(S0, N0, E0)
    for i in range(1, N0 + 1):
        ind_mr.append(psi_epsilon_obj.ppf((i - 0.5) / N0))
    return ind_mr

def plot_rank(dat, title, outfig, outfile = None):
    """Plot the predicted versus observed rank energy/body mass distribution.
    
    Input: 
    dat - numpy array with two columns, species and individual-level energy/body mass
    title - string, title for the plot
    outfig - figure output
    outfile - output file name if desired
    
    """
    spp_list = []
    em_list = []
    for row in dat:
        spp_list.append(row[0])
        em_list.append(row[1])
    em_list = np.array(em_list) / min(em_list) # Standardization
    N0 = len(spp_list)
    S0 = len(set(spp_list))
    E0 = sum(em_list)
    plt.loglog([1, 1.1 * max(em_list)], [1, 1.1 * max(em_list)])
    ind_pred = pred_rank(S0, N0, E0)
    plt.scatter(ind_pred, sorted(em_list))    
    plt.xlabel("Predicted")
    plt.ylabel("Observed")
    plt.title(title)
    plt.savefig(outfig)
    if outfile:
        out = np.array([[ind_pred[i], sorted(em_list)[i]] for i in range(len(ind_pred))])
        np.savetxt(outfile, out, delimiter = ",")        

def plot_species_EM(dat, title, outfig):
    """Plot the expected versus observed value of species-level energy or biomass
    
    when the corresponding constraing is used. 
    
    """
    spp_list = []
    em_list = []
    for row in dat:
        spp_list.append(row[0])
        em_list.append(row[1])
    rescale = min(em_list)
    em_list = np.array(em_list) / rescale
    N0 = len(spp_list)
    S0 = len(set(spp_list))
    E0 = sum(em_list)
    theta_epsilon_obj = theta_epsilon(S0, N0, E0)
    em_obs = []
    n_obs = []
    for spp in set(spp_list):
        dat_spp = dat[dat['spp'] == spp]
        n = len(dat_spp)
        em_intra = []
        for ind_spp in dat_spp: 
            em_intra.append(ind_spp[1])
        em_intra_sum = sum(em_intra) /  rescale
        em_obs.append(em_intra_sum)
        n_obs.append(n)
    em_pred = []
    for n in range(1, max(n_obs) + 1):
        em_pred.append(n * theta_epsilon_obj.E(n))
    plt.loglog(range(1, max(n_obs) + 1), em_pred)
    plt.scatter(n_obs, em_obs)
    plt.axis([0.9, 1.1 * max(n_obs), 0.9, 1.1 * max(em_obs)])
    plt.xlabel('Abundance')
    plt.ylabel('Species-level energy or biomass')
    plt.title(title)
    plt.savefig(outfig)
    
def plot_spp_frequency(dat, spp_name, title, outfig):
    """Plot the predicted vs. observed frequency distribution of energy or body mass for a specific species."""
    dat_spp = dat[dat['spp'] == spp_name]
    n = len(dat_spp)
    spp_list = []
    em_list = []
    em_list_spp = []
    for row in dat:
        spp_list.append(row[0])
        em_list.append(row[1])
    rescale = min(em_list)
    for row in dat_spp:
        em_list_spp.append(row[1])
    em_list_spp = np.array(em_list_spp) / rescale
    em_list = np.array(em_list) / rescale
    N0 = len(spp_list)
    S0 = len(set(spp_list))
    E0 = sum(em_list)
    theta_epsilon_obj = theta_epsilon(S0, N0, E0)
    f_pred = []
    for i in np.arange(1, max(np.array(em_list) / rescale) + 1):
        f_pred.append(theta_epsilon_obj.pdf(i, n))
    plt.semilogx(np.arange(1, max(np.array(em_list) / rescale) + 1), f_pred)
    bins = np.exp(np.arange(log(min(em_list_spp)), log(max(em_list_spp) + 1),
                            (log(max(em_list_spp)) - log(min(em_list_spp))) / 10))
    bin_width = []
    for i in range(len(bins) - 1):
        bin_width.append(bins[i + 1] - bins[i])
    count, bins_log, patches = plt.hist(np.log(em_list_spp), bins = np.log(bins), visible = False)
    plt.semilogx(bins[:-1], count / bin_width / n, 'r')
    plt.axis([1, n + 1, 0, max(count / bin_width / n) * 1.1])
    plt.xlabel('Energy or body mass')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.savefig(outfig)