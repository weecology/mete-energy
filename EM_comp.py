"""Compare METE energy prediction to data"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import random
import csv
from mete_distributions import *
from EM_dist import *
from macroecotools import *

def power_transform(dat, pw, outfile):
    """Use power-transformed diameter as constraint in METE. 
    
    dat - numpy array with two columns, species and individual-level energy/body mass
    pw - exponent
    outfile - output file name if desired
    
    """
    spp_list = dat[dat.dtype.names[0]]
    em_list = dat[dat.dtype.names[1]] ** pw 
    em_list = sorted(em_list) / min(em_list) # Standardization
    N0 = len(spp_list)
    S0 = len(set(spp_list))
    E0 = sum(em_list)
    psi_epsilon_obj = psi_epsilon(S0, N0, E0)
    
    out = open(outfile, 'wb')
    out_writer = csv.writer(out)
    ind_mr = [] # Result piece
    for i in range(1, N0 + 1):
        ind_mr.append(psi_epsilon_obj.ppf((i - 0.5) / N0))
        if i % 1000 == 0: # Write to file every 1000 rows to avoid crash
            out_piece = np.column_stack((ind_mr, em_list[(i - 1000):i]))
            out_writer.writerows(out_piece)
            ind_mr = []
    if len(ind_mr) > 0: 
        out_piece = np.column_stack((ind_mr, em_list[(i - i % 1000):i]))
        out_writer.writerows(out_piece)
    out.close()
    return None

def plot_rank(dat, title, outfig = False):
    """Plot the predicted versus observed rank energy/body mass distribution.
    
    Input: 
    dat - numpy array with two columns with predicted and observed epsilon,
          same format as the output from power_transform
    title - string, title for the plot
    outfig - optional, output file for the figure if desired
    
    """
    plt.figure()
    ind_pred = dat[dat.dtype.names[0]]
    ind_em = dat[dat.dtype.names[1]]
    plt.loglog([1, 1.1 * max(ind_em)], [1, 1.1 * max(ind_em)])
    plt.scatter(ind_pred, ind_em)    
    plt.xlabel("Predicted")
    plt.ylabel("Observed")
    plt.title(title)
    if outfig:
        plt.savefig(outfig)
    return None    

def plot_species_EM(dat, title, outfig = False, alt = False):
    """Plot the expected versus observed value of species-level energy or biomass
    
    when the corresponding constraing is used. 
    
    alt - if True, returns a plot with abundance against species-level total.
          if False, returns a plot with species-level average against abundance.
    
    """
    plt.figure()
    spp_list = set(dat[dat.dtype.names[0]])
    em_list = dat[dat.dtype.names[1]]
    rescale = min(em_list)
    em_list = np.array(em_list) / rescale
    N0 = len(em_list)
    S0 = len(spp_list)
    E0 = sum(em_list)
    theta_epsilon_obj = theta_epsilon(S0, N0, E0)
    em_obs = []
    n_obs = []
    em_obs_avg = []
    for spp in spp_list:
        dat_spp = dat[dat[dat.dtype.names[0]] == spp]
        n = len(dat_spp)
        em_intra = dat_spp[dat_spp.dtype.names[1]]
        em_intra_sum = sum(em_intra) / rescale
        em_obs.append(em_intra_sum)
        em_obs_avg.append(em_intra_sum / n)
        n_obs.append(n)
    em_pred = []
    em_pred_avg = []
    for n in sorted(n_obs):
        em_pred.append(n * theta_epsilon_obj.E(n))
        em_pred_avg.append(theta_epsilon_obj.E(n))
    if alt:
        plt.loglog(em_pred_avg, sorted(n_obs))
        plt.scatter(em_obs_avg, n_obs)
        plt.xlabel('Species-level average')
        plt.ylabel('Abundance')
    else:
        plt.loglog(sorted(n_obs), em_pred)
        plt.scatter(n_obs, em_obs)
        plt.xlabel('Abundance')
        plt.ylabel('Species-level total')
    plt.title(title)
    if outfig: 
        plt.savefig(outfig)
    return None

def plot_species_avg(dat, title, outfig = False):
    """Plot the expected versus observed value of species-level energy or biomass
    
    when the corresponding constraing is used. 
    
    """
    plt.figure()
    spp_list = set(dat[dat.dtype.names[0]])
    em_list = dat[dat.dtype.names[1]]
    rescale = min(em_list)
    N0 = len(em_list)
    S0 = len(spp_list)
    E0 = sum(em_list) / rescale
    theta_epsilon_obj = theta_epsilon(S0, N0, E0)
    e_spp_level = np.zeros((S0, ), dtype=[('abd','i4'), ('MR_total', 'f8')])
    em_obs = []
    n_obs = []
    for spp in set(spp_list):
        dat_spp = dat[dat[dat.dtype.names[0]] == spp]
        n_obs.append(len(dat_spp))
        em_obs.append(sum(dat_spp[dat.dtype.names[1]]) / rescale)
    e_spp_level['abd'] = np.array(n_obs)
    e_spp_level['MR_total'] = np.array(em_obs)
    num_bin = int(ceil(log(max(e_spp_level['abd'])) / log(2)) + 1) #Set up log(2) bins
    n_avg = []
    em_avg = []
    em_pred = []
    for i in range(num_bin + 1):
        record = e_spp_level[(e_spp_level['abd'] <= 2 ** i) & (e_spp_level['abd'] > 2 ** (i - 1))]
        if len(record) > 0: 
            n_avg.append(np.mean(record['abd']))
            em_avg.append(sum(record['MR_total']) / sum(record['abd']))
    for n in n_avg:
        em_pred.append(theta_epsilon_obj.E(n))
    plt.loglog(n_avg, em_pred)
    plt.loglog(n_avg, em_avg)
    plt.xlabel('Abundance')
    plt.ylabel('Within species average')
    plt.title(title)
    if outfig:
        plt.savefig(outfig)
    return None
     
def plot_spp_frequency(dat, spp_name, title, outfig):
    """Plot the predicted vs. observed frequency distribution of energy or body mass for a specific species."""
    plt.figure()
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

def group_ind_to_spp(dat):
    """Sub-function called in ind_allo_null_comp. Returns a list of total energy consumption within each species."""
    spp_list = set(dat['spp'])
    out_list = []
    for spp in spp_list:
        dat_spp = dat[dat['spp'] == spp]
        out_list.append(sum(dat_spp[dat_spp.dtype.names[1]]))
    return out_list

def ind_allo_null_comp(dat, Niter = 5000):
    """Compare the empirical allocation of individuals with different
    
    energy consumption to species with random allocations.
    
    Input:
    dat - data file in the same format as plot_e or plot_m, with one column for 
    species identity and one column for some measure of energy consumption.
    Niter - number of randomizations.
    
    """
    E_avg = sum(dat[dat.dtype.names[1]]) / len(set(dat['spp']))
    expected = [E_avg for i in range(len(set(dat['spp'])))]
    r2_emp = obs_pred_rsquare(np.array(group_ind_to_spp(dat)), np.array(expected))
    count = 0
    for i in range(Niter):
        random.shuffle(dat['spp'])
        r2_rand = obs_pred_rsquare(np.array(group_ind_to_spp(dat)), np.array(expected))
        if r2_rand < r2_emp: 
            count += 1
    return count / Niter

def plot_spp_exp(dat, title, threshold = 5, outfig = False):
    """Plot the MLE exponential parameter for each species against abundance n, 
    
    for all species with abundance higher than the threshold.
    
    """
    plt.figure()
    spp_list = set(dat[dat.dtype.names[0]])
    rescale = min(dat[dat.dtype.names[1]])
    n_list = []
    exp_list = []
    for spp in spp_list:
        dat_spp = dat[dat[dat.dtype.names[0]] == spp]
        em_intra = dat_spp[dat_spp.dtype.names[1]] / rescale
        n = len(em_intra)
        if n >= threshold:
            n_list.append(n)
            exp_list.append(1 / (np.mean(em_intra) - 1))
    plt.semilogx(n_list, exp_list, 'bo')
    plt.xlabel('Abundance')
    plt.ylabel('Parameter of exponential distribution')
    plt.title(title)
    if outfig:
        plt.savefig(outfig)
    return None
    
