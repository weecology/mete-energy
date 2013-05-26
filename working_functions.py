"""Module with all working functions for the METE energy project"""

from __future__ import division
import numpy as np
import csv
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from mete import *
import macroecotools
import macroeco_distributions as mdis
from fit_other_dist import *
from math import exp
from scipy.stats.mstats import mquantiles
from scipy.stats import ks_2samp

def import_raw_data(input_filename):
    data = np.genfromtxt(input_filename, dtype = "S15, S25, f8", skiprows = 1, 
                         names = ['site', 'sp', 'dbh'], delimiter = ",")
    return data

def import_obs_pred_data(input_filename):
    data = np.genfromtxt(input_filename, dtype = "S15,f8,f8",
                                  names = ['site','obs','pred'],
                                  delimiter = ",")
    return data

def import_par_table(par_table):
    """Import the csv file containing parameter estimates for all datasets."""
    par_table = np.genfromtxt(par_table, dtype = "S15, S15, f8, f8, f8, f8", skiprows = 1, 
                              names = ['dataset', 'site', 'expon_par', 'pareto_par', 
                                       'weibull_k', 'weibull_lmd'], delimiter = ",")
    return par_table
            
def get_obs_pred_rad(raw_data, dataset_name, data_dir='./data/', cutoff = 9):
    """Use data to compare the predicted and empirical SADs and get results in csv files
    
    (Copied and modified from the funciton 'run_test' from White et al. 2012)
    Keyword arguments:
    raw_data : numpy structured array with 3 columns: 'site','sp','dbh'
    dataset_name : short code that will indicate the name of the dataset in
                    the output file names
    data_dir : directory in which to store output
    cutoff : minimum number of species required to run - 1.
    
    """
    
    usites = np.sort(list(set(raw_data["site"])))
    f1_write = open(data_dir + dataset_name + '_obs_pred_rad.csv', 'wb')
    f1 = csv.writer(f1_write)
    
    for i in range(0, len(usites)):
        subsites = raw_data["site"][raw_data["site"] == usites[i]]        
        subsp = raw_data["sp"][raw_data["site"] == usites[i]]
        N = len(subsp)
        S = len(set(subsp))
        subab = []
        for sp in set(subsp):
            subab.append(len(subsp[subsp == sp]))
        if S > cutoff:
            print("%s, Site %s, S=%s, N=%s" % (dataset_name, i, S, N))
            # Generate predicted values and p (e ** -beta) based on METE:
            mete_pred = get_mete_rad(int(S), int(N))
            pred = np.array(mete_pred[0])
            obsab = np.sort(subab)[::-1]
            #save results to a csv file:
            results = np.zeros((len(obsab), ), dtype = ('S10, i8, i8'))
            results['f0'] = np.array([usites[i]] * len(obsab))
            results['f1'] = obsab
            results['f2'] = pred
            f1.writerows(results)
    f1_write.close()

def get_mete_pred_cdf(dbh2_scale, S0, N0, E0):
    """Compute the cdf of the individual metabolic rate (size) distribution
    
    predicted by METE given S0, N0, E0, and scaled dbh**2.
    
    """
    pred_cdf = []
    psi = psi_epsilon(S0, N0, E0)
    for dbh2 in dbh2_scale:
        pred_cdf.append(psi.cdf(dbh2))
    return np.array(pred_cdf)

def get_mete_pred_dbh2(dbh2_scale, S0, N0, E0):
    """Compute the individual metabolic rate (size) predicted by METE 
    
    for each individual given S0, N0, E0, and scaled dbh**2.
    
    """
    psi = psi_epsilon(S0, N0, E0)
    scaled_rank = [(x + 0.5) / len(dbh2_scale) for x in range(len(dbh2_scale))]
    pred_dbh2 = [psi.ppf(x) for x in scaled_rank]
    return np.array(pred_dbh2)

def get_obs_cdf(dat):
    """Compute the empirical cdf given a list or an array"""
    dat = np.array(dat)
    emp_cdf = []
    for point in dat:
        point_cdf = len(dat[dat < point]) / len(dat)
        emp_cdf.append(point_cdf)
    return np.array(emp_cdf)

def get_obs_pred_cdf(raw_data, dataset_name, data_dir = './data/', cutoff = 9):
    """Use data to compare the predicted and empirical CDFs of the 
    
    individual metabolic rate distribution and get results in csv files.
    Keyword arguments:
    raw_data : numpy structured array with 3 columns: 'site','sp','dbh'
    dataset_name : short code that will indicate the name of the dataset in
                    the output file names
    data_dir : directory in which to store output
    cutoff : minimum number of species required to run - 1.
    
    """
    usites = np.sort(list(set(raw_data["site"])))
    f1_write = open(data_dir + dataset_name + '_obs_pred_isd_cdf.csv', 'wb')
    f1 = csv.writer(f1_write)
    
    for site in usites:
        subdat = raw_data[raw_data["site"] == site]
        dbh_raw = subdat[subdat.dtype.names[2]]
        dbh_scale = np.array(sorted(dbh_raw / min(dbh_raw)))
        dbh2_scale = dbh_scale ** 2
        E0 = sum(dbh2_scale)
        N0 = len(dbh2_scale)
        S0 = len(set(subdat[subdat.dtype.names[1]]))
        if S0 > cutoff:
            cdf_pred = get_mete_pred_cdf(dbh2_scale, S0, N0, E0)
            cdf_obs = get_obs_cdf(dbh2_scale)
            #save results to a csv file:
            results = np.zeros((len(cdf_obs), ), dtype = ('S10, f8, f8'))
            results['f0'] = np.array([usites[i]] * len(cdf_obs))
            results['f1'] = cdf_obs
            results['f2'] = cdf_pred
            f1.writerows(results)
    f1_write.close()
    
def get_obs_pred_dbh2(raw_data, dataset_name, data_dir = './data/', cutoff = 9):
    """Use data to compare the predicted and empirical dbh**2 of individuals and 
    
    get results in csv files.
    Keyword arguments:
    raw_data : numpy structured array with 3 columns: 'site','sp','dbh'
    dataset_name : short code that will indicate the name of the dataset in
                    the output file names
    data_dir : directory in which to store output
    cutoff : minimum number of species required to run - 1.
    
    """
    usites = np.sort(list(set(raw_data["site"])))
    f1_write = open(data_dir + dataset_name + '_obs_pred_isd_dbh2.csv', 'wb')
    f1 = csv.writer(f1_write)
    
    for site in usites:
        subdat = raw_data[raw_data["site"] == site]
        dbh_raw = subdat[subdat.dtype.names[2]]
        dbh_scale = np.array(dbh_raw / min(dbh_raw))
        dbh2_scale = dbh_scale ** 2
        E0 = sum(dbh2_scale)
        N0 = len(dbh2_scale)
        S0 = len(set(subdat[subdat.dtype.names[1]]))
        if S0 > cutoff:
            dbh2_pred = get_mete_pred_dbh2(dbh2_scale, S0, N0, E0)
            dbh2_obs = sorted(dbh2_scale)
            #save results to a csv file:
            results = np.zeros((len(dbh2_obs), ), dtype = ('S10, f8, f8'))
            results['f0'] = np.array([usites[i]] * len(dbh2_obs))
            results['f1'] = dbh2_obs
            results['f2'] = dbh2_pred
            f1.writerows(results)
    f1_write.close()
    
def get_obs_pred_frequency(raw_data, dataset_name, data_dir = './data/', bin_size = 1.7, cutoff = 9):
    """Use data to compare the predicted and empirical frequency for each size bins
    
    and store results in csv files.
    Keyword arguments:
    raw_data : numpy structured array with 3 columns: 'site','sp','dbh'
    dataset_name : short code that will indicate the name of the dataset in
                    the output file names
    data_dir : directory in which to store output
    bin_size: power bin size for dbh ** 2, e.g., default value 1.7 means that bins for dbh ** 2 run from 1 to 1.7, 
              1.7 to 1.7 **2, etc. Default 1.7 ensures that there are at least 10 bins for each dataset in our analysis.
    cutoff : minimum number of species required to run - 1.
    
    """
    usites = np.sort(list(set(raw_data["site"])))
    f = open(data_dir + dataset_name + '_obs_pred_freq.csv', 'wb')
    f_writer = csv.writer(f)
    
    for site in usites:
        subdat = raw_data[raw_data["site"] == site]
        dbh_raw = subdat[subdat.dtype.names[2]]
        dbh_scale = np.array(sorted(dbh_raw) / min(dbh_raw))
        dbh2_scale = dbh_scale ** 2
        E0 = sum(dbh2_scale)
        N0 = len(dbh2_scale)
        S0 = len(set(subdat[subdat.dtype.names[1]]))
        if S0 > cutoff:
            psi = psi_epsilon(S0, N0, E0)
            # Plot histogram with pdf
            num_bin = int(ceil(log(max(dbh2_scale)) / log(bin_size)))
            freq_obs = []
            freq_pred = []
            for i in range(num_bin):
                count = len(dbh2_scale[(dbh2_scale < bin_size ** (i + 1)) & (dbh2_scale >= bin_size ** i)])
                freq_obs.append(count / N0 / (bin_size ** i * (bin_size - 1))) #  Divided by the interval between two ticks
                freq_pred.append((psi.cdf(bin_size ** (i + 1)) - 
                                  psi.cdf(bin_size ** i)) / (bin_size ** i * (bin_size - 1)))
            #save results to a csv file:
            results = np.zeros((len(freq_obs), ), dtype = ('S10, f8, f8'))
            results['f0'] = np.array([site] * len(freq_obs))
            results['f1'] = freq_obs
            results['f2'] = freq_pred
            f_writer.writerows(results)
    f.close()
        
def get_obs_pred_intradist(raw_data, dataset_name, data_dir = './data/', cutoff = 9, n_cutoff = 4):
    """Compare the predicted and empirical average dbh^2 as well as compute the scaled 
    
    intra-specific energy distribution for each species and get results in csv files.
    Keyword arguments:
    raw_data : numpy structured array with 3 columns: 'site','sp','dbh'
    dataset_name : short code that will indicate the name of the dataset in
                    the output file names
    data_dir : directory in which to store output
    cutoff : minimum number of species required to run - 1.
    
    """
    usites = np.sort(list(set(raw_data["site"])))
    f1_write = open(data_dir + dataset_name + '_obs_pred_avg_mr.csv', 'wb')
    f1 = csv.writer(f1_write)
    f2_write = open(data_dir + dataset_name + '_par.csv', 'wb')
    f2 = csv.writer(f2_write)
    
    for site in usites:
        subdat = raw_data[raw_data["site"] == site]
        dbh_raw = subdat[subdat.dtype.names[2]]
        dbh_scale = np.array(dbh_raw / min(dbh_raw))
        dbh2_scale = dbh_scale ** 2
        E0 = sum(dbh2_scale)
        N0 = len(dbh2_scale)
        S_list = set(subdat[subdat.dtype.names[1]])
        S0 = len(S_list)
        if S0 > cutoff:
            mr_avg_pred = []
            mr_avg_obs = []
            par_pred = []
            par_obs = []
            psi = psi_epsilon(S0, N0, E0)
            theta_epsilon_obj = theta_epsilon(S0, N0, E0)
            for sp in S_list:
                sp_dbh2 = dbh2_scale[subdat[subdat.dtype.names[1]] == sp]
                mr_avg_obs.append(sum(sp_dbh2) / len(sp_dbh2))
                mr_avg_pred.append(theta_epsilon_obj.E(len(sp_dbh2)))
                if len(sp_dbh2) > n_cutoff: 
                    par_pred.append(len(sp_dbh2) * psi.lambda2)
                    par_obs.append(1 / (sum(sp_dbh2) / len(sp_dbh2) - 1))        
            #save results to a csv file:
            results1 = np.zeros((len(mr_avg_pred), ), dtype = ('S10, f8, f8'))
            results1['f0'] = np.array([site] * len(mr_avg_pred))
            results1['f1'] = np.array(mr_avg_obs)
            results1['f2'] = np.array(mr_avg_pred)
            f1.writerows(results1)
            
            results2 = np.zeros((len(par_pred), ), dtype = ('S10, f8, f8'))
            results2['f0'] = np.array([site] * len(par_pred))
            results2['f1'] = np.array(par_obs)
            results2['f2'] = np.array(par_pred)
            f2.writerows(results2)
    f1_write.close()
    f2_write.close()
    
def ks_test_sp(sp_dbh2, Nsim = 1000, p = 0.05):
    """Kolmogorov-Smirnov test to evaluate if dbh2 of one species is significantly 
    
    different from an exponential distribution left truncated at 1.
    MLE parameter is obtained from the original sample, Nsim samples are simulated
    from a truncated exponential distribution with the MLE parameter, and the original
    sample is compared to each simulated sample with the 2-sample KS test.
    Keyword arguments:
    dbh2_sp: a vector (array) of rescaled dbh2 value for one species.
    Nsim: number of simulated samples
    p: level of significance
    
    """
    par_mle = 1 / (sum(sp_dbh2) / len(sp_dbh2) - 1)
    count = 0
    for i in range(Nsim):
        sim_dbh2 = mdis.trunc_expon.rvs(par_mle, 1, size = len(sp_dbh2))
        ks_test = ks_2samp(sp_dbh2, sim_dbh2)
        if ks_test[1] <= p:
            count += 1
    return count / Nsim

def ks_test(datasets, data_dir = './data/', Nsim = 1000, p = 0.05, cutoff = 9, n_cutoff = 4):
    """Kolmogorov-Smirnov test for each species in each dataset."""
    f_write = open(data_dir + 'ks_test_' + str(Nsim) + '_' + str(p) + '.csv', 'wb')
    f = csv.writer(f_write)
    
    for dataset in datasets:
        raw_data = import_raw_data(dataset + '.csv')
        usites = np.sort(list(set(raw_data['site'])))
        for site in usites:
            subdat = raw_data[raw_data['site'] == site]
            dbh_raw = subdat[subdat.dtype.names[2]]
            dbh_scale = np.array(dbh_raw / min(dbh_raw))
            dbh2_scale = dbh_scale ** 2
            S_list = set(subdat[subdat.dtype.names[1]])
            S0 = len(S_list)
            if S0 > cutoff: 
                sp_list = []
                sp_p_list = []
                for sp in S_list:
                    sp_dbh2 = dbh2_scale[subdat['sp'] == sp]
                    if len(sp_dbh2) >= n_cutoff:
                        sp_list.append(sp)
                        sp_p_list.append(ks_test_sp(sp_dbh2, Nsim, p))
            # Save to output
            results = np.zeros((len(sp_list), ), dtype = ('S10, S10, S10, f8'))
            results['f0'] = np.array([dataset] * len(sp_list))
            results['f1'] = np.array([site] * len(sp_list))
            results['f2'] = np.array(sp_list)
            results['f3'] = np.array(sp_p_list)
            f.writerows(results)
    f_write.close()
    
def species_rand_test(datasets, data_dir = './data/', cutoff = 9, n_cutoff = 4, Niter = 200):
    """Randomize species identity within sites and compare the r-square obtained 
    
    for abundance-size distribution and intraspecific energy distribution parameter
    of empirical data versus randomized data.
    Keyword arguments:
    datasets - a list of dataset names
    data_dir - directory for output file
    Niter - number of randomizations applying to each site in each dataset
             
    """
    f1_write = open(data_dir + 'mr_rand_sites.csv', 'wb')
    f2_write = open(data_dir + 'lambda_rand_sites.csv', 'wb')
    f1 = csv.writer(f1_write)
    f2 = csv.writer(f2_write)
    
    for dataset in datasets:
        raw_data = import_raw_data(dataset + '.csv')
        usites = np.sort(list(set(raw_data['site'])))
        for site in usites:
            subdat = raw_data[raw_data['site'] == site]
            dbh_raw = subdat[subdat.dtype.names[2]]
            dbh_scale = np.array(dbh_raw / min(dbh_raw))
            dbh2_scale = dbh_scale ** 2
            E0 = sum(dbh2_scale)
            N0 = len(dbh2_scale)
            S_list = set(subdat[subdat.dtype.names[1]])
            S0 = len(S_list)
            if S0 > cutoff: 
                mr_out_row = [dataset + '_' + site]
                lambda_out_row = [dataset + '_' + site]
                psi = psi_epsilon(S0, N0, E0)
                theta_epsilon_obj = theta_epsilon(S0, N0, E0)
                for i in range(Niter + 1):
                    if i != 0: #Shuffle species label except for the first time
                        np.random.shuffle(subdat[subdat.dtype.names[1]]) 
                    par_pred = []
                    mr_avg_obs = []
                    par_obs = []
                    mr_avg_pred = []
                    for sp in S_list:
                        sp_dbh2 = dbh2_scale[subdat[subdat.dtype.names[1]] == sp]
                        mr_avg_obs.append(sum(sp_dbh2) / len(sp_dbh2))
                        mr_avg_pred.append(theta_epsilon_obj.E(len(sp_dbh2)))
                        if len(sp_dbh2) > n_cutoff: 
                            par_pred.append(len(sp_dbh2) * psi.lambda2)
                            par_obs.append(1 / (sum(sp_dbh2) / len(sp_dbh2) - 1))
                    mr_out_row.append(macroecotools.obs_pred_mse(np.array(mr_avg_obs), 
                                                                     np.array(mr_avg_pred)))
                    lambda_out_row.append(macroecotools.obs_pred_mse(np.array(par_obs), 
                                                                         np.array(par_pred)))

                f1.writerow(mr_out_row)
                f2.writerow(lambda_out_row)
                
    f1_write.close()
    f2_write.close()
    
def get_quantiles(input_filename, data_dir = './data/'):
    """Manipulate file obtained from species_rand_test to obtain quantiles for plotting"""
    emp_list, max_list, min_list = [], [], []
    with open(data_dir + input_filename, 'rb') as datafile:
        datareader = csv.reader(datafile, delimiter = ',')
        for row in datareader:
            emp_list.append(float(row[1]))
            float_rand = [float(i) for i in row[2:]]
            quant = mquantiles(float_rand, prob = [0.025, 0.975])
            max_list.append(quant[1])
            min_list.append(quant[0])
    # Sort emp_list and use the order for the other two lists
    order = sorted(range(len(emp_list)), key = lambda k: emp_list[k])
    emp_list = [emp_list[i] for i in order]
    max_list = [max_list[i] for i in order]
    min_list = [min_list[i] for i in order]
    results = np.array([emp_list, max_list, min_list])
    return results

def plot_rand_exp(datasets, data_dir = './data/', cutoff = 9, n_cutoff = 4):
    """Plot predicted-observed MR relationship and abd-scaled parameter relationship,
    
    with expected results from randomization.
    
    """
    for dataset in datasets:
        raw_data = import_raw_data(dataset + '.csv')
        usites = np.sort(list(set(raw_data['site'])))
        for site in usites:
            subdat = raw_data[raw_data['site'] == site]
            dbh_raw = subdat[subdat.dtype.names[2]]
            dbh_scale = np.array(dbh_raw / min(dbh_raw))
            dbh2_scale = dbh_scale ** 2
            E0 = sum(dbh2_scale)
            N0 = len(dbh2_scale)
            S_list = set(subdat[subdat.dtype.names[1]])
            S0 = len(S_list)
            if S0 > cutoff: 
                mr_out_row = [dataset + '_' + site]
                lambda_out_row = [dataset + '_' + site]
                psi = psi_epsilon(S0, N0, E0)
                theta_epsilon_obj = theta_epsilon(S0, N0, E0)
                abd = []
                mr_avg_obs = []
                scaled_par_list = []
                mr_avg_pred = []
                for sp in S_list:
                    sp_dbh2 = dbh2_scale[subdat[subdat.dtype.names[1]] == sp]
                    if len(sp_dbh2) > n_cutoff: 
                        abd.append(len(sp_dbh2))
                        mr_avg_obs.append(sum(sp_dbh2) / len(sp_dbh2))
                        scaled_par = 1 / (sum(sp_dbh2) / len(sp_dbh2) - 1) / psi.lambda2
                        scaled_par_list.append(scaled_par)
                        mr_avg_pred.append(theta_epsilon_obj.E(len(sp_dbh2)))
                fig = plt.figure()
                plot_obj = fig.add_subplot(121)
                plot_obj.scatter(abd, mr_avg_pred, color = '#9400D3')
                plot_obj.scatter(abd, mr_avg_obs, color = 'red')
                plot_obj.scatter(abd, np.array([E0 / N0] * len(abd)), color = 'black')
                plot_obj.set_xscale('log')
                plot_obj.set_yscale('log')
                plt.xlabel('Species Abundance')
                plt.ylabel('Average MR')
                plt.annotate(r'MSE = %0.2f' %macroecotools.obs_pred_mse(np.array(mr_avg_obs), np.array(mr_avg_pred)),
                             xy = (0.6, 0.1), xycoords = 'axes fraction', color = 'red')
                plt.annotate(r'MSE = %0.2f' %macroecotools.obs_pred_mse(np.array([E0 / N0] * len(abd)), np.array(mr_avg_pred)),
                             xy = (0.6, 0.05), xycoords = 'axes fraction', color = 'black')

                plot_obj = fig.add_subplot(122)
                plot_obj.scatter(abd, abd, color = '#9400D3')
                plot_obj.scatter(abd, scaled_par_list, color = 'red')
                plot_obj.scatter(abd, np.array([1 / (E0 / N0 - 1) / psi.lambda2] * len(abd)), color = 'black')
                plt.xlabel('Species Abundance')
                plt.ylabel('Scaled Intraspecific Parameter')
                plot_obj.set_xscale('log')
                plot_obj.set_yscale('log')
                plt.subplots_adjust(wspace = 0.55)
                plt.annotate(r'MSE = %0.2f' %macroecotools.obs_pred_mse(np.array(scaled_par_list), np.array(abd)),
                             xy = (0.6, 0.1), xycoords = 'axes fraction', color = 'red')
                plt.annotate(r'MSE = %0.2f' %macroecotools.obs_pred_mse(np.array([1 / (E0 / N0 - 1) / psi.lambda2] * len(abd)), np.array(abd)),
                             xy = (0.6, 0.05), xycoords = 'axes fraction', color = 'black')

                plt.savefig(data_dir + dataset + '_' + site + '_rand_exp.png', dpi = 400)
                plt.close()

def plot_rand_test(data_dir = './data/'):
    """Plot the results obtained from species_rand_test"""
    rand_mr = get_quantiles('mr_rand_sites.csv', data_dir = data_dir)
    rand_lambda = get_quantiles('lambda_rand_sites.csv', data_dir = data_dir)
    fig = plt.figure(figsize = (3.42, 7)) # 8.7cm single column width required by PNAS
    ax_mr = plt.subplot(211)
    plt.semilogy(np.arange(len(rand_mr[0])), rand_mr[0], 'ko-', markersize = 2)
    ax_mr.fill_between(np.arange(len(rand_mr[0])), rand_mr[1], rand_mr[2],
                       color = '#CFCFCF', edgecolor = '#CFCFCF')
    ax_mr.axes.get_xaxis().set_ticks([])
    y_ticks_mr = [r'$1.0$', r'$10.0$', r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$', 
               r'$10^6$', r'$10^7$', r'$10^8$', r'$10^9$', r'$10^{10}$', r'$10^{11}$']
    ax_mr.set_yticklabels(y_ticks_mr, fontsize = 6)
    ax_mr.set_xlabel('Plots', fontsize = 8)
    ax_mr.set_ylabel('MSE of size-abundance relationship', fontsize = 8)
    
    ax_lambda = plt.subplot(212)
    plt.semilogy(np.arange(len(rand_lambda[0])), rand_lambda[0], 'ko-', markersize = 2)
    ax_lambda.fill_between(np.arange(len(rand_lambda[0])), rand_lambda[1], rand_lambda[2],
                       color = '#CFCFCF', edgecolor = '#CFCFCF')
    ax_lambda.axes.get_xaxis().set_ticks([])
    y_ticks_lambda = [r'$10^{-8}$', r'$10^{-7}$', r'$10^{-6}$', r'$10^{-5}$', r'$10^{-4}$', 
                      r'$10^{-3}$', r'$10^{-2}$', r'$0.1$', r'$1.0$', r'$10.0$']
    ax_lambda.set_yticklabels(y_ticks_lambda, fontsize = 6)
    ax_lambda.set_xlabel('Plots', fontsize = 8)
    ax_lambda.set_ylabel('MSE of iISD parameter', fontsize = 8)

    plt.subplots_adjust(hspace = 0.25, left = 0.25, right = 0.9, top = 0.95, bottom = 0.05)
    plt.savefig('rand_test.pdf', dpi = 400)

def get_obs_pred_from_file(datasets, data_dir, filename):
    """Read obs and pred value from a file"""
    obs = []
    pred = []
    for i, dataset in enumerate(datasets):
        obs_pred_data = import_obs_pred_data(data_dir + dataset + filename) 
        obs.extend(list(obs_pred_data['obs']))
        pred.extend(list(obs_pred_data['pred']))
    return obs, pred
        
def plot_obs_pred(obs, pred, radius, loglog, ax = None):
    """Generic function to generate an observed vs predicted figure with 1:1 line"""
    if not ax:
        fig = plt.figure(figsize = (3.5, 3.5))
        ax = plt.subplot(111)

    axis_min = max(0.9 * min(obs+pred), 10 ** -10)
    axis_max = 3 * max(obs+pred)
    macroecotools.plot_color_by_pt_dens(np.array(pred), np.array(obs), radius, loglog=loglog, plot_obj = ax)      
    plt.plot([axis_min, axis_max],[axis_min, axis_max], 'k-')
    plt.xlim(axis_min, axis_max)
    plt.ylim(axis_min, axis_max)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 6)
    plt.annotate(r'$R^2$ = %0.2f' %macroecotools.obs_pred_rsquare(np.array(obs), np.array(pred)),
                 xy = (0.72, 0.05), xycoords = 'axes fraction', fontsize = 7)
    return ax

def plot_obs_pred_sad(datasets, data_dir = "./data/", radius = 2):
    """Plot the observed vs predicted abundance for each species for multiple datasets."""
    rad_obs, rad_pred = get_obs_pred_from_file(datasets, data_dir, '_obs_pred_rad.csv')
    fig = plot_obs_pred(rad_obs, rad_pred, radius, 1)
    fig.set_xlabel('Predicted abundance', labelpad = 4, size = 8)
    fig.set_ylabel('Observed abundance', labelpad = 4, size = 8)
    plt.savefig('obs_pred_sad.png', dpi = 400)

def plot_obs_pred_dbh2(datasets, data_dir = "./data/", radius = 2):
    """Plot the observed vs predicted dbh2 for each individual for multiple datasets."""
    dbh2_obs, dbh2_pred = get_obs_pred_from_file(datasets, data_dir, '_obs_pred_isd_dbh2.csv')
    fig = plot_obs_pred(dbh2_obs, dbh2_pred, radius, 1)
    fig.set_xlabel(r'Predicted $DBH^2$', labelpad = 4, size = 8)
    fig.set_ylabel(r'Observed $DBH^2$', labelpad = 4, size = 8)
    plt.savefig('obs_pred_dbh2.png', dpi = 400)

def plot_obs_pred_cdf(datasets, data_dir = "./data/", radius = 0.05):
    """Plot the observed vs predicted cdf for multiple datasets."""
    cdf_obs, cdf_pred = get_obs_pred_from_file(datasets, data_dir, '_obs_pred_isd_cdf.csv')
    fig = plot_obs_pred(cdf_obs, cdf_pred, radius, 0)
    fig.set_xlabel('Predicted F(x)', labelpad = 4, size = 8)
    fig.set_ylabel('Observed F(x)', labelpad = 4, size = 8)
    plt.savefig('obs_pred_cdf.png', dpi = 400)

def plot_obs_pred_freq(datasets, data_dir = "./data/", radius = 0.05):
    """Plot the observed vs predicted size frequency for multiple datasets."""
    freq_obs, freq_pred = get_obs_pred_from_file(datasets, data_dir, '_obs_pred_freq.csv')
    fig = plot_obs_pred(freq_obs, freq_pred, radius, 1)
    fig.set_xlabel('Predicted frequency', labelpad = 4, size = 8)
    fig.set_ylabel('Observed frequency', labelpad = 4, size = 8)
    plt.savefig('obs_pred_freq.png', dpi = 400)

def plot_obs_pred_avg_mr(datasets, data_dir = "./data/", radius = 2):
    """Plot the observed vs predicted species-level average metabolic rate 
    
    for all species across multiple datasets.
    
    """
    mr_obs, mr_pred = get_obs_pred_from_file(datasets, data_dir, '_obs_pred_avg_mr.csv')
    fig = plot_obs_pred(mr_obs, mr_pred, radius, 1)
    fig.set_xlabel('Predicted species-average metabolic Rate', labelpad = 4, size = 8)
    fig.set_ylabel('Observed species-average metabolic Rate', labelpad = 4, size = 8)
    plt.savefig('obs_pred_average_mr.png', dpi = 400)

def plot_obs_pred_iisd_par(datasets, data_dir = "./data/", radius = 2):
    """Plot the scaled intra-specific energy distribution parameter against abundance."""
    par_obs, par_pred = get_obs_pred_from_file(datasets, data_dir, '_par.csv')
    fig = plot_obs_pred(par_obs, par_pred, radius, 1)
    fig.set_xlabel('Predicted parameter', labelpad = 4, size = 8)
    fig.set_ylabel('Observed parameter', labelpad = 4, size = 8)
    plt.savefig('intra_par.png', dpi = 400)

def plot_four_patterns(datasets, data_dir = "./data/", radius_sad = 2, radius_freq = 0.05, 
                       radius_mr = 2, radius_par = 2):
    """Plot predicted versus observed data for 4 patterns (SAD, ISD, abundance-MR relationship, 
    
    scaled parameter for intraspecific MR distribution) as subplots in a single figure.
    
    """
    fig = plt.figure(figsize = (7, 7))
    
    ax = plt.subplot(221)
    rad_obs, rad_pred = get_obs_pred_from_file(datasets, data_dir, '_obs_pred_rad.csv')
    fig1 = plot_obs_pred(rad_obs, rad_pred, radius_sad, 1, ax = ax)
    fig1.set_xlabel('Predicted abundance', labelpad = 4, size = 8)
    fig1.set_ylabel('Observed abundance', labelpad = 4, size = 8)

    ax = plt.subplot(222)
    freq_obs, freq_pred = get_obs_pred_from_file(datasets, data_dir, '_obs_pred_freq.csv')
    fig2 = plot_obs_pred(freq_obs, freq_pred, radius_freq, 1, ax = ax)
    fig2.set_xlabel('Predicted frequency', labelpad = 4, size = 8)
    fig2.set_ylabel('Observed frequency', labelpad = 4, size = 8)

    ax = plt.subplot(223)
    mr_obs, mr_pred = get_obs_pred_from_file(datasets, data_dir, '_obs_pred_avg_mr.csv')
    fig3 = plot_obs_pred(mr_obs, mr_pred, radius_mr, 1, ax = ax)
    fig3.set_xlabel('Predicted species-average metabolic rate', labelpad = 4, size = 8)
    fig3.set_ylabel('Observed species-average metabolic rate', labelpad = 4, size = 8)

    ax = plt.subplot(224)
    par_obs, par_pred = get_obs_pred_from_file(datasets, data_dir, '_par.csv')
    fig4 = plot_obs_pred(par_obs, par_pred, radius_par, 1, ax = ax)
    fig4.set_xlabel('Predicted parameter', labelpad = 4, size = 8)
    fig4.set_ylabel('Observed parameter', labelpad = 4, size = 8)

    plt.subplots_adjust(wspace = 0.2, hspace = 0.2)
    plt.savefig('four_patterns.pdf', dpi = 400)    

def plot_four_patterns_single(datasets, outfile, data_dir = "./data/", radius_sad = 2, 
                              radius_freq = 0.05, radius_mr = 2, radius_par = 2):
    """Create the four-pattern figure for each plot separately and save all figures into a single pdf."""
    pp = PdfPages(outfile)
    for dataset in datasets:
        dat_rad = import_obs_pred_data(data_dir + dataset + '_obs_pred_rad.csv')
        dat_freq = import_obs_pred_data(data_dir + dataset + '_obs_pred_freq.csv')
        dat_mr = import_obs_pred_data(data_dir + dataset + '_obs_pred_avg_mr.csv')
        dat_par = import_obs_pred_data(data_dir + dataset + '_par.csv')
        sites = np.sort(list(set(dat_rad['site'])))
        for site in sites:
            dat_rad_site = dat_rad[dat_rad['site'] == site]
            dat_freq_site = dat_freq[dat_freq['site'] == site]
            dat_mr_site = dat_mr[dat_mr['site'] == site]
            dat_par_site = dat_par[dat_par['site'] == site]
            
            fig = plt.figure(figsize = (7, 7))
            ax = plt.subplot(221)
            fig1 = plot_obs_pred(list(dat_rad_site['obs']), list(dat_rad_site['pred']), radius_sad, 1, ax = ax)
            fig1.set_xlabel('Predicted abundance', labelpad = 4, size = 8)
            fig1.set_ylabel('Observed abundance', labelpad = 4, size = 8)
        
            ax = plt.subplot(222)
            fig2 = plot_obs_pred(list(dat_freq_site['obs']), list(dat_freq_site['pred']), radius_freq, 1, ax = ax)
            fig2.set_xlabel('Predicted frequency', labelpad = 4, size = 8)
            fig2.set_ylabel('Observed frequency', labelpad = 4, size = 8)
        
            ax = plt.subplot(223)
            fig3 = plot_obs_pred(list(dat_mr_site['obs']), list(dat_mr_site['pred']), radius_mr, 1, ax = ax)
            fig3.set_xlabel('Predicted species-average metabolic rate', labelpad = 4, size = 8)
            fig3.set_ylabel('Observed species-average metabolic rate', labelpad = 4, size = 8)
        
            ax = plt.subplot(224)
            fig4 = plot_obs_pred(list(dat_par_site['obs']), list(dat_par_site['pred']), radius_par, 1, ax = ax)
            fig4.set_xlabel('Predicted parameter', labelpad = 4, size = 8)
            fig4.set_ylabel('Observed parameter', labelpad = 4, size = 8)
        
            plt.subplots_adjust(wspace = 0.2, hspace = 0.2)
            plt.suptitle(dataset + ',' + site)
            plt.savefig(pp, format = 'pdf', dpi = 400)
    pp.close()

def plot_four_patterns_single_ver2(datasets, outfile, data_dir = "./data/", radius_par = 2, title = True,
                                   bin_size = 1.7, cutoff = 9, n_cutoff = 4):
    """Version two of four-pattern figure at plot level."""
    pp = PdfPages(outfile)
    for dataset in datasets:
        dat_rad = import_obs_pred_data(data_dir + dataset + '_obs_pred_rad.csv')
        dat_raw = import_raw_data(dataset + '.csv')
        dat_freq = import_obs_pred_data(data_dir + dataset + '_obs_pred_freq.csv')
        dat_mr = import_obs_pred_data(data_dir + dataset + '_obs_pred_avg_mr.csv')
        dat_par = import_obs_pred_data(data_dir + dataset + '_par.csv')
        sites = np.sort(list(set(dat_rad['site'])))
        for site in sites:
            dat_raw_site = dat_raw[dat_raw['site'] == site]
            sp_list = set(dat_raw_site['sp'])
            S0 = len(sp_list)
            if S0 > cutoff:
                dat_rad_site = dat_rad[dat_rad['site'] == site]
                dat_freq_site = dat_freq[dat_freq['site'] == site]
                dat_mr_site = dat_mr[dat_mr['site'] == site]
                dat_par_site = dat_par[dat_par['site'] == site]

                obs = dat_rad_site['obs']
                pred = dat_rad_site['pred']
                rank_obs, relab_obs = macroecotools.get_rad_data(obs)
                rank_pred, relab_pred = macroecotools.get_rad_data(pred)
                
                dbh_raw_site = dat_raw_site['dbh']
                dbh_scale_site = np.array(dbh_raw_site / min(dbh_raw_site))
                dbh2_scale_site = dbh_scale_site ** 2
                E0 = sum(dbh2_scale_site)
                N0 = len(dbh2_scale_site)
                num_bin = int(ceil(log(max(dbh2_scale_site)) / log(bin_size)))
                emp_pdf = []
                for i in range(num_bin):
                    count = len(dbh2_scale_site[(dbh2_scale_site < bin_size ** (i + 1)) & (dbh2_scale_site >= bin_size ** i)])
                    emp_pdf.append(count / N0 / (bin_size ** i * (bin_size - 1)))
                psi = psi_epsilon(S0, N0, E0)
                psi_pdf = [float(psi.pdf(x)) for x in np.arange(1, ceil(max(dbh2_scale_site)) + 1)]
                
                theta_epsilon_obj = theta_epsilon(S0, N0, E0)
                dbh2_obs = []
                n_obs = []
                for sp in sp_list:
                    dat_sp = dbh2_scale_site[dat_raw_site['sp'] == sp]
                    n_obs.append(len(dat_sp))
                    dbh2_obs.append(sum(dat_sp) / len(dat_sp))
                n_list = [int(x) for x in np.exp(np.arange(log(min(n_obs)), log(max(n_obs)), 0.1))]
                dbh2_pred = [theta_epsilon_obj.E(n) for n in n_list]
     
                fig = plt.figure(figsize = (7, 7))
                ax = plt.subplot(221)
                ax.semilogy(rank_obs, relab_obs, 'o', markerfacecolor='none', markersize=6, 
                                  markeredgecolor='#999999', markeredgewidth=1)
                ax.semilogy(rank_pred, relab_pred, '-', color='#9400D3', linewidth=2)
                ax.tick_params(axis = 'both', which = 'major', labelsize = 6)
                plt.xlabel('Rank', fontsize = 8)
                plt.ylabel('Relative abundance', fontsize = 8)
                plt.annotate(r'$R^2$ = %0.2f' %macroecotools.obs_pred_rsquare(np.array(obs), np.array(pred)),
                             xy = (0.72, 0.9), xycoords = 'axes fraction', fontsize = 7)
                
                ax = plt.subplot(222)
                ax.loglog(np.arange(1, ceil(max(dbh2_scale_site)) + 1), psi_pdf, '#9400D3', linewidth = 2)
                ax.bar(bin_size ** np.array(range(num_bin)), emp_pdf, color = '#d6d6d6', 
                       width = 0.4 * bin_size ** np.array(range(num_bin)))
                plt.ylim((max(min(emp_pdf), 10 ** -10), 1))
                ax.tick_params(axis = 'both', which = 'major', labelsize = 6)
                plt.xlabel(r'$DBH^2$', fontsize = 8)
                plt.ylabel('Frequency', fontsize = 8)
                plt.annotate(r'$R^2$ = %0.2f' %macroecotools.obs_pred_rsquare(dat_freq_site['obs'], dat_freq_site['pred']),
                             xy = (0.72, 0.9), xycoords = 'axes fraction', fontsize = 7)
 
                ax = plt.subplot(223)
                ax.loglog(np.array(dbh2_pred), np.array(n_list), color = '#9400D3', linewidth = 2)
                ax.scatter(dbh2_obs, n_obs, color = '#999999', marker = 'o')
                ax.tick_params(axis = 'both', which = 'major', labelsize = 6)
                plt.xlabel('Species-average metabolic rate', fontsize = 8)
                plt.ylabel('Species abundance', fontsize = 8)
                plt.annotate(r'$R^2$ = %0.2f' %macroecotools.obs_pred_rsquare(dat_mr_site['obs'], dat_mr_site['pred']),
                             xy = (0.72, 0.05), xycoords = 'axes fraction', fontsize = 7)
                
                ax = plt.subplot(224)
                fig4 = plot_obs_pred(list(dat_par_site['obs']), list(dat_par_site['pred']), radius_par, 1, ax = ax)
                fig4.set_xlabel('Predicted parameter', labelpad = 4, size = 8)
                fig4.set_ylabel('Observed parameter', labelpad = 4, size = 8)
                
                plt.subplots_adjust(wspace = 0.2, hspace = 0.2)
                if title:
                    plt.suptitle(dataset + ',' + site)
                plt.savefig(pp, format = 'pdf', dpi = 400)
    pp.close()


def comp_isd(datasets, list_of_datasets, data_dir = "./data/"):
    """Compare the three visual representation of ISD: histogram with pdf, 
    
    predicted and empirical cdf, 1:1 plot of cdf. 
    
    """
    for j, dataset in enumerate(datasets):
        usites = np.sort(list(set(dataset["site"])))
        for site in usites:
            fig = plt.figure(figsize = (7, 3))
            dat_site = dataset[dataset['site'] == site]
            dbh_raw = dat_site[dat_site.dtype.names[2]]
            dbh_scale = np.array(sorted(dbh_raw / min(dbh_raw)))
            dbh2_scale = dbh_scale ** 2
            E0 = sum(dbh2_scale)
            N0 = len(dbh2_scale)
            S0 =  len(set(dat_site[dat_site.dtype.names[1]]))
            psi = psi_epsilon(S0, N0, E0)
            # Plot histogram with pdf
            num_bin = int(ceil(log(max(dbh_scale)) / log(2)))
            emp_pdf = []
            for i in range(num_bin):
                count = len(dbh_scale[(dbh_scale < 2 ** (i + 1)) & (dbh_scale >= 2 ** i)])
                emp_pdf.append(count / N0 / 2 ** i)
            psi_pdf = []
            x_array = np.arange(1, ceil(max(dbh_scale)) + 1)
            for x in x_array:
                psi_pdf.append(float(ysquareroot_pdf(x, psi)))
            plot_obj = plt.subplot(1, 3, 1)
            plot_obj.loglog(x_array, psi_pdf, '#9400D3', linewidth = 2, label = 'METE')
            plot_obj.bar(2 ** np.array(range(num_bin)), emp_pdf, color = '#d6d6d6', 
                    width = 0.4 * 2 ** np.array(range(num_bin)))
            plt.xlabel('DBH')
            plt.ylabel('Probability Density')
            # Plot predicted and empirical cdf
            plot_obj = plt.subplot(1, 3, 2)
            pred_cdf = get_mete_pred_cdf(dbh2_scale, S0, N0, E0)
            emp_cdf = get_obs_cdf(dbh2_scale)
            plot_obj.loglog(dbh2_scale, pred_cdf, color = '#9400D3', linestyle = '-', linewidth = 2)
            plot_obj.loglog(dbh2_scale, emp_cdf, 'r-', linewidth = 2)
            plt.xlabel(r'$DBH^2$')
            plt.ylabel('F(x)')
            # Plot predicted cdf against empirical cdf
            plot_obj = plt.subplot(1, 3, 3)
            obs = emp_cdf
            pred = pred_cdf
            axis_min = 0.9 * min(list(obs)+list(pred))
            axis_max = 1.1 * max(list(obs)+list(pred))
            macroecotools.plot_color_by_pt_dens(np.array(pred), np.array(obs), 0.05, loglog=0, 
                                                plot_obj=plt.subplot(1,3,3))      
            plot_obj.plot([axis_min, axis_max],[axis_min, axis_max], 'k-')
            plt.xlim(axis_min, axis_max)
            plt.ylim(axis_min, axis_max)
            plt.xlabel('Predicted F(x)')
            plt.ylabel('Observed F(x)')
            plt.text(axis_max * 0.8, axis_max / 3, 
                     r'$r^2$ = %0.2f' %macroecotools.obs_pred_rsquare(np.array(obs), np.array(pred)))
            plt.subplots_adjust(wspace = 0.55, bottom = 0.3)
            plt.savefig(data_dir + list_of_datasets[j] + '_' + site + '_comp_cdf.png', dpi = 400)
    
def plot_fig1(output_dir = ""):
    """Illustration of the 4 patterns using BCI data."""
    fig = plt.figure(figsize = (7, 7))
    # Subplot A: Observed and predicted RAD
    # Code adopted and modified from example_sad_plot in mete_sads
    obs_pred_data = import_obs_pred_data('./data/'+ 'BCI' + '_obs_pred_rad.csv')    
    obs = obs_pred_data["obs"]   
    pred = obs_pred_data["pred"]
    rank_obs, relab_obs = macroecotools.get_rad_data(obs)
    rank_pred, relab_pred = macroecotools.get_rad_data(pred)
    plot_obj = plt.subplot(2, 2, 1)
    plot_obj.semilogy(rank_obs, relab_obs, 'o', markerfacecolor='none', markersize=6, 
             markeredgecolor='#999999', markeredgewidth=1)
    plot_obj.semilogy(rank_pred, relab_pred, '-', color='#9400D3', linewidth=2)
    plot_obj.tick_params(axis = 'both', which = 'major', labelsize = 6)
    plt.xlabel('Rank', fontsize = 8)
    plt.ylabel('Relative abundance', fontsize = 8)
    # Subplot B: ISD shown as histogram with predicted pdf
    # Code adopted and modified from plot_ind_hist from fit_other_dist
    dat = import_raw_data('BCI.csv')
    dbh_raw = dat[dat.dtype.names[2]]
    dbh_scale = np.array(dbh_raw / min(dbh_raw))
    dbh2_scale = dbh_scale ** 2
    E0 = sum(dbh2_scale)
    N0 = len(dbh2_scale)
    S0 = len(set(dat[dat.dtype.names[1]]))

    num_bin = int(ceil(log(max(dbh2_scale)) / log(1.7)))
    emp_pdf = []
    for i in range(num_bin):
        count = len(dbh2_scale[(dbh2_scale < 1.7 ** (i + 1)) & (dbh2_scale >= 1.7 ** i)])
        emp_pdf.append(count / N0 / (1.7 ** i * 0.7))
    psi = psi_epsilon(S0, N0, E0)
    psi_pdf = []
    x_array = np.arange(1, ceil(max(dbh2_scale)) + 1)
    for x in x_array:
        psi_pdf.append(float(psi.pdf(x)))
    plot_obj = plt.subplot(2, 2, 2)
    plot_obj.loglog(x_array, psi_pdf, '#9400D3', linewidth = 2, label = 'METE')
    plot_obj.bar(1.7 ** np.array(range(num_bin)), emp_pdf, color = '#d6d6d6', 
            width = 0.4 * 1.7 ** np.array(range(num_bin)))
    plt.ylim((max(min(emp_pdf), 10 ** -10), 1))
    plot_obj.tick_params(axis = 'both', which = 'major', labelsize = 6)
    plt.xlabel(r'$DBH^2$', fontsize = 8)
    plt.ylabel('Probability density', fontsize = 8)
    # Subplot C: Size-abundance distribution, shown as DBH^2 against abundance
    # Code adopted and modified from plot_species_avg_single
    theta_epsilon_obj = theta_epsilon(S0, N0, E0)
    spp_list = set(dat[dat.dtype.names[1]])
    
    dbh2_obs = []
    n_obs = []
    for spp in set(spp_list):
        dat_spp = sorted(dbh2_scale[dat[dat.dtype.names[1]] == spp])
        n_obs.append(len(dat_spp))
        dbh2_obs.append(sum(dat_spp) / len(dat_spp))
    n_list = [int(x) for x in np.exp(np.arange(log(min(n_obs)), log(max(n_obs)), 0.1))]
    dbh2_pred = []
    for n in n_list:
        dbh2_pred.append(theta_epsilon_obj.E(n))
    plot_obj = plt.subplot(2, 2, 3)
    plot_obj.loglog(np.array(dbh2_pred), np.array(n_list), color = '#9400D3', linewidth = 2)
    plot_obj.scatter(dbh2_obs, n_obs, color = '#999999', marker = 'o')
    plot_obj.tick_params(axis = 'both', which = 'major', labelsize = 6)
    plt.xlabel('Species-average metabolic rate', fontsize = 8)
    plt.ylabel('Species abundance', fontsize = 8)
    # Subplot D: Intra-specific distribution for the most abundant species (Hybanthus prunifolius)
    hp_dbh2 = dbh2_scale[dat[dat.dtype.names[1]] == 'Hybanthus prunifolius']
    hp_num_bin = int(ceil(log(max(hp_dbh2)) / log(1.7)))
    hp_emp_pdf = []
    for i in range(hp_num_bin):
        count = len(hp_dbh2[(hp_dbh2 < 1.7 ** (i + 1)) & (hp_dbh2 >= 1.7 ** i)])
        hp_emp_pdf.append(count / len(hp_dbh2) / (1.7 ** i * 0.7))
    def exp_dist(x, lam):
        return lam * np.exp(-lam * x)
    lam_pred = psi.lambda2 * len(hp_dbh2)
    lam_est = 1 / (sum(hp_dbh2) / len(hp_dbh2) - 1)
    x_array = np.arange(1, ceil(max(hp_dbh2)) + 1)
    plot_obj = plt.subplot(2, 2, 4)
    p_mete, = plot_obj.loglog(x_array, exp_dist(x_array, lam_pred), '#9400D3', linewidth = 2, label = 'METE')
    p_mle, = plot_obj.loglog(x_array, exp_dist(x_array, lam_est), '#FF4040', linewidth = 2, label = 'Truncated exponential')
    plot_obj.bar(1.7 ** np.array(range(hp_num_bin)), hp_emp_pdf, color = '#d6d6d6', 
        width = 0.4 * 1.7 ** np.array(range(hp_num_bin)))
    plt.ylim((max(min(hp_emp_pdf), 10 ** -10), 1))
    plot_obj.tick_params(axis = 'both', which = 'major', labelsize = 6)
    plt.xlabel(r'$DBH^2$', fontsize = 8)
    plt.ylabel('Probability Density', fontsize = 8)
    plt.legend([p_mete, p_mle], ['METE parameter: '+str(round(lam_pred, 4)), 'MLE parameter: '+str(round(lam_est, 4))],
               loc = 1, prop = {'size': 6})
    plt.subplots_adjust(wspace = 0.29, hspace = 0.29)
    plt.savefig(output_dir + 'fig1.pdf', dpi = 400)

def get_weights_all(datasets, list_of_dataset_names, par_table, data_dir = './data/'):
    """Create a csv file with AICc weights of the four distributions"""
    f1_write = open(data_dir + 'weight_table.csv', 'wb')
    #f1 = csv.writer(f1_write)
    out = []
    for i, dataset in enumerate(datasets):
        usites = np.sort(list(set(dataset['site'])))
        for site in usites:
            dat_site = dataset[dataset['site'] == site]
            par_site = par_table[(par_table['dataset'] == list_of_dataset_names[i]) & 
                                      (par_table['site'] == site)]
            weight_site = weights(dat_site, par_site[0][2], par_site[0][3], 
                                  par_site[0][4], par_site[0][5])
            out.append([list_of_dataset_names[i], site, weight_site[0], weight_site[1], 
                        weight_site[2], weight_site[3]])
    #f1.writerows(np.array(out))
    np.savetxt(f1_write, np.array(out), 
               fmt = ('%s', '%s', '%f', '%f', '%f', '%f'), delimiter = ",")
    f1_write.close()

def plot_hist(datasets, list_of_dataset_names, par_table, data_dir = './data/'):
    """Plot histogram with density curves of the four distributions for individual sites"""
    fig = plt.figure(figsize = (7, 7))
    num_datasets = len(datasets)
    for i, dataset in enumerate(datasets):
        dat_site = dataset
        par_site = par_table[par_table['dataset'] == list_of_dataset_names[i]]
        ax = fig.add_subplot(2,2,i+1)
        plot_ind_hist(dat_site, par_site[0][2], par_site[0][3], par_site[0][4], 
                      par_site[0][5], list_of_dataset_names[i], plot_obj=plt.subplot(2,2,i+1), 
                      legend = (i == 3))      
        plt.subplots_adjust(wspace=0.29, hspace=0.29)  
        plt.title(list_of_dataset_names[i])
        fig.text(0.5, 0.04, 'DBH', ha = 'center', va = 'center')
        #fig.text(0.5, 0.04, r'$DBH^2$', ha = 'center', va = 'center')
        fig.text(0.04, 0.5, 'Probability Density', ha = 'center', va = 'center', 
                 rotation = 'vertical')
    plt.savefig('ind_dist.png', dpi = 400)

def plot_spp_exp_single(dat, threshold = 5, plot_obj = None):
    """Plot the MLE exponential parameter for each species against abundance n, 
    
    for all species with abundance higher than the threshold.
    
    Copied and modified from the same function in EM_comp. 
    """
    if plot_obj == None:
        plot_obj = plt.figure()
        
    dbh_raw = dat[dat.dtype.names[2]]
    dbh_scale = np.array(dbh_raw / min(dbh_raw))
    dbh2_scale = dbh_scale ** 2
    E0 = sum(dbh2_scale)
    N0 = len(dbh2_scale)
    S0 = len(set(dat[dat.dtype.names[1]]))
    psi = psi_epsilon(S0, N0, E0)
    lam2 = psi.lambda2
    
    spp_list = set(dat[dat.dtype.names[1]])
    n_list = []
    exp_list = []
    for spp in spp_list:
        dbh2_spp = sorted(dbh2_scale[dat[dat.dtype.names[1]] == spp])
        n = len(dbh2_spp)
        if n >= threshold:
            n_list.append(n)
            exp_list.append(1 / (np.mean(dbh2_spp) - 1))
    plot_obj.loglog(np.array(range(min(n_list), max(n_list) + 1)), 
                      np.array(range(min(n_list), max(n_list) + 1)) * lam2, 
                      color = '#9400D3', linewidth = 2)
    plot_obj.scatter(n_list, exp_list, color = '#999999', marker = 'o')
    return plot_obj

def plot_spp_exp(datasets, list_of_dataset_names, data_dir = './data/'):
    """Call plot_spp_exp_single and generate a figure with multiple subplots."""
    fig = plt.figure(figsize = (7, 7))
    num_datasets = len(datasets)
    for i, dataset in enumerate(datasets):
        dat_site = dataset
        ax = fig.add_subplot(2, 2, i+1)
        plot_spp_exp_single(dat_site, plot_obj = plt.subplot(2, 2, i + 1))
        plt.subplots_adjust(wspace = 0.29, hspace = 0.29)
        plt.title(list_of_dataset_names[i])
        fig.text(0.5, 0.04, 'Species Abundance', ha = 'center', va = 'center')
        fig.text(0.04, 0.5, r'$\lambda$ for Within Species Distribution', ha = 'center',
                 va = 'center', rotation = 'vertical')
    plt.savefig(data_dir + 'within_spp.png', dpi = 400)

def plot_species_avg_single(dat, plot_obj = None):
    """Plot the expected versus observed value of species-level energy or biomass
    
    when the corresponding constraing is used. 
    Copied and modified from the function in EM_comp. 
    
    """
    if plot_obj == None:
        plot_obj = plt.figure()
    dbh_raw = dat[dat.dtype.names[2]]
    dbh_scale = np.array(dbh_raw / min(dbh_raw))
    dbh2_scale = dbh_scale ** 2
    E0 = sum(dbh2_scale)
    N0 = len(dbh2_scale)
    S0 = len(set(dat[dat.dtype.names[1]]))
    theta_epsilon_obj = theta_epsilon(S0, N0, E0)
    spp_list = set(dat[dat.dtype.names[1]])
    
    dbh2_obs = []
    n_obs = []
    for spp in set(spp_list):
        dat_spp = sorted(dbh2_scale[dat[dat.dtype.names[1]] == spp])
        n_obs.append(len(dat_spp))
        dbh2_obs.append(sum(dat_spp) / len(dat_spp))
    n_list = sorted(list(set([int(x) for x in np.exp(np.arange(log(min(n_obs)), log(max(n_obs)), 0.1))])))
    dbh2_pred = []
    for n in n_list:
        dbh2_pred.append(theta_epsilon_obj.E(n))
    plot_obj.loglog(np.array(dbh2_pred), np.array(n_list), color = '#9400D3', linewidth = 2)
    plot_obj.scatter(dbh2_obs, n_obs, color = '#999999', marker = 'o')
    return plot_obj

def plot_species_avg(datasets, list_of_dataset_names, data_dir = './data/'):
    """Call plot_species_avg_single and generate a figure with multiple subplots."""
    fig = plt.figure(figsize = (7, 7))
    num_datasets = len(datasets)
    for i, dataset in enumerate(datasets):
        ax = fig.add_subplot(2, 2, i+1)
        plot_species_avg_single(dataset, plot_obj = plt.subplot(2, 2, i+1))
        plt.subplots_adjust(wspace = 0.29, hspace = 0.29)
        plt.title(list_of_dataset_names[i])
        fig.text(0.5, 0.04, r'Species Level Average $DBH^2$', ha = 'center', va = 'center')
        fig.text(0.04, 0.5, 'Species Abundance', ha = 'center',
                 va = 'center', rotation = 'vertical')
    plt.savefig(data_dir + 'across_spp.png', dpi = 400)

def plot_spp_frequency(dat, spp_name1, spp_name2, data_dir = './data/'):
    """Plot the predicted vs. observed frequency distribution for two species.
    
    Empirical data are plotted as ranked dots while predicted distributions
    are plotted as reverted cdf. 
    For Ethan's talk. 
    
    """
    dbh_raw = dat[dat.dtype.names[2]]
    dbh_scale = np.array(dbh_raw / min(dbh_raw))
    dbh2_scale = dbh_scale ** 2
    
    E0 = sum(dbh2_scale)
    N0 = len(dbh2_scale)
    S0 = len(set(dat[dat.dtype.names[1]]))
    theta_epsilon_obj = theta_epsilon(S0, N0, E0)

    dbh2_spp1 = dbh2_scale[dat['sp'] == spp_name1]
    dbh2_spp2 = dbh2_scale[dat['sp'] == spp_name2]
    n1 = len(dbh2_spp1)
    n2 = len(dbh2_spp2)
    
    dbh2_list1 = np.array(np.arange(min(dbh2_spp1), max(dbh2_spp1) + 2))
    dbh2_list2 = np.array(np.arange(min(dbh2_spp2), max(dbh2_spp2) + 2))
    cdf_spp1 = []
    cdf_spp2 = []
    for dbh2 in dbh2_list1:
        cdf_spp1.append(theta_epsilon_obj.cdf(dbh2, n1))
    for dbh2 in dbh2_list2:
        cdf_spp2.append(theta_epsilon_obj.cdf(dbh2, n2))

    fig = plt.figure()
    l1, = plt.loglog((1 - np.array(cdf_spp1)) * n1 + 0.5, dbh2_list1, color = '#8B3E2F', linewidth = 2)
    l2, = plt.loglog((1 - np.array(cdf_spp2)) * n2 + 0.5, dbh2_list2, color = '#00688B', linewidth = 2)
    plt.scatter(np.arange(n1, 0, -1), sorted(dbh2_spp1), color = '#FF7F50') 
    plt.scatter(np.arange(n2, 0, -1), sorted(dbh2_spp2), color = '#00BFFF')
    plt.legend([l1, l2], [spp_name1, spp_name2], loc = 1)
    
    plt.ylabel(r'$DBH^2$')
    plt.xlabel('Rank')
    plt.savefig(data_dir + 'intra_dist_2spp.png', dpi = 400)

def plot_spp_frequency_pdf(dat, spp_name1, spp_name2, data_dir = './data/'):
    """Plot the predicted vs. observed frequency distribution of energy or body mass for a specific species."""
    dbh_raw = dat[dat.dtype.names[2]]
    dbh_scale = np.array(dbh_raw / min(dbh_raw))
    dbh2_scale = dbh_scale ** 2
    
    E0 = sum(dbh2_scale)
    N0 = len(dbh2_scale)
    S0 = len(set(dat[dat.dtype.names[1]]))
    theta_epsilon_obj = theta_epsilon(S0, N0, E0)

    dbh2_spp1 = dbh2_scale[dat['sp'] == spp_name1]
    dbh2_spp2 = dbh2_scale[dat['sp'] == spp_name2]
    n1 = len(dbh2_spp1)
    n2 = len(dbh2_spp2)
    
    dbh2_list1 = np.array(np.arange(min(dbh2_spp1), max(dbh2_spp1) + 2))
    dbh2_list2 = np.array(np.arange(min(dbh2_spp2), max(dbh2_spp2) + 2))
    pdf_spp1 = []
    pdf_spp2 = []
    for dbh2 in dbh2_list1:
        pdf_spp1.append(theta_epsilon_obj.pdf(dbh2, n1))
    for dbh2 in dbh2_list2:
        pdf_spp2.append(theta_epsilon_obj.pdf(dbh2, n2))
    
    fig = plt.figure()
    l1, = plt.loglog(dbh2_list1, pdf_spp1, color = '#8B3E2F', linewidth = 2)
    l2, = plt.loglog(dbh2_list2, pdf_spp2, color = '#00688B', linewidth = 2)

    bins1 = np.exp(np.arange(log(min(dbh2_spp1)), log(max(dbh2_spp1) + 1),
                            (log(max(dbh2_spp1)) - log(min(dbh2_spp1))) / 10))
    bin_width1 = []
    for i in range(len(bins1) - 1):
        bin_width1.append(bins1[i + 1] - bins1[i])
        
    bins2 = np.exp(np.arange(log(min(dbh2_spp2)), log(max(dbh2_spp2) + 1),
                        (log(max(dbh2_spp2)) - log(min(dbh2_spp2))) / 10))
    bin_width2 = []
    for i in range(len(bins2) - 1):
        bin_width2.append(bins2[i + 1] - bins2[i])

    count1, bins_log, patches = plt.hist(np.log(dbh2_spp1), bins = np.log(bins1), visible = False)
    count2, bins_log, patches = plt.hist(np.log(dbh2_spp2), bins = np.log(bins2), visible = False)
    plt.scatter(bins1[:-1], count1 / bin_width1 / n1, color = '#FF7F50')
    plt.scatter(bins2[:-1], count2 / bin_width2 / n2, color = '#00BFFF')
    plt.axis([1, max(dbh2_spp2) * 1.5, 10 ** -6, 0.5])
    plt.xlabel(r'$DBH^2$')
    plt.ylabel('Frequency')
    plt.legend([l1, l2], [spp_name1, spp_name2], loc = 1)

    plt.savefig(data_dir + 'intra_dist_2spp_pdf.png', dpi = 400)
