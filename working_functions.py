"""Module with all working functions for the METE energy project"""

from __future__ import division
import numpy as np
import csv
import matplotlib
matplotlib.use('Agg')

import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from mete import *
from mete_sads import hist_mete_r2
from mete_distributions import *
import macroecotools
import macroeco_distributions as mdis
from math import exp, log
import scipy
from scipy.stats.mstats import mquantiles
from scipy.stats import ks_2samp
import multiprocessing
import math

def import_raw_data(input_filename):
    data = np.genfromtxt(input_filename, dtype = "S15, S25, f8", skiprows = 1, 
                         names = ['site', 'sp', 'dbh'], delimiter = ",")
    return data

def import_obs_pred_data(input_filename):
    data = np.genfromtxt(input_filename, dtype = "S15,f8,f8",
                                  names = ['site','obs','pred'],
                                  delimiter = ",")
    return data

def import_density_file(input_filename):
    data = np.genfromtxt(input_filename, dtype = "S25, f8", skiprows = 1, 
                         names = ['sp', 'wsg'], delimiter = ",")
    return data

def import_par_table(par_table):
    """Import the csv file containing parameter estimates for all datasets."""
    par_table = np.genfromtxt(par_table, dtype = "S15, S15, f8, f8, f8, f8", skiprows = 1, 
                              names = ['dataset', 'site', 'expon_par', 'pareto_par', 
                                       'weibull_k', 'weibull_lmd'], delimiter = ",")
    return par_table

def import_bootstrap_file(input_filename, Niter = 100):
    """import the txt file containing results from bootstrap or Monte Carlo simulations."""
    type_data = 'S25,S25' + ',f15' * (Niter + 1)
    names_orig = ['dataset', 'site', 'orig']
    names_sim = ['sample'+str(i) for i in xrange(1, Niter + 1)]
    names_orig.extend(names_sim)
    data = np.genfromtxt(input_filename, delimiter = ',', names = names_orig, dtype = type_data)
    return data
    
def get_obs_pred_rad(raw_data, dataset_name, data_dir='./out_files/', cutoff = 9):
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
            results = np.zeros((len(obsab), ), dtype = ('S15, i8, i8'))
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

def get_iisd_pred_dbh2(dbh2_scale, E0, iisd_par):
    """Compute the individual metabolic rate (size) predicted by METE 
    
    for each individual in one species from the intraspecific individual size distribution
    given scaled dbh ** 2, E0, and the predicted parameter 
    for the exponential distribution (iISD).
    
    """
    scaled_rank = [(x + 0.5) / len(dbh2_scale) for x in range(len(dbh2_scale))]
    m = exp(- iisd_par)
    pred_dbh2 = [log(m - x * (m - m ** E0)) / (- iisd_par) for x in scaled_rank]
    return np.array(pred_dbh2)

def get_obs_cdf(dat):
    """Compute the empirical cdf given a list or an array"""
    dat = np.array(dat)
    emp_cdf = []
    for point in dat:
        point_cdf = len(dat[dat < point]) / len(dat)
        emp_cdf.append(point_cdf)
    return np.array(emp_cdf)

def get_obs_pred_cdf(raw_data, dataset_name, data_dir = './out_files/', cutoff = 9):
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
            results = np.zeros((len(cdf_obs), ), dtype = ('S15, f8, f8'))
            results['f0'] = np.array([site] * len(cdf_obs))
            results['f1'] = cdf_obs
            results['f2'] = cdf_pred
            f1.writerows(results)
    f1_write.close()
    
def get_obs_pred_dbh2(raw_data, dataset_name, data_dir = './out_files/', cutoff = 9):
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
            results = np.zeros((len(dbh2_obs), ), dtype = ('S15, f8, f8'))
            results['f0'] = np.array([site] * len(dbh2_obs))
            results['f1'] = dbh2_obs
            results['f2'] = dbh2_pred
            f1.writerows(results)
    f1_write.close()
    
def get_obs_pred_frequency(raw_data, dataset_name, data_dir = './out_files/', bin_size = 1.7, cutoff = 9):
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
            results = np.zeros((len(freq_obs), ), dtype = ('S15, f8, f8'))
            results['f0'] = np.array([site] * len(freq_obs))
            results['f1'] = freq_obs
            results['f2'] = freq_pred
            f_writer.writerows(results)
    f.close()
        
def get_obs_pred_intradist(raw_data, dataset_name, data_dir = './out_files/', cutoff = 9, n_cutoff = 4):
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
            results1 = np.zeros((len(mr_avg_pred), ), dtype = ('S15, f8, f8'))
            results1['f0'] = np.array([site] * len(mr_avg_pred))
            results1['f1'] = np.array(mr_avg_obs)
            results1['f2'] = np.array(mr_avg_pred)
            f1.writerows(results1)
            
            results2 = np.zeros((len(par_pred), ), dtype = ('S15, f8, f8'))
            results2['f0'] = np.array([site] * len(par_pred))
            results2['f1'] = np.array(par_obs)
            results2['f2'] = np.array(par_pred)
            f2.writerows(results2)
    f1_write.close()
    f2_write.close()

def get_obs_pred_iisd(raw_data, dataset_name, data_dir = './out_files/', cutoff = 9):
    """Compare the predicted and empirical individual dbh^2 from  
    
    intra-specific energy distribution for each species and get results in csv files.
    Keyword arguments:
    raw_data : numpy structured array with 3 columns: 'site','sp','dbh'
    dataset_name : short code that will indicate the name of the dataset in
                    the output file names
    data_dir : directory in which to store output
    cutoff : minimum number of species required to run - 1.
    
    """
    usites = np.sort(list(set(raw_data["site"])))
    f1_write = open(data_dir + dataset_name + '_obs_pred_iisd_dbh2.csv', 'wb')
    f1 = csv.writer(f1_write)
    
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
            psi = psi_epsilon(S0, N0, E0)
            for sp in S_list:
                iisd_obs = sorted(dbh2_scale[subdat[subdat.dtype.names[1]] == sp])
                n = len(iisd_obs)
                iisd_pred = get_iisd_pred_dbh2(iisd_obs, E0, n * psi.lambda2)
                results = np.zeros((len(iisd_obs), ), dtype = ('S15, f8, f8'))
                results['f0'] = np.array([site] * len(iisd_obs))
                results['f1'] = np.array(iisd_obs)
                results['f2'] = np.array(iisd_pred)
                f1.writerows(results)
                
    f1_write.close()

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
    Output:
    Proportion of tests that are significant
    
    """
    par_mle = 1 / (sum(sp_dbh2) / len(sp_dbh2) - 1)
    count = 0
    for i in range(Nsim):
        sim_dbh2 = mdis.trunc_expon.rvs(par_mle, 1, size = len(sp_dbh2))
        ks_result = ks_2samp(sp_dbh2, sim_dbh2)
        if ks_result[1] <= p:
            count += 1
    return count / Nsim

def ks_test(datasets, data_dir = './out_files/', Nsim = 1000, p = 0.05, cutoff = 9, n_cutoff = 4):
    """Kolmogorov-Smirnov test for each species in each dataset."""
    f_write = open(data_dir + 'ks_test_' + str(Nsim) + '_' + str(p) + '.csv', 'wb')
    f = csv.writer(f_write)
    
    for dataset in datasets:
        raw_data = import_raw_data('./data/' + dataset + '.csv')
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
    
def species_rand_test(datasets, data_dir = './out_files/', cutoff = 9, n_cutoff = 4, Niter = 200):
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
        raw_data = import_raw_data('./data/' + dataset + '.csv')
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
                    mr_out_row.append(macroecotools.obs_pred_mse(np.log(mr_avg_obs), 
                                                                     np.log(mr_avg_pred)))
                    lambda_out_row.append(macroecotools.obs_pred_mse(np.log(par_obs), 
                                                                         np.log(par_pred)))
                f1.writerow(mr_out_row)
                f2.writerow(lambda_out_row)
                
    f1_write.close()
    f2_write.close()
    
def get_quantiles(input_filename, data_dir = './out_files/'):
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

def plot_rand_exp(datasets, data_dir = './out_files/', cutoff = 9, n_cutoff = 4):
    """Plot predicted-observed MR relationship and abd-scaled parameter relationship,
    
    with expected results from randomization.
    
    """
    for dataset in datasets:
        raw_data = import_raw_data('./data/'+ dataset + '.csv')
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

def plot_rand_test(data_dir = './out_files/'):
    """Plot the results obtained from species_rand_test"""
    rand_mr = get_quantiles('mr_rand_sites.csv', data_dir = data_dir)
    rand_lambda = get_quantiles('lambda_rand_sites.csv', data_dir = data_dir)
    fig = plt.figure(figsize = (7, 4)) # 8.7cm single column width required by PNAS
    ax_mr = plt.subplot(121)
    plt.plot(np.arange(len(rand_mr[0])), rand_mr[0], 'ko-', markersize = 2)
    ax_mr.fill_between(np.arange(len(rand_mr[0])), rand_mr[1], rand_mr[2],
                       color = '#CFCFCF', edgecolor = '#CFCFCF')
    ax_mr.axes.get_xaxis().set_ticks([])
    ax_mr.set_xlabel('Plots', fontsize = 8)
    ax_mr.set_ylabel(r'$R^2$ of size-abundance relationship', fontsize = 8)
    
    ax_lambda = plt.subplot(122)
    plt.plot(np.arange(len(rand_lambda[0])), rand_lambda[0], 'ko-', markersize = 2)
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
    sites= []
    obs = []
    pred = []
    for i, dataset in enumerate(datasets):
        obs_pred_data = import_obs_pred_data(data_dir + dataset + filename)
        site_list = [dataset + '_' + x for x in list(obs_pred_data['site'])]
        sites.extend(site_list)
        obs.extend(list(obs_pred_data['obs']))
        pred.extend(list(obs_pred_data['pred']))
    return np.array(sites), np.array(obs), np.array(pred)
        
def plot_obs_pred(obs, pred, radius, loglog, ax = None, inset = False, sites = None):
    """Generic function to generate an observed vs predicted figure with 1:1 line"""
    if not ax:
        fig = plt.figure(figsize = (3.5, 3.5))
        ax = plt.subplot(111)

    axis_min = 0.9 * min(list(obs[obs > 0]) + list(pred[pred > 0]))
    if loglog:
        axis_max = 3 * max(list(obs)+list(pred))
    else:
        axis_max = 1.1 * max(list(obs)+list(pred))
    macroecotools.plot_color_by_pt_dens(np.array(pred), np.array(obs), radius, loglog=loglog, plot_obj = ax)      
    plt.plot([axis_min, axis_max],[axis_min, axis_max], 'k-')
    plt.xlim(axis_min, axis_max)
    plt.ylim(axis_min, axis_max)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 6)
    if loglog:
        plt.annotate(r'$R^2$ = %0.2f' %macroecotools.obs_pred_rsquare(np.log10(obs[(obs != 0) * (pred != 0)]), np.log10(pred[(obs != 0) * (pred != 0)])),
                     xy = (0.05, 0.85), xycoords = 'axes fraction', fontsize = 7)
    else:
        plt.annotate(r'$R^2$ = %0.2f' %macroecotools.obs_pred_rsquare(obs, pred),
                     xy = (0.05, 0.85), xycoords = 'axes fraction', fontsize = 7)
    if inset:
        axins = inset_axes(ax, width="30%", height="30%", loc=4)
        if loglog:
            hist_mete_r2(sites[(obs != 0) * (pred != 0)], np.log10(obs[(obs != 0) * (pred != 0)]), 
                         np.log10(pred[(obs != 0) * (pred != 0)]))
        else:
            hist_mete_r2(sites, obs, pred)
        plt.setp(axins, xticks=[], yticks=[])
    return ax

def plot_obs_pred_sad(datasets, data_dir = "./out_files/", out_name = 'obs_pred_sad.png', dest_dir = "", radius = 2, inset = False):
    """Plot the observed vs predicted abundance for each species for multiple datasets."""
    rad_sites, rad_obs, rad_pred = get_obs_pred_from_file(datasets, data_dir, '_obs_pred_rad.csv')
    if inset:
        fig = plot_obs_pred(rad_obs, rad_pred, radius, 1, inset = True, sites = rad_sites)
    else: 
        fig = plot_obs_pred(rad_obs, rad_pred, radius, 1)
    fig.set_xlabel('Predicted abundance', labelpad = 4, size = 8)
    fig.set_ylabel('Observed abundance', labelpad = 4, size = 8)
    plt.savefig(out_name, dpi = 400)

def plot_obs_pred_dbh2(datasets, data_dir = "./out_files/", out_name = 'obs_pred_dbh2.png', radius = 2, inset = False):
    """Plot the observed vs predicted dbh2 for each individual for multiple datasets."""
    dbh2_sites, dbh2_obs, dbh2_pred = get_obs_pred_from_file(datasets, data_dir, '_obs_pred_isd_dbh2.csv')
    if inset:
        fig = plot_obs_pred(dbh2_obs, dbh2_pred, radius, 1, inset = True, sites = dbh2_sites)
    else:
        fig = plot_obs_pred(dbh2_obs, dbh2_pred, radius, 1)
    fig.set_xlabel(r'Predicted $DBH^2$', labelpad = 4, size = 8)
    fig.set_ylabel(r'Observed $DBH^2$', labelpad = 4, size = 8)
    plt.savefig(out_name, dpi = 400)

def plot_obs_pred_cdf(datasets, data_dir = "./out_files/", out_name = 'obs_pred_cdf.png', radius = 0.05, inset = False):
    """Plot the observed vs predicted cdf for multiple datasets."""
    cdf_sites, cdf_obs, cdf_pred = get_obs_pred_from_file(datasets, data_dir, '_obs_pred_isd_cdf.csv')
    if inset:
        fig = plot_obs_pred(cdf_obs, cdf_pred, radius, 0, inset = True, sites = cdf_sites)
    else:
        fig = plot_obs_pred(cdf_obs, cdf_pred, radius, 0)
    fig.set_xlabel('Predicted F(x)', labelpad = 4, size = 8)
    fig.set_ylabel('Observed F(x)', labelpad = 4, size = 8)
    plt.savefig(out_name, dpi = 400)

def plot_obs_pred_freq(datasets, data_dir = "./out_files/", out_name = 'obs_pred_freq.png', radius = 0.05, inset = False):
    """Plot the observed vs predicted size frequency for multiple datasets."""
    freq_sites, freq_obs, freq_pred = get_obs_pred_from_file(datasets, data_dir, '_obs_pred_freq.csv')
    if inset:
        fig = plot_obs_pred(freq_obs, freq_pred, radius, 1, inset = True, sites = freq_sites)
    else:
        fig = plot_obs_pred(freq_obs, freq_pred, radius, 1)
    fig.set_xlabel('Predicted frequency', labelpad = 4, size = 8)
    fig.set_ylabel('Observed frequency', labelpad = 4, size = 8)
    plt.savefig(out_name, dpi = 400)

def plot_obs_pred_avg_mr(datasets, data_dir = "./out_files/", out_name = 'obs_pred_average_mr.png', radius = 2, inset = False):
    """Plot the observed vs predicted species-level average metabolic rate 
    
    for all species across multiple datasets.
    
    """
    mr_sites, mr_obs, mr_pred = get_obs_pred_from_file(datasets, data_dir, '_obs_pred_avg_mr.csv')
    if inset:
        fig = plot_obs_pred(mr_obs, mr_pred, radius, 1, inset = True, sites = mr_sites)
    else:
        fig = plot_obs_pred(mr_obs, mr_pred, radius, 1)
    fig.set_xlabel('Predicted species-average metabolic Rate', labelpad = 4, size = 8)
    fig.set_ylabel('Observed species-average metabolic Rate', labelpad = 4, size = 8)
    plt.savefig(out_name, dpi = 400)

def plot_obs_pred_iisd_par(datasets, data_dir = "./out_files/", out_name = 'intra_par.png', radius = 2, inset = False):
    """Plot the scaled intra-specific energy distribution parameter against abundance."""
    par_sites, par_obs, par_pred = get_obs_pred_from_file(datasets, data_dir, '_par.csv')
    if inset:
        fig = plot_obs_pred(par_obs, par_pred, radius, 1, inset = True, sites = par_sites)
    else:
        fig = plot_obs_pred(par_obs, par_pred, radius, 1)
    fig.set_xlabel('Predicted parameter', labelpad = 4, size = 8)
    fig.set_ylabel('Observed parameter', labelpad = 4, size = 8)
    plt.savefig(out_name, dpi = 400)

def plot_four_patterns(datasets, data_dir = "./out_files/", radius_sad = 2, radius_freq = 0.05, 
                       radius_mr = 2, radius_par = 2, inset = False):
    """Plot predicted versus observed data for 4 patterns (SAD, ISD, abundance-MR relationship, 
    
    scaled parameter for intraspecific MR distribution) as subplots in a single figure.
    
    """
    fig = plt.figure(figsize = (7, 7))
    
    ax = plt.subplot(221)
    rad_sites, rad_obs, rad_pred = get_obs_pred_from_file(datasets, data_dir, '_obs_pred_rad.csv')
    if inset:
        fig1 = plot_obs_pred(rad_obs, rad_pred, radius_sad, 1, ax = ax, inset = True, sites = rad_sites)
    else:
        fig1 = plot_obs_pred(rad_obs, rad_pred, radius_sad, 1, ax = ax)
    fig1.annotate('(A)', xy = (0.05, 0.92), xycoords = 'axes fraction', fontsize = 10)
    fig1.set_xlabel('Predicted abundance', labelpad = 4, size = 8)
    fig1.set_ylabel('Observed abundance', labelpad = 4, size = 8)

    ax = plt.subplot(222)
    freq_sites, freq_obs, freq_pred = get_obs_pred_from_file(datasets, data_dir, '_obs_pred_freq.csv')
    if inset:
        fig2 = plot_obs_pred(freq_obs, freq_pred, radius_freq, 1, ax = ax, inset = True, sites = freq_sites)
    else:
        fig2 = plot_obs_pred(freq_obs, freq_pred, radius_freq, 1, ax = ax)
    fig2.annotate('(B)', xy = (0.05, 0.92), xycoords = 'axes fraction', fontsize = 10)
    fig2.set_xlabel('Predicted frequency', labelpad = 4, size = 8)
    fig2.set_ylabel('Observed frequency', labelpad = 4, size = 8)

    ax = plt.subplot(223)
    mr_sites, mr_obs, mr_pred = get_obs_pred_from_file(datasets, data_dir, '_obs_pred_avg_mr.csv')
    if inset:
        fig3 = plot_obs_pred(mr_obs, mr_pred, radius_mr, 1, ax = ax, inset = True, sites = mr_sites)
    else:
        fig3 = plot_obs_pred(mr_obs, mr_pred, radius_mr, 1, ax = ax)
    fig3.annotate('(C)', xy = (0.05, 0.92), xycoords = 'axes fraction', fontsize = 10)
    fig3.set_xlabel('Predicted species-average metabolic rate', labelpad = 4, size = 8)
    fig3.set_ylabel('Observed species-average metabolic rate', labelpad = 4, size = 8)

    ax = plt.subplot(224)
    par_sites, par_obs, par_pred = get_obs_pred_from_file(datasets, data_dir, '_par.csv')
    if inset:
        fig4 = plot_obs_pred(par_obs, par_pred, radius_par, 1, ax = ax, inset = True, sites = par_sites)
    else:
        fig4 = plot_obs_pred(par_obs, par_pred, radius_par, 1, ax = ax)
    fig4.annotate('(D)', xy = (0.05, 0.92), xycoords = 'axes fraction', fontsize = 10)
    fig4.set_xlabel('Predicted parameter', labelpad = 4, size = 8)
    fig4.set_ylabel('Observed parameter', labelpad = 4, size = 8)

    plt.subplots_adjust(wspace = 0.2, hspace = 0.2)
    plt.savefig('four_patterns.pdf', dpi = 400)    

def plot_four_patterns_ver2(datasets, data_dir = "./out_files/", radius_sad = 2, radius_isd = 2, 
                       radius_mr = 2, radius_iisd = 2, inset = False, out_name = 'four_patterns'):
    """Plot predicted versus observed data for 4 patterns (rank abundance for SAD, 
    
    individual body size for ISD and iISD, average body size for SDR) as subplots in a single figure.
    
    """
    fig = plt.figure(figsize = (7, 7))
    
    ax = plt.subplot(221)
    rad_sites, rad_obs, rad_pred = get_obs_pred_from_file(datasets, data_dir, '_obs_pred_rad.csv')
    if inset:
        fig1 = plot_obs_pred(rad_obs, rad_pred, radius_sad, 1, ax = ax, inset = True, sites = rad_sites)
    else:
        fig1 = plot_obs_pred(rad_obs, rad_pred, radius_sad, 1, ax = ax)
    fig1.annotate('(A)', xy = (0.05, 0.92), xycoords = 'axes fraction', fontsize = 10)
    fig1.set_xlabel('Predicted abundance', labelpad = 4, size = 8)
    fig1.set_ylabel('Observed abundance', labelpad = 4, size = 8)

    ax = plt.subplot(222)
    isd_sites, isd_obs, isd_pred = get_obs_pred_from_file(datasets, data_dir, '_obs_pred_isd_dbh2.csv')
    if inset:
        fig2 = plot_obs_pred(isd_obs, isd_pred, radius_isd, 1, ax = ax, inset = True, sites = isd_sites)
    else:
        fig2 = plot_obs_pred(isd_obs, isd_pred, radius_isd, 1, ax = ax)
    fig2.annotate('(B)', xy = (0.05, 0.92), xycoords = 'axes fraction', fontsize = 10)
    fig2.set_xlabel(r'Predicted $DBH^2$ from ISD', labelpad = 4, size = 8)
    fig2.set_ylabel(r'Observed $DBH^2$', labelpad = 4, size = 8)

    ax = plt.subplot(223)
    mr_sites, mr_obs, mr_pred = get_obs_pred_from_file(datasets, data_dir, '_obs_pred_avg_mr.csv')
    if inset:
        fig3 = plot_obs_pred(mr_obs, mr_pred, radius_mr, 1, ax = ax, inset = True, sites = mr_sites)
    else:
        fig3 = plot_obs_pred(mr_obs, mr_pred, radius_mr, 1, ax = ax)
    fig3.annotate('(C)', xy = (0.05, 0.92), xycoords = 'axes fraction', fontsize = 10)
    fig3.set_xlabel('Predicted species-average metabolic rate', labelpad = 4, size = 8)
    fig3.set_ylabel('Observed species-average metabolic rate', labelpad = 4, size = 8)

    ax = plt.subplot(224)
    iisd_sites, iisd_obs, iisd_pred = get_obs_pred_from_file(datasets, data_dir, '_obs_pred_iisd_dbh2.csv')
    if inset:
        fig4 = plot_obs_pred(iisd_obs, iisd_pred, radius_iisd, 1, ax = ax, inset = True, sites = iisd_sites)
    else:
        fig4 = plot_obs_pred(iisd_obs, iisd_pred, radius_iisd, 1, ax = ax)
    fig4.annotate('(D)', xy = (0.05, 0.92), xycoords = 'axes fraction', fontsize = 10)
    fig4.set_xlabel(r'Predicted $DBH^2$ from iISD', labelpad = 4, size = 8)
    fig4.set_ylabel(r'Observed $DBH^2$', labelpad = 4, size = 8)

    plt.subplots_adjust(wspace = 0.2, hspace = 0.2)
    plt.savefig(out_name + '.pdf', format = 'pdf', dpi = 400)
    
def plot_four_patterns_single(datasets, outfile, data_dir = "./out_files/", radius_sad = 2, 
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
            fig1.annotate('(A)', xy = (0.05, 0.92), xycoords = 'axes fraction', fontsize = 10)
            fig1.set_xlabel('Predicted abundance', labelpad = 4, size = 8)
            fig1.set_ylabel('Observed abundance', labelpad = 4, size = 8)
        
            ax = plt.subplot(222)
            fig2 = plot_obs_pred(list(dat_freq_site['obs']), list(dat_freq_site['pred']), radius_freq, 1, ax = ax)
            fig2.annotate('(B)', xy = (0.05, 0.92), xycoords = 'axes fraction', fontsize = 10)
            fig2.set_xlabel('Predicted frequency', labelpad = 4, size = 8)
            fig2.set_ylabel('Observed frequency', labelpad = 4, size = 8)
        
            ax = plt.subplot(223)
            fig3 = plot_obs_pred(list(dat_mr_site['obs']), list(dat_mr_site['pred']), radius_mr, 1, ax = ax)
            fig3.annotate('(C)', xy = (0.05, 0.92), xycoords = 'axes fraction', fontsize = 10)
            fig3.set_xlabel('Predicted species-average metabolic rate', labelpad = 4, size = 8)
            fig3.set_ylabel('Observed species-average metabolic rate', labelpad = 4, size = 8)
        
            ax = plt.subplot(224)
            fig4 = plot_obs_pred(list(dat_par_site['obs']), list(dat_par_site['pred']), radius_par, 1, ax = ax)
            fig4.annotate('(D)', xy = (0.05, 0.92), xycoords = 'axes fraction', fontsize = 10)
            fig4.set_xlabel('Predicted parameter', labelpad = 4, size = 8)
            fig4.set_ylabel('Observed parameter', labelpad = 4, size = 8)
        
            plt.subplots_adjust(wspace = 0.2, hspace = 0.2)
            plt.suptitle(dataset + ',' + site)
            plt.savefig(pp, format = 'pdf', dpi = 400)
    pp.close()

def plot_four_patterns_single_ver2(datasets, outfile, data_dir = "./out_files/", radius_par = 2, title = True,
                                   bin_size = 1.7, cutoff = 9, n_cutoff = 4):
    """Version two of four-pattern figure at plot level."""
    pp = PdfPages(outfile)
    for dataset in datasets:
        dat_rad = import_obs_pred_data(data_dir + dataset + '_obs_pred_rad.csv')
        dat_raw = import_raw_data('./data/'+ dataset + '.csv')
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
                plt.annotate(r'$R^2$ = %0.2f' %macroecotools.obs_pred_rsquare(np.log10(obs[(obs != 0) * (pred != 0)]), np.log10(pred[(obs != 0) * (pred != 0)])),
                             xy = (0.72, 0.9), xycoords = 'axes fraction', fontsize = 7)
                plt.annotate('(A)', xy = (0.05, 0.92), xycoords = 'axes fraction', fontsize = 10)
                
                ax = plt.subplot(222)
                ax.loglog(np.arange(1, ceil(max(dbh2_scale_site)) + 1), psi_pdf, '#9400D3', linewidth = 2)
                ax.bar(bin_size ** np.array(range(num_bin)), emp_pdf, color = '#d6d6d6', 
                       width = 0.4 * bin_size ** np.array(range(num_bin)))
                plt.ylim((max(min(emp_pdf), 10 ** -10), 1))
                ax.tick_params(axis = 'both', which = 'major', labelsize = 6)
                plt.xlabel(r'$DBH^2$', fontsize = 8)
                plt.ylabel('Frequency', fontsize = 8)
                obs = dat_freq_site['obs']
                pred = dat_freq_site['pred']
                plt.annotate(r'$R^2$ = %0.2f' %macroecotools.obs_pred_rsquare(np.log10(obs[(obs != 0) * (pred != 0)]), np.log10(pred[(obs != 0) * (pred != 0)])),
                             xy = (0.72, 0.9), xycoords = 'axes fraction', fontsize = 7)
                plt.annotate('(B)', xy = (0.05, 0.92), xycoords = 'axes fraction', fontsize = 10)
 
                ax = plt.subplot(223)
                ax.loglog(np.array(dbh2_pred), np.array(n_list), color = '#9400D3', linewidth = 2)
                ax.scatter(dbh2_obs, n_obs, color = '#999999', marker = 'o')
                ax.tick_params(axis = 'both', which = 'major', labelsize = 6)
                plt.xlabel('Species-average metabolic rate', fontsize = 8)
                plt.ylabel('Species abundance', fontsize = 8)
                obs = dat_mr_site['obs']
                pred = dat_mr_site['pred']
                plt.annotate(r'$R^2$ = %0.2f' %macroecotools.obs_pred_rsquare(np.log10(obs[(obs != 0) * (pred != 0)]), np.log10(pred[(obs != 0) * (pred != 0)])),
                             xy = (0.05, 0.85), xycoords = 'axes fraction', fontsize = 7)
                plt.annotate('(C)', xy = (0.05, 0.92), xycoords = 'axes fraction', fontsize = 10)
                
                ax = plt.subplot(224)
                fig4 = plot_obs_pred(dat_par_site['obs'], dat_par_site['pred'], radius_par, 1, ax = ax)
                fig4.set_xlabel('Predicted parameter', labelpad = 4, size = 8)
                fig4.set_ylabel('Observed parameter', labelpad = 4, size = 8)
                fig4.annotate('(D)', xy = (0.05, 0.92), xycoords = 'axes fraction', fontsize = 10)
                
                plt.subplots_adjust(wspace = 0.2, hspace = 0.2)
                if title:
                    plt.suptitle(dataset + ',' + site)
                plt.savefig(pp, format = 'pdf', dpi = 400)
    pp.close()

def plot_four_patterns_single_ver3(datasets, outfile, data_dir = "./out_files/", radius_par = 2, title = True,
                                   bin_size = 1.7, cutoff = 9, n_cutoff = 4):
    """Version three of four-pattern figure at plot level (consist with plot_four_patterns_ver2)"""
    pp = PdfPages(outfile)
    for dataset in datasets:
        dat_rad = import_obs_pred_data(data_dir + dataset + '_obs_pred_rad.csv')
        dat_raw = import_raw_data('./data/' + dataset + '.csv')
        dat_isd = import_obs_pred_data(data_dir + dataset + '_obs_pred_isd_dbh2.csv')
        dat_mr = import_obs_pred_data(data_dir + dataset + '_obs_pred_avg_mr.csv')
        dat_iisd = import_obs_pred_data(data_dir + dataset + '_obs_pred_iisd_dbh2.csv')
        sites = np.sort(list(set(dat_rad['site'])))
        for site in sites:
            dat_raw_site = dat_raw[dat_raw['site'] == site]
            sp_list = set(dat_raw_site['sp'])
            S0 = len(sp_list)
            if S0 > cutoff:
                dat_rad_site = dat_rad[dat_rad['site'] == site]
                dat_isd_site = dat_isd[dat_isd['site'] == site]
                dat_mr_site = dat_mr[dat_mr['site'] == site]
                dat_iisd_site = dat_iisd[dat_iisd['site'] == site]

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
                plt.annotate(r'$R^2$ = %0.2f' %macroecotools.obs_pred_rsquare(np.log10(obs[(obs != 0) * (pred != 0)]), np.log10(pred[(obs != 0) * (pred != 0)])),
                             xy = (0.72, 0.9), xycoords = 'axes fraction', fontsize = 7)
                plt.annotate('(A)', xy = (0.05, 0.92), xycoords = 'axes fraction', fontsize = 10)
                
                ax = plt.subplot(222)
                ax.loglog(np.arange(1, ceil(max(dbh2_scale_site)) + 1), psi_pdf, '#9400D3', linewidth = 2)
                ax.bar(bin_size ** np.array(range(num_bin)), emp_pdf, color = '#d6d6d6', 
                       width = 0.4 * bin_size ** np.array(range(num_bin)))
                plt.ylim((max(min(emp_pdf), 10 ** -10), 1))
                ax.tick_params(axis = 'both', which = 'major', labelsize = 6)
                plt.xlabel(r'$DBH^2$', fontsize = 8)
                plt.ylabel('Frequency', fontsize = 8)
                obs = dat_isd_site['obs']
                pred = dat_isd_site['pred']
                plt.annotate(r'$R^2$ = %0.2f' %macroecotools.obs_pred_rsquare(np.log10(obs), np.log10(pred)),
                             xy = (0.72, 0.9), xycoords = 'axes fraction', fontsize = 7)
                plt.annotate('(B)', xy = (0.05, 0.92), xycoords = 'axes fraction', fontsize = 10)
 
                ax = plt.subplot(223)
                ax.loglog(np.array(dbh2_pred), np.array(n_list), color = '#9400D3', linewidth = 2)
                ax.scatter(dbh2_obs, n_obs, color = '#999999', marker = 'o')
                ax.tick_params(axis = 'both', which = 'major', labelsize = 6)
                plt.xlabel('Species-average metabolic rate', fontsize = 8)
                plt.ylabel('Species abundance', fontsize = 8)
                obs = dat_mr_site['obs']
                pred = dat_mr_site['pred']
                plt.annotate(r'$R^2$ = %0.2f' %macroecotools.obs_pred_rsquare(np.log10(obs[(obs != 0) * (pred != 0)]), np.log10(pred[(obs != 0) * (pred != 0)])),
                             xy = (0.05, 0.85), xycoords = 'axes fraction', fontsize = 7)
                plt.annotate('(C)', xy = (0.05, 0.92), xycoords = 'axes fraction', fontsize = 10)
                
                ax = plt.subplot(224)
                fig4 = plot_obs_pred(dat_iisd_site['obs'], dat_iisd_site['pred'], radius_par, 1, ax = ax)
                fig4.set_xlabel(r'Predicted $DBH^2$ from iISD', labelpad = 4, size = 8)
                fig4.set_ylabel(r'Observed $DBH^2$', labelpad = 4, size = 8)
                fig4.annotate('(D)', xy = (0.05, 0.92), xycoords = 'axes fraction', fontsize = 10)
                
                plt.subplots_adjust(wspace = 0.2, hspace = 0.2)
                if title:
                    plt.suptitle(dataset + ',' + site)
                plt.savefig(pp, format = 'pdf', dpi = 200)
    pp.close()

def comp_isd(datasets, list_of_datasets, data_dir = "./out_files/"):
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
                     r'$r^2$ = %0.2f' %macroecotools.obs_pred_rsquare(np.log10(obs[(obs != 0) * (pred != 0)]), np.log10(pred[(obs != 0) * (pred != 0)])))
            plt.subplots_adjust(wspace = 0.55, bottom = 0.3)
            plt.savefig(data_dir + list_of_datasets[j] + '_' + site + '_comp_cdf.png', dpi = 400)
    
def plot_fig1(dat_name = 'BCI', sp_name = 'Hybanthus prunifolius', dat_dir = './out_files/', output_dir = ""):
    """Illustration of the 4 patterns.
    
    The figure in the paper is produced with data from BCI, and the iISD 
    pattern is created using data for the most abundant species, Hybanthus prunifolius.
    """
    fig = plt.figure(figsize = (7, 7))
    # Subplot A: Observed and predicted RAD
    # Code adopted and modified from example_sad_plot in mete_sads
    obs_pred_data = import_obs_pred_data(dat_dir + dat_name + '_obs_pred_rad.csv')    
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
    plot_obj.annotate('(A)', xy = (0.05, 0.92), xycoords = 'axes fraction', fontsize = 10)
    # Subplot B: ISD shown as histogram with predicted pdf
    # Code adopted and modified from plot_ind_hist from fit_other_dist
    dat = import_raw_data('./data/' + dat_name + '.csv')
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
    plot_obj.annotate('(B)', xy = (0.05, 0.92), xycoords = 'axes fraction', fontsize = 10)
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
    plot_obj.annotate('(C)', xy = (0.05, 0.92), xycoords = 'axes fraction', fontsize = 10)
    # Subplot D: Intra-specific distribution for one species
    hp_dbh2 = dbh2_scale[dat[dat.dtype.names[1]] == sp_name]
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
    plot_obj.bar(1.7 ** np.array(range(hp_num_bin)), hp_emp_pdf, color = '#d6d6d6', 
        width = 0.4 * 1.7 ** np.array(range(hp_num_bin)))
    plt.ylim((max(min(hp_emp_pdf), 10 ** -10), 1))
    plot_obj.tick_params(axis = 'both', which = 'major', labelsize = 6)
    plt.xlabel(r'$DBH^2$', fontsize = 8)
    plt.ylabel('Probability Density', fontsize = 8)
    plot_obj.annotate('(D)', xy = (0.05, 0.92), xycoords = 'axes fraction', fontsize = 10)
    plt.subplots_adjust(wspace = 0.29, hspace = 0.29)
    plt.savefig(output_dir + 'Figure 1.pdf', format = 'pdf', dpi = 300)

def get_weights_all(datasets, list_of_dataset_names, par_table, data_dir = './out_files/'):
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

def plot_hist(datasets, list_of_dataset_names, par_table, data_dir = './out_files/'):
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

def plot_dens_ks_test(p_list, sig_level, out_fig = 'Density_KS_test.png'):
    """Create an across-species density plot of the proportion of significant Kolmogorov-Smirnov tests.
    
    Input:
    p_list - a list of proportions of significant KS tests. One value for each species.
    sig_level - significance level at which the KS tests were conducted.
    
    """
    fig = plt.figure(figsize = (7, 7))
    hist_p = np.histogram(p_list, 20, range=(0, 1))
    densities = hist_p[0] / len(p_list)
    plt.bar(np.arange(0, 1, 0.05), densities, width = 0.05, color = '#d6d6d6')
    plt.xlabel('Proportion of significant tests within species', fontsize = 12)
    plt.ylabel('Density', fontsize = 12)
    plt.vlines(sig_level, 0, 1, color = 'k', linestyle = 'dashed', linewidth = 2)
    plt.axis([0, 1, 0, 1])
    plt.annotate('Species with non-exponential iISD: %s%%' %(round(len(p_list[np.array(p_list) >= sig_level])/len(p_list)*100, 2)),
                 xy = (0.3, 0.7), xycoords = 'axes fraction', color = 'k')

    plt.savefig(out_fig, dpi = 400)

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

def plot_spp_exp(datasets, list_of_dataset_names, data_dir = './out_files/'):
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

def plot_species_avg(datasets, list_of_dataset_names, data_dir = './out_files/'):
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

def plot_spp_frequency(dat, spp_name1, spp_name2, data_dir = './out_files/'):
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

def plot_spp_frequency_pdf(dat, spp_name1, spp_name2, data_dir = './out_files/'):
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

def plot_fig_A5():
    """Create a 1*2 plot comparing the empirical vs. randomized size-density relationship using BCI data"""
    dat = import_raw_data('./data/BCI.csv')
    dbh_scale = np.array(dat['dbh'] / min(dat['dbh']))
    dbh2_scale = dbh_scale ** 2
    E0 = sum(dbh2_scale)
    N0 = len(dbh2_scale)
    S_list = np.unique(dat['sp'])
    S0 = len(S_list)
    n_list = []
    emp_dbh2_list = []
    rand_dbh2_list = []
    for sp in S_list:
        sp_dbh2 = dbh2_scale[dat['sp'] == sp]
        n_list.append(len(sp_dbh2))
        emp_dbh2_list.append(sum(sp_dbh2) / len(sp_dbh2))
    np.random.shuffle(dat['sp'])
    for sp in S_list:
        sp_dbh2 = dbh2_scale[dat['sp'] == sp]
        rand_dbh2_list.append(sum(sp_dbh2) / len(sp_dbh2))
    theta_epsilon_obj = theta_epsilon(S0, N0, E0)
    dbh2_pred = [theta_epsilon_obj.E(n) for n in n_list]
    
    fig = plt.figure(figsize = (7, 3.8)) 
    ax_emp = plt.subplot(121)
    ax_emp.loglog(np.array(sorted(dbh2_pred, reverse = True)), np.array(sorted(n_list)), color = '#9400D3', linewidth = 2)
    ax_emp.scatter(np.array(emp_dbh2_list), np.array(n_list), color = '#999999', marker = 'o')
    ax_emp.tick_params(axis = 'both', which = 'major', labelsize = 6)
    plt.xlabel('Species-average metabolic rate', fontsize = 8)
    plt.ylabel('Species abundance', fontsize = 8)
    plt.annotate(r'$R^2$ = %0.2f' %macroecotools.obs_pred_rsquare(np.log10(emp_dbh2_list), np.log10(dbh2_pred)),
                 xy = (0.05, 0.85), xycoords = 'axes fraction', fontsize = 7)
    plt.title('BCI, empirical')
    
    ax_rand = plt.subplot(122)
    ax_rand.loglog(np.array(sorted(dbh2_pred, reverse = True)), np.array(sorted(n_list)), color = '#9400D3', linewidth = 2)
    ax_rand.scatter(np.array(rand_dbh2_list), np.array(n_list), color = '#999999', marker = 'o')
    ax_rand.tick_params(axis = 'both', which = 'major', labelsize = 6)
    plt.xlabel('Species-average metabolic rate', fontsize = 8)
    plt.ylabel('Species abundance', fontsize = 8)
    plt.annotate(r'$R^2$ = %0.2f' %macroecotools.obs_pred_rsquare(np.log10(rand_dbh2_list), np.log10(dbh2_pred)),
                 xy = (0.05, 0.85), xycoords = 'axes fraction', fontsize = 7)
    plt.title('BCI, randomized')
    
    plt.savefig('Fig_A5.png', dpi = 400)

def get_site_for_alt_analysis(dat_list, density_dic, list_of_sites_and_forest_types, cutoff = 0.3):
    """Obtain the list of sites to be used for the analysis using alternative
    
    metabolic scaling, where the proportion of individuals with density information 
    is above the designated cutoff threshold.
    
    Inputs:
    dat_list - list of dataset names
    density_dic - a dictionary holding the Latin binomial of species and their densities
    list_of_sites_and_forest_types - input file name with three columns - dat_name, site, forest type
    cutoff - the maximal proportion of individials in the community without density values
             for the community to be included, default at 0.3.
    
    """
    site_types = np.genfromtxt(list_of_sites_and_forest_types, dtype = "S15,S15,S15",
                              names = ['dat_name','site','forest_type'],
                              delimiter = ",")    
    list_of_site = []
    for dat_name in dat_list:
        if dat_name in site_types['dat_name']:
            site_type_dat = site_types[site_types['dat_name'] == dat_name]
            dat_i = import_raw_data('./data/' + dat_name + '.csv')
            sites_i = np.unique(dat_i['site'])
            for site in sites_i:
                if site in site_type_dat['site']:
                    type_site = site_type_dat[site_type_dat['site'] == site]['forest_type'][0]
                    dat_site = dat_i[dat_i['site'] == site]
                    records_with_no_wsg = 0
                    for record in dat_site:
                        if record[1] not in density_dic:
                            records_with_no_wsg += 1
                    if records_with_no_wsg <= cutoff * len(dat_site):
                        list_of_site.append([dat_name, site, type_site])
    
    list_of_site_structured = np.zeros(len(list_of_site), 
                                       dtype =[('dat_name', 'S15'), ('site', 'S15'), ('forest_type', 'S15')])
    for i, dat_site_trio in enumerate(list_of_site):
        list_of_site_structured['dat_name'][i] = dat_site_trio[0]
        list_of_site_structured['site'][i] = dat_site_trio[1]
        list_of_site_structured['forest_type'][i] = dat_site_trio[2]   
    return list_of_site_structured

def dbh_to_mass(dbh, wsg, forest_type):
    """Convert DBH to biomass based on formulas taken from Chave et al. 2005.
    
    Input: 
    dbh - diameter at breast height (cm)
    wsg - wood specific gravity (g/cm**3)
    forest_type - forest type that determines fitting parameters. 
                  Can take one of the four values: "dry", "moist", "wet" and "mangrove". 
    Output: above ground biomass (kg)
    
    """
    if forest_type == "dry":
        a, b = -0.667, 1.784
    elif forest_type == "moist":
        a, b = -1.499, 2.148
    elif forest_type == "wet":
        a, b = -1.239, 1.98
    elif forest_type == "mangrove":
        a, b = -1.349, 1.98
    else: 
        print "Error: unidentified forest type."
        return None
    c, d = 0.207, -0.0281
    return wsg * exp(a + b * log(dbh) + c * log(dbh) ** 2 + d * log(dbh) ** 3)

def mass_to_resp(agb):
    """Convert aboveground biomass (kg) to respiration rate (miu-mol/s) based on Mori et al. 2010 Eqn 2"""
    G, H, g, h = 6.69, 0.421, 1.082, 0.78
    return 1 / (1 / (G * agb ** g) + 1 / (H * agb ** h))

def get_mr_alt(list_for_analysis, density_dic, data_dir = './data/'):
    """Alternative method to obtain metabolic rate from dbh.
    
    Input:
    list_for_analysis: a list with [dataset_name, site_name] for the alternative analysis
    density_dic: a dictionary holding wood density (wsg) of each species.
    data_dir: directory to save the output.
    cutoff: maximum proportion of records that are species not included in density_dic for the dataset to be dropped.
    Output:
    A csv file for each site with columns 'site', 'sp', and 'mr'.
    
    """
    mean_msg = sum(density_dic.values()) / len(density_dic.values())
    
    for dat_name in np.unique(list_for_analysis['dat_name']):
        raw_data = import_raw_data('./data/' + dat_name + '.csv')
        f_write = open(data_dir + dat_name + '_alt.csv', 'wb')
        f = csv.writer(f_write)
        col_names = np.zeros(1, dtype = ('S15, S15, S15'))
        col_names['f0'] = 'site'
        col_names['f1'] = 'sp'
        col_names['f2'] = 'mr_square_root'
        f.writerows(col_names)
            
        sites_for_dat = list_for_analysis[list_for_analysis['dat_name'] == dat_name]
        for site_name in sites_for_dat['site']:
            site_type = sites_for_dat[sites_for_dat['site'] == site_name]['forest_type'][0]
            dat_site = raw_data[raw_data['site'] == site_name]
            mr_list = []
            for record in dat_site:
                if record['sp'] in density_dic:
                    msg_record = density_dic[record['sp']]
                else: 
                    msg_record = mean_msg
                mass_record = dbh_to_mass(record['dbh'], msg_record, site_type)
                mr_list.append(mass_to_resp(mass_record))
            results = np.zeros(len(dat_site), dtype = ('S15, S25, f8'))
            results['f0'] = dat_site['site']
            results['f1'] = dat_site['sp']
            results['f2'] = np.array(mr_list) ** 0.5 # Square-root to be consistent with dbh in raw_data
            f.writerows(results)
        f_write.close()

def AICc_ISD(dat, expon_par, pareto_par, weibull_k, weibull_lmd):
    """Calculates the AICc for the four ISDs:
    
    truncated expontial, truncated Pareto, truncated Weibull, METE.
    
    """
    dbh_raw = dat['dbh']
    dbh_scale = np.array(dbh_raw / min(dbh_raw))
    dbh2_scale = dbh_scale ** 2

    N0 = len(dbh2_scale)
    E0 = sum(dbh2_scale)
    S0 = len(set(dat['sp']))
    ll_expon = sum(np.log(xsquare_pdf(dbh2_scale, trunc_expon, expon_par, 1)))
    ll_pareto = sum(np.log(xsquare_pdf(dbh2_scale, trunc_pareto, pareto_par, 1)))
    ll_weibull = sum(np.log(xsquare_pdf(dbh2_scale, trunc_weibull, weibull_k, weibull_lmd, 1)))
    ll_list = [ll_expon, ll_pareto, ll_weibull]
    k_list = [2, 2, 3]
    AICc_list = []
    for i, ll in enumerate(ll_list):
        AICc_dist = macroecotools.AICc(k_list[i], ll, N0)
        AICc_list.append(AICc_dist)
    psi = psi_epsilon(S0, N0, E0)
    ll_psi = 0
    for dbh2 in dbh2_scale:
        ll_psi += log(psi.pdf(dbh2))
    AICc_psi = macroecotools.AICc(3, ll_psi, N0)
    AICc_list.append(AICc_psi)
    return np.array(AICc_list)

def AICc_ISD_to_file(dat_list, par_file, cutoff = 9, outfile = 'ISD_comp_all_sites.csv'):
    """Obtain AICc value for the four ISDs and write to file"""
    f = open(outfile, 'wb')
    f_writer = csv.writer(f)    
    pars = np.genfromtxt(par_file, dtype = "S15, S15, f8, f8, f8, f8",
                         names = ['dat','site','expon_par','pareto_par','weibull_k','weibull_lmd'],
                                  delimiter = ",")
    for dat_name in dat_list:
        dat_i = import_raw_data('./data/' + dat_name + '.csv')
        for site in np.unique(dat_i['site']):
            dat_site = dat_i[dat_i['site'] == site]
            S = len(np.unique(dat_site['sp']))
            if S > cutoff:
                par_row = pars[(pars['dat'] == dat_name) * (pars['site'] == site)]
                expon_par, pareto_par, weibull_k, weibull_lmd = par_row['expon_par'], par_row['pareto_par'], par_row['weibull_k'], par_row['weibull_lmd']
                AICc_list_site = AICc_ISD(dat_site, expon_par, pareto_par, weibull_k, weibull_lmd)
                results = np.zeros((1, ), dtype = ('S15, S15, f8, f8, f8, f8'))
                results['f0'] = dat_name
                results['f1'] = site
                results['f2'] = AICc_list_site[0]
                results['f3'] = AICc_list_site[1]
                results['f4'] = AICc_list_site[2]
                results['f5'] = AICc_list_site[3]
                f_writer.writerows(results)
    f.close()

def write_to_file(path, content, new_line = True):
    """Function to facilitate writing to file to avoid redundancy"""
    out_file =  open(path, 'a')
    if new_line:
        print>>out_file , content
    else:
        print>>out_file , content,
    out_file.close()
               
def get_sample_stats_sad(obs, pred, p, upper_bound):
    """Function that returns the three statistics for sample"""
    dat_rsquare = macroecotools.obs_pred_rsquare(np.log10(obs), np.log10(pred))
    dat_loglik = sum(np.log([mdis.trunc_logser.pmf(x, p, upper_bound) for x in obs]))
    emp_cdf = macroecotools.get_emp_cdf(obs)
    dat_ks = max(abs(emp_cdf - np.array([mdis.trunc_logser.cdf(x, p, upper_bound) for x in obs])))
    return dat_rsquare, dat_loglik, dat_ks

def bootstrap_SAD(dat_name, cutoff = 9, Niter = 500):
    """Compare the goodness of fit of the empirical SAD to 
    
    that of the boostrapped samples from the proposed METE distribution.
    Inputs:
    dat_name - name of study
    cutoff - minimum number of species required to run - 1
    Niter - number of bootstrap samples
    """
    dat = import_raw_data('./data/' + dat_name + '.csv')
    site_list = np.unique(dat['site'])
    dat_obs_pred = import_obs_pred_data('./out_files/' + dat_name + '_obs_pred_rad.csv')
        
    for site in site_list:
        out_list_rsquare, out_list_loglik, out_list_ks = [dat_name, site], [dat_name, site], [dat_name, site]
        dat_site = dat[dat['site'] == site]
        S_list = set(dat_site['sp'])
        S0 = len(S_list)
        if S0 > cutoff:
            N0 = len(dat_site)
            dbh_scale = np.array(dat_site['dbh'] / min(dat_site['dbh']))
            dbh2_scale = dbh_scale ** 2
            E0 = sum(dbh2_scale)
            beta = get_beta(S0, N0)
            
            dat_site_obs_pred = dat_obs_pred[dat_obs_pred['site'] == site]
            dat_site_obs = dat_site_obs_pred['obs']
            dat_site_pred = dat_site_obs_pred['pred']
            
            emp_rsquare, emp_loglik, emp_ks = get_sample_stats_sad(dat_site_obs, dat_site_pred, exp(-beta), N0)
            out_list_rsquare.append(emp_rsquare)
            out_list_loglik.append(emp_loglik)
            out_list_ks.append(emp_ks)
            
            for i in range(Niter):
                sample_i = sorted(mdis.trunc_logser.rvs(exp(-beta), N0, size = S0), reverse = True)
                sample_rsquare, sample_loglik, sample_ks = get_sample_stats_sad(sample_i, dat_site_pred, exp(-beta), N0)
                out_list_rsquare.append(sample_rsquare)
                out_list_loglik.append(sample_loglik)
                out_list_ks.append(sample_ks)
  
            write_to_file('./out_files/SAD_bootstrap_rsquare.txt', ",".join(str(x) for x in out_list_rsquare))
            write_to_file('./out_files/SAD_bootstrap_loglik.txt', ",".join(str(x) for x in out_list_loglik))
            write_to_file('./out_files/SAD_bootstrap_ks.txt', ",".join(str(x) for x in out_list_ks))

def generate_isd_sample(psi):
    """Function for parallel computing called in bootstrap_rsquare_loglik_ISD"""
    np.random.seed()
    rand_cdf = scipy.stats.uniform.rvs(size = 100)
    rand_sample = [psi.ppf(x) for x in rand_cdf]
    # Generate one hundred random numbers at a time
    return rand_cdf, rand_sample

def get_sample_stats_isd(obs, pred, psi):
    """Function that returns the three statistics for sample"""
    dat_rsquare = macroecotools.obs_pred_rsquare(np.log10(obs), np.log10(pred))
    dat_loglik = sum(np.log([psi.pdf(x) for x in obs]))
    return dat_rsquare, dat_loglik

def bootstrap_ISD(dat_name, cutoff = 9, Niter = 500):
    """Compare the goodness of fit of the empirical ISD to 
    
    that of the boostrapped samples from the proposed METE distribution.
    This function can be very slow for large dataset thus parallel computing 
    is adopted inside the function for speed.
    Note that if parallel computing is not adopted, or if it is implemented outside
    the function, psi_epsilon.ppf() can cause memory leak.
    Inputs:
    dat_name - name of study
    cutoff - minimum number of species required to run - 1
    Niter - number of bootstrap samples
    """
    dat = import_raw_data('./data/' + dat_name + '.csv')
    site_list = np.unique(dat['site'])
    dat_obs_pred = import_obs_pred_data('./out_files/' + dat_name + '_obs_pred_isd_dbh2.csv')
        
    for site in site_list:
        dat_site = dat[dat['site'] == site]
        S_list = set(dat_site['sp'])
        S0 = len(S_list)
        if S0 > cutoff:
            N0 = len(dat_site)
            dbh_scale = np.array(dat_site['dbh'] / min(dat_site['dbh']))
            dbh2_scale = dbh_scale ** 2
            E0 = sum(dbh2_scale)
            psi = psi_epsilon(S0, N0, E0)
            
            dat_site_obs_pred = dat_obs_pred[dat_obs_pred['site'] == site]
            dat_site_obs = dat_site_obs_pred['obs']
            dat_site_pred = dat_site_obs_pred['pred']
            
            emp_rsquare, emp_loglik = get_sample_stats_isd(dat_site_obs, dat_site_pred, psi)
            emp_cdf = macroecotools.get_emp_cdf(dat_site_obs)
            emp_ks = max(abs(emp_cdf - np.array([psi.cdf(x) for x in dat_site_obs])))
            
            del dbh_scale, dbh2_scale, dat_site_obs 
            
            write_to_file('./out_files/ISD_bootstrap_rsquare.txt', ",".join([dat_name, site, str(emp_rsquare)]), new_line = False)
            write_to_file('./out_files/ISD_bootstrap_loglik.txt', ",".join([dat_name, site, str(emp_loglik)]), new_line = False)
            write_to_file('./out_files/ISD_bootstrap_ks.txt', ",".join([dat_name, site, str(emp_ks)]), new_line = False)
            
            num_pools = 8  # Assuming that 8 pools are to be created
            for i in xrange(Niter):
                sample_i = []
                cdf_i = []
                while len(sample_i) < N0:
                    pool = multiprocessing.Pool(num_pools)
                    out_sample = pool.map(generate_isd_sample, [psi for j in xrange(num_pools)])
                    for combo in out_sample:
                        cdf_sublist, sample_sublist = combo
                        sample_i.extend(sample_sublist)
                        cdf_i.extend(cdf_sublist)
                    pool.close()
                    pool.join()
                sample_i = sorted(sample_i[:N0])
                sample_rsquare, sample_loglik = get_sample_stats_isd(sample_i, dat_site_pred, psi)
                cdf_i = sorted(cdf_i[:N0])
                sample_ks = max([abs(x - (i+1)/N0) for i, x in enumerate(cdf_i)])
                
                write_to_file('./out_files/ISD_bootstrap_rsquare.txt', "".join([',', str(sample_rsquare)]), new_line = False)
                write_to_file('./out_files/ISD_bootstrap_loglik.txt', "".join([',', str(sample_loglik)]), new_line = False)
                write_to_file('./out_files/ISD_bootstrap_ks.txt', "".join([',', str(sample_ks)]), new_line = False)
            
            write_to_file('./out_files/ISD_bootstrap_rsquare.txt', '\t')
            write_to_file('./out_files/ISD_bootstrap_loglik.txt', '\t')
            write_to_file('./out_files/ISD_bootstrap_ks.txt', '\t')

def generate_SDR_iISD_sample(input_list):
    dat_site, theta, dbh2_scale, dat_site_pred_sdr, dat_site_pred_iisd, dat_name, site = input_list
    S_list = set(dat_site['sp'])
    sample_i_sdr = []
    sample_i_iisd = []
    sample_i_ks = []
    loglik_i = 0
    for sp in S_list:
        dbh2_site_sp = dbh2_scale[dat_site['sp'] == sp]
        n_sp = len(dbh2_site_sp)
        np.random.seed()
        sample_sp = theta.rvs(n_sp, n_sp)
        sample_i_sdr.append(sum(sample_sp) / n_sp)
        sample_i_iisd.extend(sorted(sample_sp))
        loglik_i += sum(np.log([theta.pdf(x, n_sp) for x in sample_sp]))
        emp_cdf_sp_i = macroecotools.get_emp_cdf(sample_sp)
        ks_sp_i = max(abs(emp_cdf_sp_i - np.array([theta.cdf(x, n_sp) for x in sample_sp])))
        sample_i_ks.append(ks_sp_i)
        
    write_to_file('./out_files/iISD_bootstrap_ks/iISD_bootstrap_ks_' + dat_name + '_' + site + '.txt', ",".join(str(x) for x in sample_i_ks))    
    
    sample_sdr_rsquare = macroecotools.obs_pred_rsquare(np.log10(sample_i_sdr), np.log10(dat_site_pred_sdr))
    sample_iisd_rsquare = macroecotools.obs_pred_rsquare(np.log10(sample_i_iisd), np.log10(dat_site_pred_iisd))
    return sample_sdr_rsquare, sample_iisd_rsquare, loglik_i
    
def bootstrap_SDR_iISD(dat_name, cutoff = 9, Niter = 500):
    """Compare the goodness of fit of the empirical SDR and iISD to 
    
    that of the boostrapped samples from the proposed METE distribution.
    Inputs:
    dat_name - name of study
    cutoff - minimum number of species required to run - 1
    Niter - number of bootstrap samples
    """
    if not os.path.exists('./out_files/iISD_bootstrap_ks/'):
        os.makedirs('./out_files/iISD_bootstrap_ks/')
    dat = import_raw_data('./data/' + dat_name + '.csv')
    site_list = np.unique(dat['site'])
    dat_obs_pred_sdr = import_obs_pred_data('./out_files/' + dat_name + '_obs_pred_avg_mr.csv')
    dat_obs_pred_iisd = import_obs_pred_data('./out_files/' + dat_name + '_obs_pred_iisd_dbh2.csv')
        
    for site in site_list:
        dat_site = dat[dat['site'] == site]
        S_list = set(dat_site['sp'])
        S0 = len(S_list)
        if S0 > cutoff:
            N0 = len(dat_site)
            dbh_scale = np.array(dat_site['dbh'] / min(dat_site['dbh']))
            dbh2_scale = dbh_scale ** 2
            E0 = sum(dbh2_scale)
            theta = theta_epsilon(S0, N0, E0)
            
            dat_site_obs_pred_sdr = dat_obs_pred_sdr[dat_obs_pred_sdr['site'] == site]
            dat_site_obs_sdr = dat_site_obs_pred_sdr['obs']
            dat_site_pred_sdr = dat_site_obs_pred_sdr['pred']
            
            dat_site_obs_pred_iisd = dat_obs_pred_iisd[dat_obs_pred_iisd['site'] == site]
            dat_site_obs_iisd = dat_site_obs_pred_iisd['obs']
            dat_site_pred_iisd = dat_site_obs_pred_iisd['pred']
            
            emp_sdr_rsquare = macroecotools.obs_pred_rsquare(np.log10(dat_site_obs_sdr), np.log10(dat_site_pred_sdr))
            emp_iisd_rsquare = macroecotools.obs_pred_rsquare(np.log10(dat_site_obs_iisd), np.log10(dat_site_pred_iisd))
                       
            out_list_sdr_rsquare = [dat_name, site, emp_sdr_rsquare]
            out_list_iisd_rsquare = [dat_name, site, emp_iisd_rsquare]
            
            emp_iisd_loglik = 0
            emp_ks_list = []
            n_list = []
            for i, sp in enumerate(S_list):
                dbh2_site_sp = dbh2_scale[dat_site['sp'] == sp]
                n_sp = len(dbh2_site_sp)
                n_list.append(n_sp)
                emp_iisd_loglik += sum(np.log([theta.pdf(x, n_sp) for x in dbh2_site_sp]))
                
                emp_cdf = macroecotools.get_emp_cdf(dbh2_site_sp)
                ks_sp = max(abs(emp_cdf - np.array([theta.cdf(x, n_sp) for x in dbh2_site_sp])))
                emp_ks_list.append(ks_sp)
                
            out_list_iisd_loglik = [dat_name, site, emp_iisd_loglik]
            write_to_file('./out_files/iISD_bootstrap_ks/iISD_bootstrap_ks_' + dat_name + '_' + site + '.txt', ",".join(str(x) for x in n_list)) 
            write_to_file('./out_files/iISD_bootstrap_ks/iISD_bootstrap_ks_' + dat_name + '_' + site + '.txt', ",".join(str(x) for x in emp_ks_list)) 
            
            input_list = [dat_site, theta, dbh2_scale, dat_site_pred_sdr, dat_site_pred_iisd, dat_name, site]
            num_pools = 8
            Nround = int(math.floor(Niter / num_pools))
            for i in xrange(Nround):
                pool = multiprocessing.Pool(num_pools)
                out_sample = pool.map(generate_SDR_iISD_sample, [input_list for j in xrange(num_pools)])
                for output in out_sample:
                    sample_sdr_rsquare, sample_iisd_rsquare, loglik_i = output
                    out_list_sdr_rsquare.append(sample_sdr_rsquare)
                    out_list_iisd_rsquare.append(sample_iisd_rsquare)
                    out_list_iisd_loglik.append(loglik_i)
                pool.close()
                pool.join()
            
            Nleft = Niter - num_pools * Nround
            pool = multiprocessing.Pool(Nleft)
            out_sample = pool.map(generate_SDR_iISD_sample, [input_list for j in xrange(Nleft)])
            for output in out_sample:
                sample_sdr_rsquare, sample_iisd_rsquare, loglik_i = output
                out_list_sdr_rsquare.append(sample_sdr_rsquare)
                out_list_iisd_rsquare.append(sample_iisd_rsquare)
                out_list_iisd_loglik.append(loglik_i)
            pool.close()
            pool.join()
            
            write_to_file('./out_files/SDR_bootstrap_rsquare.txt', ",".join(str(x) for x in out_list_sdr_rsquare))
            write_to_file('./out_files/iISD_bootstrap_rsquare.txt', ",".join(str(x) for x in out_list_iisd_rsquare))
            write_to_file('./out_files/iISD_bootstrap_loglik.txt', ",".join(str(x) for x in  out_list_iisd_loglik))
            
def get_pred_isd_point(psi_x):
    psi, x = psi_x
    return psi.ppf(x)

def get_pred_cdf_point(psi_x):
    psi, x = psi_x
    return psi.cdf(x)

def montecarlo_uniform_SAD_ISD(dat_name, cutoff = 9, Niter = 500):
    """Generate Monte Carlo samples from a discrete uniform distribution 
    
    for the SAD and a continuous uniform distribution for the ISD, fit METE to 
    sample data, and compare the goodness-of-fit measures (R-squared and K-S)
    with that of empirical data.
    
    dat_name - name of study
    cutoff - minimum number of species required to run - 1
    Niter - number of bootstrap samples
    """
    dat = import_raw_data('./data/' + dat_name + '.csv')
    site_list = np.unique(dat['site'])
    dat_obs_pred_sad = import_obs_pred_data('./out_files/' + dat_name + '_obs_pred_rad.csv')
    dat_obs_pred_isd = import_obs_pred_data('./out_files/' + dat_name + '_obs_pred_isd_dbh2.csv')
    
    num_pools = 8  # Assuming that 8 pools are to be created
    for site in site_list:
        dat_site = dat[dat['site'] == site]
        S0 = len(set(dat_site['sp']))
        if S0 > cutoff:
            N0 = len(dat_site)
            dbh_scale = np.array(dat_site['dbh'] / min(dat_site['dbh']))
            dbh2_scale = dbh_scale ** 2
            E0 = sum(dbh2_scale)
            psi = psi_epsilon(S0, N0, E0)
            beta = get_beta(S0, N0)
            
            dat_site_obs_pred_sad = dat_obs_pred_sad[dat_obs_pred_sad['site'] == site]
            dat_site_obs_sad = dat_site_obs_pred_sad['obs']
            dat_site_pred_sad = dat_site_obs_pred_sad['pred']
            emp_rsquare_sad, emp_loglik_sad, emp_ks_sad = get_sample_stats_sad(dat_site_obs_sad, dat_site_pred_sad, exp(-beta), N0)
            
            dat_site_obs_pred_isd = dat_obs_pred_isd[dat_obs_pred_isd['site'] == site]
            dat_site_obs_isd = dat_site_obs_pred_isd['obs']
            dat_site_pred_isd = dat_site_obs_pred_isd['pred']
            emp_rsquare_isd = macroecotools.obs_pred_rsquare(np.log10(dat_site_obs_isd), np.log10(dat_site_pred_isd))
            emp_cdf_isd = sorted(macroecotools.get_emp_cdf(dat_site_obs_isd))
            
            # Obtain predicted cdf is one of the time-limiting steps. Use multiprocessing. 
            pool = multiprocessing.Pool(num_pools)
            emp_cdf_isd_pred = pool.map(get_pred_cdf_point, [(psi, x) for x in dat_site_obs_isd])
            pool.close()
            pool.join()
            emp_cdf_isd_pred = sorted(emp_cdf_isd_pred)
            emp_ks_isd = max(abs(np.array(emp_cdf_isd) - np.array(emp_cdf_isd_pred)))
            
            write_to_file('SAD_mc_rsquare.txt', ",".join([dat_name, site, str(emp_rsquare_sad)]), new_line = False)
            write_to_file('SAD_mc_ks.txt', ",".join([dat_name, site, str(emp_ks_sad)]), new_line = False)
            write_to_file('ISD_mc_rsquare.txt', ",".join([dat_name, site, str(emp_rsquare_isd)]), new_line = False)
            write_to_file('ISD_mc_ks.txt', ",".join([dat_name, site, str(emp_ks_isd)]), new_line = False)
            
            for i in xrange(Niter):
                sad_mc_sample = sorted(np.random.random_integers(1, (2 * N0 - S0) / S0, S0), reverse = True)
                N_sample = sum(sad_mc_sample)
                isd_mc_sample = sorted(np.random.uniform(1, 2*E0/N_sample - 1, N_sample), reverse = True)
                E_sample = sum(isd_mc_sample)
                sad_pred_sample = get_mete_rad(int(S0), int(N_sample))[0]
                psi_sample = psi_epsilon(S0, N_sample, E_sample)
                beta_sample = get_beta(S0, N_sample)
                sample_rsquare_sad, sample_loglik_sad, sample_ks_sad = \
                    get_sample_stats_sad(sad_mc_sample, sad_pred_sample, exp(-beta_sample), N_sample)
                
                pool = multiprocessing.Pool(num_pools)
                sample_cdf_isd_pred = pool.map(get_pred_cdf_point, [(psi_sample, x) for x in isd_mc_sample])
                pool.close()
                pool.join()
                sample_cdf_isd_pred = sorted(sample_cdf_isd_pred)
                sample_ks_isd = max(abs(np.array(sorted(macroecotools.get_emp_cdf(isd_mc_sample))) - \
                                        np.array(sample_cdf_isd_pred)))
                # Use multiprocessing to generate sample ISD
                scaled_rank = [(x + 0.5) / N_sample for x in range(N_sample)]
                num_piece = int(N_sample / 100 / num_pools) # Do calculation for only 100 individuals at a time on each core to prevent crash
                pred_isd_sample = []
                for j in range(num_piece + 1):
                    pool = multiprocessing.Pool(num_pools)
                    scaled_rank_piece = scaled_rank[(j * 100 * num_pools):min((j + 1) * 100 * num_pools, N_sample)]
                    pred_isd_piece = pool.map(get_pred_isd_point, [(psi_sample, x) for x in scaled_rank_piece])
                    pred_isd_sample.extend(pred_isd_piece)
                    pool.close()
                    pool.join()
                pred_isd_sample = sorted(pred_isd_sample, reverse = True)
                sample_rsquare_isd = macroecotools.obs_pred_rsquare(np.log10(isd_mc_sample), np.log10(pred_isd_sample))
                
                write_to_file('SAD_mc_rsquare.txt', "".join([',', str(sample_rsquare_sad)]), new_line = False)
                write_to_file('SAD_mc_ks.txt', "".join([',', str(sample_ks_sad)]), new_line = False)
                write_to_file('ISD_mc_rsquare.txt', "".join([',', str(sample_rsquare_isd)]), new_line = False)
                write_to_file('ISD_mc_ks.txt', "".join([',', str(sample_ks_isd)]), new_line = False)
            
            write_to_file('SAD_mc_rsquare.txt', '\t')
            write_to_file('SAD_mc_ks.txt', '\t')
            write_to_file('ISD_mc_rsquare.txt', '\t')
            write_to_file('ISD_mc_ks.txt', '\t')

def plot_montecarlo(dat_mc, ax = None):
    """Plot the comparison between original statistic and Monte Carlo simulations."""
    if not ax:
        fig = plt.figure(figsize = (3.5, 3.5))
        ax = plt.subplot(111)
    
    dat_sort = dat_mc[dat_mc.argsort(order = 'orig')]
    stat_orig = dat_sort['orig']
    stat_low = [min(list(dat_sort[i])[3:]) for i in range(len(dat_sort))]
    stat_high = [max(list(dat_sort[i])[3:]) for i in range(len(dat_sort))]
    plt.scatter(range(1, len(dat_sort) + 1), stat_orig, c = 'black', s = 8)
    plt.fill_between(range(1, len(dat_sort) + 1), stat_low, stat_high, color = 'grey')
    ax.tick_params(axis = 'both', which = 'major', labelsize = 6)  
    return ax

def create_Fig_D1(dat_sad, dat_isd):
    fig = plt.figure(figsize = (7, 3.8))
    ax_sad = plt.subplot(121)
    plot_montecarlo(dat_sad, ax = ax_sad)
    plt.xlabel('Site', fontsize = 8)
    plt.ylabel(r'$R^2$', fontsize = 8)
    plt.title('SAD', fontsize = 14)
    ax_isd = plt.subplot(122)
    plot_montecarlo(dat_isd, ax = ax_isd)
    plt.xlabel('Site', fontsize = 8)
    plt.ylabel(r'$R^2$', fontsize = 8)
    plt.title('ISD', fontsize = 14)
    plt.savefig('Figure D1.pdf', dpi = 600)

def create_Fig_E1():
    fig = plt.figure(figsize = (7, 7))
    
def bootstrap_alternative(dat_name, cutoff = 9, Niter = 500):
    """Alternative method to generate bootstrap samples from the 
    
    distributions predicted by METE and compare with the fit of METE
    to emprical data.
    Here the patterns are generated together, instead of each independently from the 
    predicted distribution. First, a sample of length S0 is drawn from the SAD predicted
    by METE. Then for each of the S0 species, a sample of length n (species abundance)
    is drawn from the iISD with corresponding n predicted by METE. METE is then applied 
    to the bootstrap data, and goodness-of-fit measures (R^2 and K-S statistic) recorded.
    
    dat_name - name of study
    cutoff - minimum number of species required to run - 1
    Niter - number of bootstrap samples
    """
    dat = import_raw_data('./data/' + dat_name + '.csv')
    site_list = np.unique(dat['site'])
    
    num_pools = 8  # Assuming that 8 pools are to be created
    for site in site_list:
        dat_site = dat[dat['site'] == site]
        S0 = len(set(dat_site['sp']))
        if S0 > cutoff:
            N0 = len(dat_site)
            dbh_scale = np.array(dat_site['dbh'] / min(dat_site['dbh']))
            dbh2_scale = dbh_scale ** 2
            E0 = sum(dbh2_scale)
            beta = get_beta(S0, N0)
            theta = theta_epsilon(S0, N0, E0)
            emp_SAD = sorted([len(dat_site[dat_site['sp'] == sp]) for sp in sorted(list(set(dat_site['sp'])))], reverse = True)
            
            write_to_file('SAD_bootstrap_alt_rsquare.txt', ",".join([dat_name, site]), new_line = False)
            write_to_file('SAD_bootstrap_alt_ks.txt', ",".join([dat_name, site]), new_line = False)
            write_to_file('ISD_bootstrap_alt_rsquare.txt', ",".join([dat_name, site]), new_line = False)
            write_to_file('ISD_bootstrap_alt_ks.txt', ",".join([dat_name, site]), new_line = False)
            write_to_file('SDR_bootstrap_alt_rsquare.txt', ",".join([dat_name, site]), new_line = False)
            write_to_file('iISD_bootstrap_alt_rsquare.txt', ",".join([dat_name, site]), new_line = False)

            for i in xrange(Niter+1):
                sp_list = []
                dbh2_list = []  
                if i == 0: # Record results for empirical data for the first round                  
                    sad_sample = emp_SAD
                    N_sample = N0
                    for sp in sorted(list(set(dat_site['sp']))):
                        dbh2_sp = dbh2_scale[dat_site['sp'] == sp]
                        sp_list.extend([sp] * len(dbh2_sp))
                        dbh2_list.extend(sorted(dbh2_sp, reverse = True))
                    write_to_file('iISD_bootstrap_alt_ks_' + dat_name + '_' + site + '.txt', \
                                  ",".join(str(x) for x in emp_SAD))
                    E_sample  = E0
                    beta_sample = beta
                    theta_sample = theta
                    
                else: # For Niter iterations
                    # Generate bootstrap SAD from predicted logseries
                    sad_sample = sorted(mdis.trunc_logser.rvs(exp(-beta), N0, size = S0), reverse = True)
                    N_sample = sum(sad_sample)
                    # For each species generated iISD from predicted exponential
                    for sp in xrange(S0):
                        n_sp = sad_sample[sp]
                        sp_list.extend([str(sp)] * n_sp)
                        dbh2_list.extend(sorted(theta.rvs(n_sp, n_sp)))
                    E_sample = sum(dbh2_list)
                    # Re-apply METE to bootstrap data
                    beta_sample = get_beta(S0, N_sample)
                    theta_sample = theta_epsilon(S0, N_sample, E_sample)
                psi_sample = psi_epsilon(S0, N_sample, E_sample)
                sp_and_dbh2 = np.zeros((N_sample, ), dtype = [('sp', 'S25'), ('dbh2', 'f8')])
                sp_and_dbh2['sp'] = np.array(sp_list)
                sp_and_dbh2['dbh2'] = np.array(dbh2_list)
                # 1. SAD
                pred_sad_sample = get_mete_rad(int(S0), int(N_sample))[0]
                sample_rsquare_sad = macroecotools.obs_pred_rsquare(np.log10(sad_sample), np.log10(pred_sad_sample))
                write_to_file('SAD_bootstrap_alt_rsquare.txt', "".join([',', str(sample_rsquare_sad)]), new_line = False)
                sample_sad_cdf = macroecotools.get_emp_cdf(sad_sample)
                sample_ks_sad = max(abs(sample_sad_cdf - \
                                        np.array([mdis.trunc_logser.cdf(x, exp(-beta_sample), N_sample) for x in sad_sample])))
                write_to_file('SAD_bootstrap_alt_ks.txt', "".join([',', str(sample_ks_sad)]), new_line = False)
                # 2. ISD
                isd_sample = sorted(sp_and_dbh2['dbh2'], reverse = True)
                scaled_rank = [(x + 0.5) / N_sample for x in xrange(N_sample)]
                num_piece = int(N_sample / 100 / num_pools) # Do calculation for only 100 individuals at a time on each core to prevent crash
                pred_isd_sample = []                
                for j in range(num_piece + 1):
                    pool = multiprocessing.Pool(num_pools)
                    scaled_rank_piece = scaled_rank[(j * 100 * num_pools):min((j + 1) * 100 * num_pools, N_sample)]
                    pred_isd_piece = pool.map(get_pred_isd_point, [(psi_sample, x) for x in scaled_rank_piece])
                    pred_isd_sample.extend(pred_isd_piece)
                    pool.close()
                    pool.join()
                pred_isd_sample = sorted(pred_isd_sample, reverse = True)
                sample_rsquare_isd = macroecotools.obs_pred_rsquare(np.log10(isd_sample), np.log10(pred_isd_sample))
                write_to_file('ISD_bootstrap_alt_rsquare.txt', "".join([',', str(sample_rsquare_isd)]), new_line = False)
                
                sample_isd_cdf = macroecotools.get_emp_cdf(isd_sample)                
                pool = multiprocessing.Pool(num_pools)
                sample_isd_cdf_pred = pool.map(get_pred_cdf_point, [(psi_sample, x) for x in isd_sample])
                pool.close()
                pool.join()
                sample_isd_cdf_pred = sorted(sample_isd_cdf_pred, reverse = True)
                sample_ks_isd = max(abs(sample_isd_cdf - np.array(sample_isd_cdf_pred)))
                write_to_file('ISD_bootstrap_alt_ks.txt', "".join([',', str(sample_ks_isd)]), new_line = False)
                # 3. SDR and iISD
                sdr_sample = []
                pred_sdr_sample = []
                n_list = []
                sample_iisd_ks_list = []
                pred_iisd_sample = []
                iisd_sample = []
                for sp in np.unique(sp_and_dbh2['sp']):
                    sample_dbh2_sp = sp_and_dbh2[sp_and_dbh2['sp'] == sp]['dbh2']
                    sdr_sample.append(sum(sample_dbh2_sp) / len(sample_dbh2_sp))
                    pred_sdr_sample.append(theta_sample.E(len(sample_dbh2_sp)))
                    iisd_sample.extend(sorted(sample_dbh2_sp))
                    pred_iisd_sample_sp = get_iisd_pred_dbh2(sample_dbh2_sp, E_sample, len(sample_dbh2_sp) * psi_sample.lambda2)
                    pred_iisd_sample.extend(pred_iisd_sample_sp)
                    n_list.append(len(sample_dbh2_sp))
                    sample_iisd_cdf_sp = macroecotools.get_emp_cdf(sample_dbh2_sp)
                    sample_iisd_ks_sp = max(abs(sample_iisd_cdf_sp - \
                                                np.array([theta_sample.cdf(x, len(sample_dbh2_sp)) for x in sample_dbh2_sp])))
                    sample_iisd_ks_list.append(sample_iisd_ks_sp)
                sample_sdr_rsquare = macroecotools.obs_pred_rsquare(np.log10(sdr_sample), np.log10(pred_sdr_sample))
                write_to_file('SDR_bootstrap_alt_rsquare.txt', "".join([',', str(sample_sdr_rsquare)]), new_line = False)
                sample_iisd_ks = [ks for (n, ks) in sorted(zip(n_list, sample_iisd_ks_list), reverse = True)]
                write_to_file('iISD_bootstrap_alt_ks_' + dat_name + '_' + site + '.txt', \
                              ",".join(str(x) for x in sample_iisd_ks))                
                sample_iisd_rsquare = macroecotools.obs_pred_rsquare(np.log10(iisd_sample), np.log10(pred_iisd_sample))
                write_to_file('iISD_bootstrap_alt_rsquare.txt', "".join([',', str(sample_iisd_rsquare)]), new_line = False)
                
            write_to_file('SAD_bootstrap_alt_rsquare.txt', '\t')
            write_to_file('SAD_bootstrap_alt_ks.txt', '\t')
            write_to_file('ISD_bootstrap_alt_rsquare.txt', '\t')
            write_to_file('ISD_bootstrap_alt_ks.txt', '\t')
            write_to_file('SDR_bootstrap_alt_rsquare.txt', '\t')
            write_to_file('iISD_bootstrap_alt_rsquare.txt', '\t')
