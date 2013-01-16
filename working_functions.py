"""Module with all working functions for the METE energy project"""

from __future__ import division
import numpy as np
import csv
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from mete import *
import macroecotools
from fit_other_dist import *
from EM_dist import *
from math import exp

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

def run_test(raw_data, dataset_name, data_dir='./data/', cutoff = 9):
    """Use data to compare the predicted and empirical SADs and get results in csv files
    
    (Copied and modified from the funciton of the same name from White et al. 2012)
    Keyword arguments:
    raw_data : numpy structured array with 3 columns: 'site','sp','dbh'
    dataset_name : short code that will indicate the name of the dataset in
                    the output file names
    data_dir : directory in which to store output
    cutoff : minimum number of species required to run - 1.
    
    """
    
    usites = np.sort(list(set(raw_data["site"])))
    f1_write = open(data_dir + dataset_name + '_obs_pred.csv', 'wb')
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

#def plot_obs_pred_sad(datasets, data_dir='./data/', radius=2):
    #"""Multiple obs-predicted plotter
    
    #(Copied and modified from White et al. 2012)
    
    #"""
    ## Only for illustration purpose for the poster. 
    #fig = plt.figure(figsize = (7, 7))
    #num_datasets = len(datasets)
    #for i, dataset in enumerate(datasets):
        #obs_pred_data = import_obs_pred_data(data_dir + dataset + '_obs_pred.csv') 
        #site = ((obs_pred_data["site"]))
        #obs = ((obs_pred_data["obs"]))
        #pred = ((obs_pred_data["pred"]))
        
        #axis_min = 0.5 * min(obs)
        #axis_max = 2 * max(obs)
        #ax = fig.add_subplot(2,2,i+1)
        #macroecotools.plot_color_by_pt_dens(pred, obs, radius, loglog=1, 
                                            #plot_obj=plt.subplot(2,2,i+1))      
        #plt.plot([axis_min, axis_max],[axis_min, axis_max], 'k-')
        #plt.xlim(axis_min, axis_max)
        #plt.ylim(axis_min, axis_max)
        #plt.subplots_adjust(wspace=0.29, hspace=0.29)  
        #plt.title(dataset)
        #plt.text(1, axis_max / 3, r'$r^2$ = %0.2f' %macroecotools.obs_pred_rsquare(obs, pred))
        ### Create inset for histogram of site level r^2 values
        ##axins = inset_axes(ax, width="30%", height="30%", loc=4)
        ##hist_mete_r2(site, np.log10(obs), np.log10(pred))
        ##plt.setp(axins, xticks=[], yticks=[])
    
    #fig.text(0.5, 0.04, 'Predicted Abundance', ha = 'center', va = 'center')
    #fig.text(0.04, 0.5, 'Observed Abundance', ha = 'center', va = 'center', 
             #rotation = 'vertical')
    #plt.savefig('obs_pred_plots.png', dpi=400)

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
            results = ((np.column_stack((np.array([site] * len(cdf_obs)), cdf_obs, cdf_pred))))
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
            results = ((np.column_stack((np.array([site] * len(dbh2_obs)), dbh2_obs, dbh2_pred))))
            f1.writerows(results)
    f1_write.close()

def get_obs_pred_frequency(raw_data, dataset_name, data_dir = './data/', cutoff = 9):
    """Use data to compare the predicted and empirical frequency for each size bins
    
    and store results in csv files.
    Keyword arguments:
    raw_data : numpy structured array with 3 columns: 'site','sp','dbh'
    dataset_name : short code that will indicate the name of the dataset in
                    the output file names
    data_dir : directory in which to store output
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
            num_bin = int(ceil(log(max(dbh_scale)) / log(2)))
            freq_obs = []
            freq_pred = []
            for i in range(num_bin):
                count = len(dbh_scale[(dbh_scale < 2 ** (i + 1)) & (dbh_scale >= 2 ** i)])
                freq_obs.append(count / N0 / 2 ** i)
                freq_pred.append((psi.cdf((2 ** (i + 1)) ** 2) - 
                                  psi.cdf((2 ** i) ** 2))/ 2 ** i)
            #save results to a csv file:
            #results = ((np.column_stack((np.array([site] * len(freq_obs)), freq_obs, freq_pred))))
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
    f2_write = open(data_dir + dataset_name + '_scaled_par.csv', 'wb')
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
            abd = []
            scaled_par_list = []
            psi = psi_epsilon(S0, N0, E0)
            theta_epsilon_obj = theta_epsilon(S0, N0, E0)
            for sp in S_list:
                sp_dbh2 = dbh2_scale[subdat[subdat.dtype.names[1]] == sp]
                if len(sp_dbh2) > n_cutoff: 
                    abd.append(len(sp_dbh2))
                    mr_avg_obs.append(sum(sp_dbh2) / len(sp_dbh2))
                    scaled_par = 1 / (sum(sp_dbh2) / len(sp_dbh2) - 1) / psi.lambda2
                    scaled_par_list.append(scaled_par)
                    mr_avg_pred.append(theta_epsilon_obj.E(len(sp_dbh2)))
            #save results to a csv file:
            results1 = np.zeros((len(abd), ), dtype = ('S10, f8, f8'))
            results1['f0'] = np.array([site] * len(abd))
            results1['f1'] = np.array(mr_avg_obs)
            results1['f2'] = np.array(mr_avg_pred)
            f1.writerows(results1)
            
            results2 = np.zeros((len(abd), ), dtype = ('S10, f8, i8'))
            results2['f0'] = np.array([site] * len(abd))
            results2['f1'] = np.array(scaled_par_list)
            results2['f2'] = np.array(abd)
            f2.writerows(results2)
    f1_write.close()
    f2_write.close()

def species_rand_test(datasets, data_dir = './data/', cutoff = 9, n_cutoff = 4, Niter = 200, lumped = True):
    """Randomize species identity within sites and compare the r-square obtained 
    
    for abundance-size distribution and intraspecific energy distribution parameter
    of empirical data versus randomized data.
    Keyword arguments:
    datasets - a list of dataset names
    data_dir - directory for output file
    Niter - number of randomizations applying to each site in each dataset
    lumped - whether the output r-square will be lumped (one line for all datasets), 
             or separate into sites (one line for each site in each dataset).
             
    """
    if lumped:
        f1_write = open(data_dir + 'mr_rand_lumped.csv', 'wb')
        f2_write = open(data_dir + 'lambda_rand_lumped.csv', 'wb')
    else:
        f1_write = open(data_dir + 'mr_rand_sites.csv', 'wb')
        f2_write = open(data_dir + 'lambda_rand_sites.csv', 'wb')
    f1 = csv.writer(f1_write)
    f2 = csv.writer(f2_write)
    
    if not lumped:
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
                    # Compute the two r2 for original data 
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
                    mr_out_row.append(macroecotools.obs_pred_rsquare(np.array(mr_avg_obs), 
                                                                     np.array(mr_avg_pred)))
                    lambda_out_row.append(macroecotools.obs_pred_rsquare(np.array(abd), 
                                                                         np.array(scaled_par_list)))
                    # Iteration of randomization
                    for i in np.arange(Niter):
                        np.random.shuffle(subdat[subdat.dtype.names[1]])
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
                        mr_out_row.append(macroecotools.obs_pred_rsquare(np.array(mr_avg_obs), 
                                                                                 np.array(mr_avg_pred)))
                        lambda_out_row.append(macroecotools.obs_pred_rsquare(np.array(abd), 
                                                                             np.array(scaled_par_list)))

                    f1.writerow(mr_out_row)
                    f2.writerow(lambda_out_row)
    f1_write.close()
    f2_write.close()

def plot_obs_pred(datasets, data_dir, radius, loglog, filename):
    """Generic function to generate an observed vs predicted figure with 1:1 line"""
    fig = plt.figure(figsize = (7, 7))
    num_datasets = len(datasets)
    obs = []
    pred = []
    for i, dataset in enumerate(datasets):
        obs_pred_data = import_obs_pred_data(data_dir + dataset + filename) 
        obs.extend(list(obs_pred_data['obs']))
        pred.extend(list(obs_pred_data['pred']))

    axis_min = 0.9 * min(obs+pred)
    axis_max = 1.1 * max(obs+pred)
    macroecotools.plot_color_by_pt_dens(np.array(pred), np.array(obs), radius, loglog=loglog)      
    plt.plot([axis_min, axis_max],[axis_min, axis_max], 'k-')
    plt.xlim(axis_min, axis_max)
    plt.ylim(axis_min, axis_max)
    plt.annotate(r'$r^2$ = %0.2f' %macroecotools.obs_pred_rsquare(np.array(obs), np.array(pred)),
                 xy = (0.75, 0.05), xycoords = 'axes fraction')
    #plt.text(0.05 * axis_max, 1.3 * axis_min, r'$r^2$ = %0.2f' %macroecotools.obs_pred_rsquare(np.array(obs), np.array(pred)))
    return fig

def plot_obs_pred_sad(datasets, data_dir = "./data/", radius = 2):
    """Plot the observed vs predicted abundance for each species for multiple datasets."""
    fig = plot_obs_pred(datasets, data_dir, radius, 1, '_obs_pred.csv')
    fig.text(0.5, 0.04, 'Predicted Abundance', ha = 'center', va = 'center')
    fig.text(0.04, 0.5, 'Observed Abundance', ha = 'center', va = 'center', 
             rotation = 'vertical')
    plt.savefig('obs_pred_sad.png', dpi = 400)

def plot_obs_pred_dbh2(datasets, data_dir = "./data/", radius = 2):
    """Plot the observed vs predicted dbh2 for each individual for multiple datasets."""
    fig = plot_obs_pred(datasets, data_dir, radius, 1, '_obs_pred_isd_dbh2.csv')
    fig.text(0.5, 0.04, r'Predicted $DBH^2$', ha = 'center', va = 'center')
    fig.text(0.04, 0.5, r'Observed $DBH^2$', ha = 'center', va = 'center', 
             rotation = 'vertical')
    plt.savefig('obs_pred_dbh2.png', dpi = 400)

def plot_obs_pred_cdf(datasets, data_dir = "./data/", radius = 0.05):
    """Plot the observed vs predicted cdf for multiple datasets."""
    fig = plot_obs_pred(datasets, data_dir, radius, 0, '_obs_pred_isd_cdf.csv')
    fig.text(0.5, 0.04, 'Predicted F(x)', ha = 'center', va = 'center')
    fig.text(0.04, 0.5, 'Observed F(x)', ha = 'center', va = 'center', 
             rotation = 'vertical')
    plt.savefig('obs_pred_cdf.png', dpi = 400)

def plot_obs_pred_freq(datasets, data_dir = "./data/", radius = 0.05):
    """Plot the observed vs predicted size frequency for multiple datasets."""
    fig = plot_obs_pred(datasets, data_dir, radius, 1, '_obs_pred_freq.csv')
    fig.text(0.5, 0.04, 'Predicted frequency', ha = 'center', va = 'center')
    fig.text(0.04, 0.5, 'Observed frequency', ha = 'center', va = 'center', 
             rotation = 'vertical')
    plt.savefig('obs_pred_freq.png', dpi = 400)

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
    
def plot_fig1(data_dir = "./data/"):
    """Illustration of the 4 patterns using BCI data."""
    fig = plt.figure(figsize = (7, 7))
    # Subplot A: Observed and predicted RAD
    # Code adopted and modified from example_sad_plot in mete_sads
    obs_pred_data = import_obs_pred_data('./data/'+ 'BCI' + '_obs_pred.csv')    
    obs = obs_pred_data["obs"]   
    pred = obs_pred_data["pred"]
    rank_obs, relab_obs = macroecotools.get_rad_data(obs)
    rank_pred, relab_pred = macroecotools.get_rad_data(pred)
    plot_obj = plt.subplot(2, 2, 1)
    plot_obj.semilogy(rank_obs, relab_obs, 'o', markerfacecolor='none', markersize=6, 
             markeredgecolor='#999999', markeredgewidth=1)
    plot_obj.semilogy(rank_pred, relab_pred, '-', color='#9400D3', linewidth=2)
    plt.xlabel('Rank', fontsize = '14')
    plt.ylabel('Relative Abundance', fontsize = '14')
    # Subplot B: ISD shown as histogram with predicted pdf
    # Code adopted and modified from plot_ind_hist from fit_other_dist
    dat = import_raw_data('bci7.csv')
    dbh_raw = dat[dat.dtype.names[2]]
    dbh_scale = np.array(dbh_raw / min(dbh_raw))
    dbh2_scale = dbh_scale ** 2
    E0 = sum(dbh2_scale)
    N0 = len(dbh2_scale)
    S0 = len(set(dat[dat.dtype.names[1]]))
    
    num_bin = int(ceil(log(max(dbh_scale)) / log(2)))
    emp_pdf = []
    for i in range(num_bin):
        count = len(dbh_scale[(dbh_scale < 2 ** (i + 1)) & (dbh_scale >= 2 ** i)])
        emp_pdf.append(count / N0 / 2 ** i)
    psi = psi_epsilon(S0, N0, E0)
    psi_pdf = []
    x_array = np.arange(1, ceil(max(dbh_scale)) + 1)
    for x in x_array:
        psi_pdf.append(float(ysquareroot_pdf(x, psi)))
    plot_obj = plt.subplot(2, 2, 2)
    plot_obj.loglog(x_array, psi_pdf, '#9400D3', linewidth = 2, label = 'METE')
    plot_obj.bar(2 ** np.array(range(num_bin)), emp_pdf, color = '#d6d6d6', 
            width = 0.4 * 2 ** np.array(range(num_bin)))
    plt.xlabel('DBH')
    plt.ylabel('Probability Density')
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
    n_list = sorted(list(set([int(x) for x in np.exp(np.arange(log(min(n_obs)), log(max(n_obs)), 0.1))])))
    dbh2_pred = []
    for n in n_list:
        dbh2_pred.append(theta_epsilon_obj.E(n))
    plot_obj = plt.subplot(2, 2, 3)
    plot_obj.loglog(np.array(dbh2_pred), np.array(n_list), color = '#9400D3', linewidth = 2)
    plot_obj.scatter(dbh2_obs, n_obs, color = '#999999', marker = 'o')
    plt.xlabel(r'Species Level Average $DBH^2$')
    plt.ylabel('Species Abundance')
    # Subplot D: Intra-specific distribution for the most abundant species (Hybanthus prunifolius)
    hp_dbh2 = dbh2_scale[dat[dat.dtype.names[1]] == 'Hybanthus prunifolius']
    hp_num_bin = int(ceil(log(max(hp_dbh2)) / log(2)))
    hp_emp_pdf = []
    for i in range(hp_num_bin):
        count = len(hp_dbh2[(hp_dbh2 < 2 ** (i + 1)) & (hp_dbh2 >= 2 ** i)])
        hp_emp_pdf.append(count / len(hp_dbh2) / 2 ** i)
    def exp_dist(x, lam):
        return lam * np.exp(-lam * x)
    lam_pred = psi.lambda2 * len(hp_dbh2)
    lam_est = 1 / (sum(hp_dbh2) / len(hp_dbh2) - 1)
    x_array = np.arange(1, ceil(max(hp_dbh2)) + 1)
    plot_obj = plt.subplot(2, 2, 4)
    plot_obj.loglog(x_array, exp_dist(x_array, lam_pred), '#9400D3', linewidth = 2, label = 'METE')
    plot_obj.loglog(x_array, exp_dist(x_array, lam_est), '#FF4040', linewidth = 2, label = 'Truncated exponential')
    plot_obj.bar(2 ** np.array(range(hp_num_bin)), hp_emp_pdf, color = '#d6d6d6', 
        width = 0.4 * 2 ** np.array(range(hp_num_bin)))
    plt.xlabel(r'$DBH^2$')
    plt.ylabel('Probability Density')
    plt.subplots_adjust(wspace = 0.29, hspace = 0.29)
    plt.savefig(data_dir + 'fig1.png', dpi = 400)

def plot_obs_pred_avg_mr(datasets, data_dir = "./data/", radius = 2):
    """Plot the observed vs predicted species-level average metabolic rate 
    
    for all species across multiple datasets.
    
    """
    fig = plot_obs_pred(datasets, data_dir, radius, 1, '_obs_pred_avg_mr.csv')
    fig.text(0.5, 0.04, 'Predicted Species-Average Metabolic Rate', ha = 'center', va = 'center')
    fig.text(0.04, 0.5, 'Observed Species-Average Metabolic Rate', ha = 'center', va = 'center', 
             rotation = 'vertical')
    plt.savefig('obs_pred_average_mr.png', dpi = 400)

def plot_scaled_par(datasets, data_dir = "./data/", radius = 2):
    """Plot the scaled intra-specific energy distribution parameter against abundance."""
    fig = plot_obs_pred(datasets, data_dir, radius, 1, '_scaled_par.csv')
    fig.text(0.5, 0.04, 'Abundance', ha = 'center', va = 'center')
    fig.text(0.04, 0.5, r'Scaled $\lambda$ for Within Species Distribution', ha = 'center', 
             va = 'center', rotation = 'vertical')
    plt.savefig('intra_scaled_par.png', dpi = 400)

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
