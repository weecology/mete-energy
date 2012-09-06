"""Module with all working functions for the METE energy project"""

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
            results = ((np.column_stack((np.array([usites[i]] * len(obsab)), obsab, pred))))
            f1.writerows(results)
    f1_write.close()

def plot_obs_pred_sad(datasets, data_dir='./data/', radius=2):
    """Multiple obs-predicted plotter
    
    (Copied and modified from White et al. 2012)
    
    """
    # Only for illustration purpose for the poster. 
    fig = plt.figure(figsize = (7, 7))
    num_datasets = len(datasets)
    for i, dataset in enumerate(datasets):
        obs_pred_data = import_obs_pred_data(data_dir + dataset + '_obs_pred.csv') 
        site = ((obs_pred_data["site"]))
        obs = ((obs_pred_data["obs"]))
        pred = ((obs_pred_data["pred"]))
        
        axis_min = 0.5 * min(obs)
        axis_max = 2 * max(obs)
        ax = fig.add_subplot(2,2,i+1)
        macroecotools.plot_color_by_pt_dens(pred, obs, radius, loglog=1, 
                                            plot_obj=plt.subplot(2,2,i+1))      
        plt.plot([axis_min, axis_max],[axis_min, axis_max], 'k-')
        plt.xlim(axis_min, axis_max)
        plt.ylim(axis_min, axis_max)
        plt.subplots_adjust(wspace=0.29, hspace=0.29)  
        plt.title(dataset)
        plt.text(1, axis_max / 3, r'$r^2$ = %0.2f' %macroecotools.obs_pred_rsquare(obs, pred))
        ## Create inset for histogram of site level r^2 values
        #axins = inset_axes(ax, width="30%", height="30%", loc=4)
        #hist_mete_r2(site, np.log10(obs), np.log10(pred))
        #plt.setp(axins, xticks=[], yticks=[])
    
    fig.text(0.5, 0.04, 'Predicted Abundance', ha = 'center', va = 'center')
    fig.text(0.04, 0.5, 'Observed Abundance', ha = 'center', va = 'center', 
             rotation = 'vertical')
    plt.savefig('obs_pred_plots.png', dpi=400)

def get_mete_pred_cdf(dbh2_scale, S0, N0, E0):
    """Compute the cdf of the individual metabolic rate (size) distribution
    
    predicted by METE given S0, N0, E0, and scaled dbh**2.
    
    """
    pred_cdf = []
    psi = psi_epsilon(S0, N0, E0)
    for dbh2 in dbh2_scale:
        pred_cdf.append(psi.cdf(dbh2))
    return np.array(pred_cdf)

def get_obs_cdf(dat):
    """Compute the empirical cdf given a list or an array"""
    dat = np.array(dat)
    emp_cdf = []
    for point in dat:
        point_cdf = len(dat[dat <= point]) / len(dat)
        emp_cdf.append(point_cdf)
    return np.array(emp_cdf)

def get_obs_pred_cdf(raw_data, dataset_name, data_dir='./data/', cutoff = 9):
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
            print("%s, Site %s, S=%s, N=%s" % (dataset_name, i, S0, N0))
            # Generate predicted values and p (e ** -beta) based on METE:
            cdf_pred = get_mete_pred_cdf(dbh2_scale, S0, N0, E0)
            cdf_obs = get_obs_cdf(dbh2_scale)
            #save results to a csv file:
            results = ((np.column_stack((np.array([site] * len(cdf_obs)), cdf_obs, cdf_pred))))
            f1.writerows(results)
    f1_write.close()

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
