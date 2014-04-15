"""Rerun all analysis to make sure that results are up-to-date"""

import os
from working_functions import *
from pyper import *

dat_list = ['FERP', 'ACA', 'WesternGhats', 'BCI', 'BVSF', 'Lahei', 
             'LaSelva', 'NC', 'Oosting', 'Serimbu']

if not os.path.exists('./out_files/'):
    os.makedirs('./out_files/')
    
for dat_name in dat_list:
    dat_i = import_raw_data('./data/' + dat_name + '.csv')
    get_obs_pred_rad(dat_i, dat_name)
    get_obs_pred_dbh2(dat_i, dat_name)
    get_obs_pred_intradist(dat_i, dat_name)
    get_obs_pred_iisd(dat_i, dat_name)
    get_obs_pred_frequency(dat_i, dat_name)
    
#Reproducing results in main text
#Figure 1
plot_fig1()

#Figure 2
plot_four_patterns_ver2(dat_list, radius_isd = 0.5, radius_iisd = 0.5, inset = True, out_name = 'Figure 2')

#Reproducing results in appendices
#Figure B1
density_files = ['global', 'tropical', 'bci'] #Note that we don't have permission to redistribute Luquillo data
density_dic = {}
for dens_file in density_files:
    dens_data = import_density_file('./data/' + dens_file + '_density_cleaned.csv')
    for record in dens_data:
        if record[0] not in density_dic:
            density_dic[record[0]] = record[1]

sites_for_alt_analysis = get_site_for_alt_analysis(dat_list, density_dic)
get_mr_alt('./data/tropical_forest_types.csv', sites_for_alt_analysis, density_dic)

dat_alt_list = [dat_site_combo[0] + '_alt' for dat_site_combo in sites_for_alt_analysis]
dat_alt_list = sorted(list(set(dat_alt_list)))
for dat_name in dat_alt_list:
    dat_i = import_raw_data('./data/' + dat_name + '.csv')
    get_obs_pred_rad(dat_i, dat_name)
    get_obs_pred_dbh2(dat_i, dat_name)
    get_obs_pred_intradist(dat_i, dat_name)
    get_obs_pred_iisd(dat_i, dat_name)

plot_four_patterns_single_ver3(dat_alt_list, 'Figure B1.pdf', radius_par = 0.5)

#Figures C1-C3 
plot_obs_pred_freq(dat_list, out_name = 'Figure C1.png', inset = True)
ks_test(dat_list)
ks_results = np.genfromtxt('./out_files/ks_test_1000_0.05.csv', dtype = "S15,S15,S15, f8",
                                  names = ['dat_name','site', 'sp','sig'],
                                  delimiter = ",")
plot_dens_ks_test(ks_results['sig'], 0.05, out_fig = 'Figure C2.png')
plot_obs_pred_iisd_par(dat_list, out_name = 'Figure C3.png', inset = True)

#Figure E1
plot_four_patterns_single_ver3(dat_list, 'Figure E1.pdf')

#Table F1
# First need to create file "par_est.csv" with R
r = R()
r("source('dist_par_est.r')")
r("dat_list = c('FERP', 'ACA', 'WesternGhats', 'BCI', 'BVSF', 'Cocoli', 'Lahei', 'LaSelva', 'Luquillo', 'NC', 'Oosting', 'Serimbu', 'Shirakami', 'Sherman')")
r("get_par(dat_list, 'par_est.csv')")

AICc_ISD_to_file(dat_list, 'par_est.csv', outfile = 'Table F1.csv') 