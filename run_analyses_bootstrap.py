"""Run bootstrap analyses"""

from working_functions import *
import multiprocessing

# Datasets not included: BCI, Cocoli, Sherman, Shiramaki, Lahei
dat_list = ['FERP', 'ACA', 'WesternGhats', 'BVSF', 
             'LaSelva', 'Luquillo', 'NC', 'Oosting', 'Serimbu'] 

# The following analyses assume that analyses in run_analyses.py
# has already been conducted for corresponding datasets.

# 1. Appendix D: Simulate null communities to validate R^2
for dat_name in dat_list:
        montecarlo_uniform_SAD_ISD(dat_name, Niter = 100)
        
# Figure D1
sad_mc = import_bootstrap_file('SAD_mc_rsquare.txt')
isd_mc = import_bootstrap_file('ISD_mc_rsquare.txt')

create_Fig_D1(sad_mc, isd_mc)

# 2. Bootstrap analysis
# Bootstrap SAD
pool = multiprocessing.Pool(8)  # Assuming that there are 8 cores
pool.map(bootstrap_SAD, dat_list)
pool.close()
pool.join

# Bootstrap ISD - Note: Can be extremely time-consuming
for dat_name in dat_list:
        bootstrap_ISD(dat_name)

# Bootstrap SDR and iISD
for dat_name in dat_list:
        bootstrap_SDR_iISD(dat_name)
        
# Figures E1 & E2
create_Fig_E1()
create_Fig_E2()
