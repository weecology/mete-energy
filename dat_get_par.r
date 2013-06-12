source('dist_par_est.r')

dat_list = c('FERP', 'ACA', 'WesternGhats', 'BCI', 'BVSF', 'Cocoli', 'Lahei', 
             'LaSelva', 'Luquillo', 'NC', 'Oosting', 'Serimbu', 'Shirakami', 'Sherman')
             
get_par(dat_list, 'par_est.csv')


