source('dist_par_est.r')

dat = read.csv(pipe('cat /dev/stdin'))
dat_dbh = dat[,2]
dat_dbh = dat_dbh / min(dat_dbh)
trunc_expon_par = trunc_expon_par_est(dat_dbh, 1)
trunc_pareto_par = trunc_pareto_par_est(dat_dbh, 1)
trunc_weibull_par = trunc_weibull_par_est(dat_dbh, 1)

print(c(trunc_expon_par, trunc_pareto_par, trunc_weibull_par))