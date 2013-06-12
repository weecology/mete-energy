library(MASS)

trunc_weibull = function(x, k, lmd, lower_bound){
  return (dweibull(x, k, lmd) / (1 - pweibull(lower_bound, k, lmd)))
}

trunc_weibull_par_est = function(x, lower_bound){
  trunc_weibull_lb = function(x, k, lmd){
    return (trunc_weibull(x, k, lmd, lower_bound))
  }
  weibull_est = as.numeric((fitdistr(x, 'weibull'))$estimate)
  return (as.numeric(fitdistr(x, trunc_weibull_lb, list(k = weibull_est[1], lmd = weibull_est[2]))$estimate))
}

trunc_expon_par_est = function(x, lower_bound){
  return (1 / (mean(x) - lower_bound))
}

trunc_pareto_par_est = function(x, lower_bound){
  return (length(x) / sum(log(x) - log(lower_bound)))
}

read_file = function(path){
  dat = read.csv(path,
    colClasses = c('character', 'character', 'numeric'))
  return(dat)
}

get_par = function(dat_list, out_file){
  par_out = as.data.frame(matrix(NA, 0, 6))
  k = 1
  for (i in 1:length(dat_list)){
    dat_i = read_file(paste(dat_list[i], '.csv', sep = ''))
    site_list = unique(dat_i[, 1])
    for (j in 1:length(site_list)){
      dat_site = dat_i[dat_i[, 1] == site_list[j], ]
      dat_dbh = dat_site[, 3] / min(dat_site[, 3]) # Rescale so that min MR is 1
      out = as.data.frame(matrix(NA, 1, 6))
      out[1] = dat_list[i]
      out[2] = site_list[j]
      out[3] = trunc_expon_par_est(dat_dbh, 1)
      out[4] = trunc_pareto_par_est(dat_dbh, 1)
      out[5:6] = trunc_weibull_par_est(dat_dbh, 1)
      par_out = rbind(par_out, out)
      k = k + 1
    }
  }
  names(par_out) = c('dataset', 'site', 'expon_par', 'pareto_par', 'weibull_k', 'weibull_lmd')
  write.csv(par_out, out_file, row.names = F, quote = F)
}
