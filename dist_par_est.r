library(MASS)

trunc_weibull = function(x, k, lmd, lower_bound){
  return (dweibull(x, k, lmd) / (1 - pweibull(lower_bound, k, lmd)))
}

trunc_weibull_par_est = function(x, lower_bound){
  trunc_weibull_lb = function(x, k, lmd){
    return (trunc_weibull(x, k, lmd, lower_bound))
  }
  return (as.numeric(fitdistr(x, trunc_weibull_lb, list(k = 1, lmd = mean(x) / gamma(1 + 1/1)))$estimate))
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
  list_elements = names(dat_list)
  par_out = as.data.frame(matrix(NA, 1, 6))
  names(par_out) = c('dataset', 'site', 'expon_par', 'pareto_par', 'weibull_k', 'weibull_lmd')
  k = 1
  for (i in 1:length(list_elements)){
    dat_name = list_elements[i]
    dat = dat_list[[dat_name]]
    site_list = unique(dat[, 1])
    for (j in 1:length(site_list)){
      par_out[k, 1] = dat_name
      par_out[k, 2] = site_list[j]
      dat_site = dat[dat[, 1] == site_list[j], ]
      dat_dbh = dat_site[, 3] / min(dat_site[, 3]) # Rescale so that min MR is 1
      par_out[k, 3] = trunc_expon_par_est(dat_dbh^2, 1)
      par_out[k, 4] = trunc_pareto_par_est(dat_dbh^2, 1)
      par_out[k, 5:6] = trunc_weibull_par_est(dat_dbh^2, 1)
      k = k + 1
    }
  }
  write.csv(par_out, out_file, row.names = F, quote = F)
}
