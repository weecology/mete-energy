source('dist_par_est.r')

BCI = read_file('bci7.csv')
Luquillo = read_file('lfdp2.csv')
WesternGhats = read_file('WesternGhats.csv')
FERP = read_file('FERP.csv')
Oosting = read_file('Palmer2007.csv')
LaSelva = read_file('Baribault2011.csv')

dat_list = list(BCI = BCI, Luquillo = Luquillo, WesternGhats = WesternGhats,
  FERP = FERP, Oosting = Oosting, LaSelva = LaSelva)
out_names = c('BCI', 'Luquillo', 'WesternGhats', 'FERP', 'Oosting', 'LaSelva')

get_par(dat_list, 'par_est_6sites.csv')


