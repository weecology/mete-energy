from __future__ import division
import numpy as np
from scipy.optimize import bisect
from scipy.stats.distributions import uniform
from math import exp
from mete import *
import sys
import mpmath

class theta_epsilon:
    """Intraspecific energy/mass distribution predicted by METE (Eqn 7.24)
    
    lower truncated at 1 and upper truncated at E0.
    
    Methods:
    pdf - probability density function
    cdf - cumultaive density function
    ppf - inverse cdf
    rvs - random number generator
    E - first moment (mean)
    
    """
    def __init__(self, S0, N0, E0):
        self.a, self.b = 1, E0
        self.beta = get_beta(S0, N0)
        self.lambda2 = get_lambda2(S0, N0, E0)
        self.lambda1 = self.beta - self.lambda2
        self.sigma = self.beta + (E0 - 1) * self.lambda2
 
    def pdf(self, x, n):
        pdf = self.lambda2 * n * exp(-(self.lambda1 + 
                                       self.lambda2 * x) * n) / (exp(-self.beta * n) - 
                                                                 exp(-self.sigma * n))
        return pdf

    def cdf(self, x, n):
        def pdf_n(x):
            return self.pdf(x, n)
        cdf = mpmath.quad(pdf_n, [1, x])
        return float(cdf) 
    
    def ppf(self, n, q):
        y = lambda t: self.cdf(t, n) - q
        x = bisect(y, self.a, self.b, xtol = 1.490116e-08)
        return x
    
    def rvs(self, n, size):
        out = []
        rand_list = uniform.rvs(size = size)
        for rand_num in rand_list:
            out.append(self.ppf(n, rand_num))
        return out
        
    def E(self, n):
        """Expected value of the distribution"""
        def mom_1(x):
            return x * self.pdf(x, n)
        return float(mpmath.quad(mom_1, [self.a, self.b]))


class theta_m_no_error_gen(rv_continuous):
    """Intraspecific mass distribution when constraint is E0.
    
    Lower truncated at (1/c) ** (1/a) and upper truncated at (E0/c) ** (1/a).
    
    """        
    def _pdf(self, x, n, S0, N0, E0, c, a):
        beta = get_beta(S0, N0)
        lambda2 = get_lambda2(S0, N0, E0)
        lambda1 = beta - lambda2
        sigma = beta + (E0 - 1) * lambda2
        x = np.array(x)
        pdf = lambda2 * c * a * n * x ** (a - 1) * np.exp(-(lambda1 + 
                                                            lambda2 * c * x ** a) * n) / (np.exp(-beta * n) - 
                                                                                          np.exp(-sigma * n))
        return pdf

    def _ppf(self, q, n, S0, N0, E0, c, a):
        x = []
        for q_i in q: 
            y_i = lambda t: self._cdf(t, n, S0, N0, E0, c, a) - q_i
            x.append(bisect(y_i, self.a, self.b, xtol = 1.490116e-08))
        return np.array(x)
    
    def E(self, n, S0, N0, E0, c, a):
        """Expected value of the distribution"""
        def mom_1(x):
            return x * self.pdf(x, n, S0, N0, E0, c, a)
        return quad(mom_1, self.a, self.b)[0]
    
    def _argcheck(self, *args):
        self.a = (1 / args[4]) ** (1 / args[5])
        self.b = (args[3] / args[4]) ** (1 / args[5])
        cond = (args[0] > 0) & (args[1] > 0) & (args[2] > 0) & (args[3] > 0)
        return cond

theta_m_no_error = theta_m_no_error_gen(name='theta_m_no_error', shapes="n, S0, N0, E0, c, a",
                              longname='Intraspecific body mass distribution, no error'
                              )

class theta_epsilon_no_error_gen(rv_continuous):
    """Intraspecific energy distribution when constraint is M0.
    
    Lower truncated at c and upper truncated at c * M0 ** a.

    """        
    def _pdf(self, x, n, S0, N0, E0, c, a):
        beta = get_beta(S0, N0)
        lambda2 = get_lambda2(S0, N0, E0)
        lambda1 = beta - lambda2
        sigma = beta + (E0 - 1) * lambda2
        x = np.array(x)
        pdf = lambda2 * (1 / c) ** (1 / a) / a * n * x ** (1 / a - 1) * np.exp(-(lambda1 + 
                                                            lambda2 * (1 / c) ** (1 / a) * x ** (1 / a)) * n) / (np.exp(-beta * n) - 
                                                                                          np.exp(-sigma * n))
        return pdf

    def _ppf(self, q, n, S0, N0, E0, c, a):
        x = []
        for q_i in q: 
            y_i = lambda t: self._cdf(t, n, S0, N0, E0, c, a) - q_i
            x.append(bisect(y_i, self.a, self.b, xtol = 1.490116e-08))
        return np.array(x)
    
    def E(self, n, S0, N0, E0, c, a):
        """Expected value of the distribution"""
        def mom_1(x):
            return x * self.pdf(x, n, S0, N0, E0, c, a)
        return quad(mom_1, self.a, self.b)[0]
    
    def _argcheck(self, *args):
        self.a = args[4]
        self.b = args[4] * args[3] ** args[5]
        cond = (args[0] > 0) & (args[1] > 0) & (args[2] > 0) & (args[3] > 0)
        return cond

theta_epsilon_no_error = theta_epsilon_no_error_gen(name='theta_epsilon_no_error', shapes="n, S0, N0, E0, c, a",
                              longname='Intraspecific energy distribution, no error'
                              )
