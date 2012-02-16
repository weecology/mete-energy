from __future__ import division
import numpy as np
from scipy.stats import rv_continuous
from scipy.optimize import bisect
from math import exp
from mete import *
import sys
import mpmath
from mete_distributions import *

class theta_epsilon_gen(rv_continuous):
    """Intraspecific energy/mass distribution predicted by METE (Eqn 7.24)
    
    lower truncated at 1 and upper truncated at E0 or M0.
    
    Usage:
    PDF: theta_epsilon_m.pdf(list_of_epsilons, n, S0, N0, E0)
    CDF: theta_epsilon_m.cdf(list_of_epsilons, n, S0, N0, E0)
    Random Numbers: theta_epsilon_m.rvs(n, S0, N0, E0, size = 1)
    
    """  
    def __init__(self, n, S0, N0, E0):
        rv_continuous.__init__(self, momtype=1, a=1, b=E0, xa=-10.0, xb=10.0, xtol=1e-14, badvalue=None, 
                               name='theta_epsilon', longname='METE within species individual energy distribution', shapes="n,S0,N0,E0", extradoc=None)
        self.n = n
        self.beta = get_beta(S0, N0)
        self.lambda2 = get_lambda2(S0, N0, E0)
        self.lambda1 = self.beta - self.lambda2
        self.sigma = self.beta + (E0 - 1) * self.lambda2
 
    def _pdf(self, x):
        pdf = self.lambda2 * self.n * exp(-(self.lambda1 + self.lambda2 * x) * self.n) / (exp(-self.beta * self.n) - 
                                                                    exp(-self.sigma * self.n))
        return pdf

    def _cdf(self, x):
        cdf = mpmath.quad(self.pdf, [1, x])[0]
        return cdf 
    
    def _ppf(self, q):
        y = lambda t: self.cdf(t) - q
        x = bisect(y, self.a, self.b, xtol = 1.490116e-08)
        return x
    
    def E(self, n):
        """Expected value of the distribution"""
        def mom_1(x):
            return x * self.pdf(x)
        return mpmath.quad(mom_1, [self.a, self.b])[0]

#theta_epsilon = theta_epsilon_gen(name='theta_epsilon_m',
                              #longname='METE intraspecific energy distribution'
                              #)

# The following two distributions assume that there is an allometric relationship 
# between m and e, i.e., e = c * m ** a, with no error. 
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
