"""Conversion from DBH to mass to respiration rate for trees"""

from __future__ import division
from math import log
import numpy as np

def multi_stem_dbh(list_of_stems):
    """Convert DBH of multiple stems of an individual to a single trunk measure. 
    
    Formula is taken from Ernest et al. 2009. 
    
    """
    stems = np.array(list_of_stems)
    return sum(stems ** 2) ** 0.5

def dbh_to_mass(dbh, rho, forest_type):
    """Convert DBH to biomass based on formulas taken from Chave et al. 2005.
    
    Input: 
    dbh - diameter at breast height (cm)
    rho - wood specific gravity (g/cm**3)
    forest_type - forest type that determines fitting parameters. 
                  Can take one of the four values: "dry", "moist", "wet" and "mangrove". 
    Output: above ground biomass (kg)
    
    """
    if forest_type == "dry":
        a, b = -0.667, 1.784
    elif forest_type == "moist":
        a, b = -1.499, 2.148
    elif forest_type == "wet":
        a, b = -1.239, 1.98
    elif forest_type == "mangrove":
        a, b = -1.349, 1.98
    else: 
        print "Error: unidentified forest type."
        return None
    c, d = 0.207, -0.0281
    return rho * exp(a + b * log(dbh) + c * log(dbh) ** 2 + d * log(dbh) ** 3)

def mass_to_resp(agb):
    """Convert aboveground biomass (kg) to respiration rate (miu-mol/s) based on Mori et al. 2010 Eqn 2"""
    G, H, g, h = 6.69, 0.421, 1.082, 0.78
    return 1 / (1 / (G * agb ** g) + 1 / (H * agb ** h))
