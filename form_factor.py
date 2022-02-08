import numpy as np
from numpy import linspace, sqrt, array, pi, cos, sin, dot, exp, sinh, log, log10, cosh, sinh
from scipy.interpolate import interp1d, interp2d
from scipy import integrate
import sys, os, glob
import pandas as pd

############################################################################################

# Function to load form factor

def load_form_factor(self,datadir,filename):

    form_factor_path = datadir + self.target+'/'+ filename

    if( not os.path.exists(form_factor_path)):
        print("Warning! Form factor not loaded. Need to set form_factor_filename if needed. Otherwise defaults to massive mediator ")
        self.form_factor_loaded=False
    else:
        self.form_factor_loaded=True
        formfactordat = np.loadtxt(form_factor_path).T
        print("Loaded " + filename + " for form factor")
        self.form_factor = formfactordat
        self.form_factor_interp = interp1d(self.form_factor[0],self.form_factor[1],kind='linear')
        self.form_factor_range = [ self.form_factor[0][0], self.form_factor[0][-1] ]

    return

def form_factor_func(self, q):
    '''Defaults to making form factor function that sets all q above data range as endpoints
    and 0 below endpoints'''

    if self.form_factor_range[0] < q < self.form_factor_range[1]:
        return self.form_factor_interp(q)
    elif q <= self.form_factor_range[0]:
        return 0
    elif q >= self.form_factor_range[1]:
        return self.form_factor[1][-1]
    else:
        return 0
