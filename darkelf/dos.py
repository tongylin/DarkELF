import numpy as np
from numpy import linspace, sqrt, array, pi, cos, sin, dot, exp, sinh, log, log10, cosh, sinh
from scipy.interpolate import interp1d, interp2d
from scipy import integrate
import sys, os, glob
import pandas as pd

############################################################################################

# Function to load density of states

def load_phonon_dos(self,datadir,filename):

    dos_path = datadir + self.target+'/'+ filename

    if( not os.path.exists(dos_path)):
        print("Warning! Density of states not loaded. Need to set dos_filename ")
        self.phonon_DoS_loaded=False
    else:
        self.phonon_DoS_loaded=True
        dosdat = np.loadtxt(dos_path).T
        print("Loaded " + filename + " for density of states")
        self.phonon_DoS = dosdat
        self.DoS_interp = interp1d(self.phonon_DoS[0],self.phonon_DoS[1],kind='linear', fill_value = 0)
        self.dos_omega_range = [ dosdat[0][0], dosdat[0][-1] ]
        self.omega_bar = np.trapz(self.phonon_DoS[1]*self.phonon_DoS[0],
                                    x=self.phonon_DoS[0])
        self.omega_inverse_bar = np.trapz(self.phonon_DoS[1]/self.phonon_DoS[0],
                                    x=self.phonon_DoS[0])
    return

# Function to load Fn(omega) data corresponding to density of states file

def load_Fn(self,datadir,filename):

    dos_path = datadir + self.target+'/'+ filename
    fn_path = datadir + self.target+'/'+ filename.replace('_DoS','_Fn')

    if (not os.path.exists(fn_path)) and self.phonon_DoS_loaded:
        print("Warning! Fn(omega) functions not loaded. Need to calculate Fn(omega) from density of states")
        print("Use the function 'create_Fn_omega' to produce these files")
        self.phonon_Fn_loaded=False
    elif (not os.path.exists(fn_path)) and (not self.phonon_DoS_loaded):
        print("Warning! Fn(omega) functions not loaded. Need to load density of states")
        print("then use the function 'create_Fn_omega' to produce these Fn files")
        self.phonon_Fn_loaded=False
    else:
        self.phonon_Fn_loaded=True
        phonondat = np.loadtxt(fn_path).T
        print("Loaded Fn(omega) functions corresponding to density of states in: ", filename)
        self.phonon_Fn = phonondat

    if self.phonon_Fn_loaded:
        # makes interpolations
        self.Fn_interpolations = {}
        for n in range(1, len(self.phonon_Fn)):
            self.Fn_interpolations[n] = interp1d(self.phonon_Fn[0], self.phonon_Fn[n])

    return
