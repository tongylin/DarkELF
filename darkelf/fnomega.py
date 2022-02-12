import numpy as np
from numpy import linspace, sqrt, array, pi, cos, sin, dot, exp, sinh, log, log10, cosh, sinh
from math import factorial
from scipy.special import erf, erfc, gamma, gammaincc, exp1
from scipy.interpolate import interp1d, interp2d
from scipy import integrate
import vegas
import gvar
import time
import sys, os, glob
import pandas as pd

##############################################################################
# Makes Fn(omega) files from an input DoS file


def create_Fn_omega(self, datadir=None, dos_filename=None, npoints=250):
    """
    Function to create .dat files for Fn(omega), used in multiphonons calculation.

    Default makes the Fn files for 10 phonons with the density of states
        loaded at class instantiation, then loads in new Fn.
    
    Inputs
    ------
    datadir: string
        directory with DoS file, default is self.eps_data_dir with all the other data
    dos_filename: string
        DoS filename, default is self.dos_filename which is set when the class is instantiated
    npoints: int
        number of omega points to compute Fn grid, default is 250
        (1000 were used for calculations on draft, takes ~four hours)
    """

    if(datadir == None):
        datadir = self.eps_data_dir
    if(dos_filename == None):
        dos_filename = self.dos_filename

    dos_path = datadir + self.target+'/'+ dos_filename
    fn_path = datadir + self.target+'/'+ dos_filename.replace('_DoS','_Fn')

    phonons = 10

    omegarange = np.linspace(self.dos_omega_range[0],
                            (phonons/2)*self.dos_omega_range[1], npoints)

    omegapart = np.array([])

    for n in range(phonons):
        start = time.time()
        narray = np.array([])
        for i, omega in enumerate(omegarange):
            narray = np.append(narray, self.Fn_vegas(omega, n + 1))
            if n > 0 and ((i+1) % 50 == 0):
                print(f'Finished point #{i + 1} out of {npoints} for {n + 1} phonons')
        end = time.time()
        if n > 0:
            print(f'{end - start} seconds for {n + 1} phonons')
        omegapart = np.append(omegapart, narray)

    omegapart = omegapart.reshape((phonons, len(omegarange)))
    omegadata = np.reshape(np.append(omegarange,omegapart),(phonons+1,len(omegarange)))
    label = '# First column is omega in [eV], second column is F1(omega) in [eV-2], third column is F2(omega) in [eV-3], etc.'
    np.savetxt(fn_path, omegadata.T,header=label)

    self.load_Fn(datadir, dos_filename)

    return



##############################################################################
# Auxiliary functions for create_Fn_omega to call

# This is the integrand for the Fn function, where the final omega_j variable has been replaced with (omega - \sum_{i \neq j} \omega_i)
def Fn_integrand(self, omega, n):
    # minimum and maximum energy for a single phonon
    omegaminus = self.phonon_DoS[0][0]
    omegaplus = self.phonon_DoS[0][-1]
    def DoSintegrand(omegaivec):
        if len(omegaivec) == n - 1:
            omegaj = omega - (np.sum(omegaivec))
            if omegaminus < omegaj < omegaplus:
                result = self.DoS_interp(omegaj)/(omegaj)
                for omegai in omegaivec:
                    result *= self.DoS_interp(omegai)/omegai
                return result
            else:
                return 0
        else:
             raise Exception('wrong n')
    return DoSintegrand

# This function integrates Fn_integrand using vegas for a given omega and n
def Fn_vegas(self, omega, n):

    # There is a minimum number of phonons for a given omega, return 0 if n isn't large enough
    minimum = np.floor(omega/self.phonon_DoS[0][-1]) + 1
    if n < minimum:
        return 0
    
    # Compute integral if n > 1
    if (n > 1) and (isinstance(n, int)):
        integrationrange = (n - 1)*[[self.phonon_DoS[0][0],self.phonon_DoS[0][-1]]]
        integ = vegas.Integrator(integrationrange)
        # first perform adaptation; we will throw away these results
        # This is discussed in https://vegas.readthedocs.io/en/latest/tutorial.html#basic-integrals, under "Early Iterations"
        integ(self.Fn_integrand(omega=omega, n=n), nitn=10, neval=1000)
        # Keep these results after adaptation
        result = integ(self.Fn_integrand(omega=omega, n=n), nitn=10, neval=1000)
        # print(result.summary())
        return gvar.mean(result)/factorial(n)
    elif n == 1:
        # note this will automatically return 0 if omega is outside the range for phonon_DoS
        return self.DoS_interp(omega)/omega
    else:
        raise Exception('n must be a nonnegative integer')




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
