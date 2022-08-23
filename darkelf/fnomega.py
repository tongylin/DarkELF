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


def create_Fn_omega(self, datadir=None, dos_filename=None, phonons = 10,npoints=250):
    """
    Function to create .dat files for Fn(omega), used in multiphonons calculation.

    Default makes the Fn files for up to 10 phonons with the density of states
        loaded at class instantiation, then loads in new Fn.

    Inputs
    ------
    datadir: string
        directory with DoS file, default is self.eps_data_dir with all the other data
    dos_filename: list of strings
        DoS filename(s), default is self.dos_filename which is set when the class is instantiated
    phonons: int
        specifies up to how many phonons Fn is calculated for. Default value is 10.
    npoints: int
        number of omega points to compute Fn grid, default is 250
        (750 were used for calculations in draft, takes ~four hours)
    """

    if(datadir == None):
        datadir = self.eps_data_dir
    if(dos_filename == None):
        dos_filename = self.dos_filename

    for atom, pdos in enumerate(dos_filename):
        dos_path = datadir + self.target+'/'+ pdos
        fn_path = datadir + self.target+'/'+ pdos.replace('_pDoS','_Fn')

        # max number of phonons in Fn function
        phonons = 10
        # omega range in Fn function (determined by DoS range)
        omegarange = np.linspace(self.dos_omega_range[0],
                                (phonons/2)*self.dos_omega_range[1], npoints)

        omegapart = np.array([])

        for n in range(phonons):
            start = time.time()
            narray = np.array([])
            for i, omega in enumerate(omegarange):
                narray = np.append(narray, self.Fn_vegas(omega, n + 1, atom))
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
        print("result saved in "+fn_path)

    self.load_Fn(datadir, dos_filename)

    return



##############################################################################
# Auxiliary functions for create_Fn_omega to call

# This is the integrand for the Fn function, where the final omega_j variable has been replaced with (omega - \sum_{i \neq j} \omega_i)
def Fn_integrand(self, omega, n, atom):
    # minimum and maximum energy for a single phonon
    omegaminus = self.dos_omega_range[0]
    omegaplus = self.dos_omega_range[1]
    def DoSintegrand(omegaivec):
        if len(omegaivec) == n - 1:
            omegaj = omega - (np.sum(omegaivec))
            if omegaminus < omegaj < omegaplus:
                result = self.DoS_interp[atom](omegaj)/(omegaj)
                for omegai in omegaivec:
                    result *= self.DoS_interp[atom](omegai)/omegai
                return result
            else:
                return 0
        else:
             raise Exception('wrong n')
    return DoSintegrand

# This function integrates Fn_integrand using vegas for a given omega and n
def Fn_vegas(self, omega, n, atom):

    # There is a minimum number of phonons for a given omega, return 0 if n isn't large enough
    minimum = np.floor(omega/self.dos_omega_range[1]) + 1
    if n < minimum:
        return 0

    # Compute integral if n > 1
    if (n > 1) and (isinstance(n, int)):
        integrationrange = (n - 1)*[[self.dos_omega_range[0],self.dos_omega_range[1]]]
        integ = vegas.Integrator(integrationrange)
        # first perform adaptation; we will throw away these results
        # This is discussed in https://vegas.readthedocs.io/en/latest/tutorial.html#basic-integrals, under "Early Iterations"
        integ(self.Fn_integrand(omega=omega, n=n, atom=atom), nitn=10, neval=1000)
        # Keep these results after adaptation
        result = integ(self.Fn_integrand(omega=omega, n=n, atom=atom), nitn=10, neval=1000)
        # print(result.summary())
        return gvar.mean(result)/factorial(n)
    elif n == 1:
        # note this will automatically return 0 if omega is outside the range for phonon_DoS
        if omega <= 0:
            return 0
        else:
            return self.DoS_interp[atom](omega)/omega
    else:
        raise Exception('n must be a nonnegative integer')




############################################################################################

# Function to load density of states
def load_phonon_dos(self,datadir,filename):

    dos_paths = [datadir + self.target+'/'+ fi for fi in filename]
    self.phonon_DoS = []

    for file in dos_paths:

        if not os.path.exists(file):
            print(f"Warning, {file} does not exist! Density of states not loaded. Need to set dos_filename for all atoms.")
        else:
            (self.phonon_DoS).append(np.loadtxt(file).T)
            print("Loaded " + file + " for partial densities of states")

    self.DoS_interp = np.array([interp1d(i[0],i[1],kind='linear', fill_value = 0, bounds_error=False) for i in self.phonon_DoS])
    self.dos_omega_range = np.array([ self.phonon_DoS[0][0][0], self.phonon_DoS[0][0][-1] ])
    # Assuming same omega range for all pDOS!

    self.omega_bar = np.array([np.trapz(i[1]*i[0], x=i[0]) for i in self.phonon_DoS])
    self.omega_inverse_bar = np.array([np.trapz([i[1][j]/i[0][j] if i[0][j] != 0 else 0 for j in range(len(i[0]))],
                                            x=i[0]) for i in self.phonon_DoS])
    # if else statement in second line so that there's no divide by 0 error at omega = 0

    return


# Function to load Fn(omega) data corresponding to density of states file
def load_Fn(self,datadir,filename):

    Fn_paths = [datadir + self.target+'/'+ fi.replace('_pDoS','_Fn') for fi in filename]

    self.phonon_Fn = []
    for file in Fn_paths:

        if not os.path.exists(file):
            print(f"Warning! {file} does not exist! Need to calculate Fn(omega) from DoS. Use the function 'create_Fn_omega' to produce these files ")

        else:
            (self.phonon_Fn).append(np.loadtxt(file).T)
            print("Loaded " + file + " for Fn(omega)")

    # dictionary for Fn functions in terms of number of phonons (offset from index by 1)
    self.Fn_interpolations = {}
    for i, Fn in enumerate(self.phonon_Fn):
        tempdict = {}
        for n in range(1, len(Fn)):
            tempdict[n] = interp1d(Fn[0], Fn[n], fill_value=0, bounds_error=False)
        self.Fn_interpolations[i] = tempdict

    return
