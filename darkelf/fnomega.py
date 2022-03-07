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

    if type(dos_filename) is str:

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

    elif (len(dos_filename) == 2):

        for atom, pdos in enumerate(dos_filename):

            dos_path = datadir + self.target+'/'+ pdos
            fn_path = datadir + self.target+'/'+ pdos.replace('_pDoS','_Fn')

            phonons = 10

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

    else:
        print('dos_filename error')

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

    if self.n_atoms == 2:
        # If there are two distinct atoms, dos_filename needs to be
        # set to two element list of filenames
        if len(filename) != 2:
            print("Warning! Density of states not loaded. dos_filename must be list of two partial DoS filenames")
            self.phonon_DoS_loaded=False
        else:
            partial_dos_1_path = datadir + self.target+'/'+ filename[0]
            partial_dos_2_path = datadir + self.target+'/'+ filename[1]

            if (not os.path.exists(partial_dos_1_path)) or (not os.path.exists(partial_dos_2_path)):
                print("Warning! Density of states not loaded. Need to set dos_filename for both atoms")
                self.phonon_DoS_loaded=False

            else:
                self.phonon_DoS_loaded=True
                dosdat_1 = np.loadtxt(partial_dos_1_path).T
                dosdat_2 = np.loadtxt(partial_dos_2_path).T
                print("Loaded " + filename[0] + " and " + filename[1] + " for partial densities of states")
                self.phonon_DoS = np.array([dosdat_1, dosdat_2])
                self.DoS_interp = np.array([interp1d(i[0],i[1],kind='linear', fill_value = 0) for i in self.phonon_DoS])
                self.dos_omega_range = np.array([self.phonon_DoS[0][0][0],self.phonon_DoS[0][0][-1] ])
                # Assuming same omega range for both pDOS

                self.omega_bar = np.array([np.trapz(i[1]*i[0],
                                            x=i[0]) for i in self.phonon_DoS])
                self.omega_inverse_bar = np.array([np.trapz([i[1][j]/i[0][j] if i[0][j] != 0 else 0 for j in range(len(i[0]))],
                                            x=i[0]) for i in self.phonon_DoS])
                # if else statement so that there's no divide by 0 error at omega = 0

    else:
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


    if self.n_atoms == 2:
        partial_dos_path_1 = datadir + self.target+'/'+ filename[0]
        partial_fn_path_1 = datadir + self.target+'/'+ filename[0].replace('_pDoS','_Fn')
        partial_dos_path_2 = datadir + self.target+'/'+ filename[1]
        partial_fn_path_2 = datadir + self.target+'/'+ filename[1].replace('_pDoS','_Fn')

        if (not (os.path.exists(partial_fn_path_1) and os.path.exists(partial_fn_path_2))) and self.phonon_DoS_loaded:
            print("Warning! Fn(omega) functions not loaded. Need to calculate Fn(omega) from density of states")
            print("Use the function 'create_Fn_omega' to produce these files")
            self.phonon_Fn_loaded=False
        elif (not (os.path.exists(partial_fn_path_1) and os.path.exists(partial_fn_path_2))) and (not self.phonon_DoS_loaded):
            print("Warning! Fn(omega) functions not loaded. Need to load density of states")
            print("then use the function 'create_Fn_omega' to produce these Fn files")
            self.phonon_Fn_loaded=False
        else:
            self.phonon_Fn_loaded=True
            phonondat = [np.loadtxt(partial_fn_path_1).T, np.loadtxt(partial_fn_path_2).T]
            print("Loaded Fn(omega) functions corresponding to density of states in: ", filename[0], " ", filename[1])
            self.phonon_Fn = phonondat

        if self.phonon_Fn_loaded:
            # makes interpolations
            self.Fn_interpolations = {}
            for i, Fn in enumerate(self.phonon_Fn):
                tempdict = {}
                for n in range(1, len(Fn)):
                    tempdict[n] = interp1d(Fn[0], Fn[n], fill_value=0, bounds_error=False)
                self.Fn_interpolations[i] = tempdict


    else:
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
                self.Fn_interpolations[n] = interp1d(self.phonon_Fn[0], self.phonon_Fn[n], fill_value=0, bounds_error=False)

    return
