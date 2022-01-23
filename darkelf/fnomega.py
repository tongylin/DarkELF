import numpy as np
from numpy import linspace, sqrt, array, pi, cos, sin, dot, exp, sinh, log, log10, cosh, sinh
from math import factorial
from scipy.special import erf, erfc, gamma, gammaincc, exp1
from scipy.interpolate import interp1d, interp2d
from scipy import integrate
import vegas
import gvar
import time

##############################################################################
# Makes Fn(omega) files from an input DoS file


def create_Fn_omega(self, datadir=None, filename=None):
    # Default makes the Fn files for 10 phonons with the density of states
    # loaded at class instantiation, then loads in new Fn

    datadir = self.eps_data_dir
    filename = self.dos_filename

    dos_path = datadir + self.target+'/'+ filename
    fn_path = datadir + self.target+'/'+ filename.replace('_DoS','_Fn')

    phonons = 10

    # Can change this number of points (I used 1000 for calculations on draft) to calculate Fn grid on
    # to some smaller value to reduce the runtime, 1000 points currently takes ~four hours

    npoints = 250

    omegarange = np.linspace(self.dos_omega_range[0],
                            (phonons/2)*self.dos_omega_range[1], npoints)

    omegapart = np.array([])

    for n in range(phonons):
        start = time.time()
        narray = np.array([])
        for i, omega in enumerate(omegarange):
            narray = np.append(narray, self.structurefactornomegapart(omega, n + 1))
            if n > 0:
                print(f'Finished point #{i + 1} out of {npoints} for {n + 1} phonons')
        end = time.time()
        if n > 0:
            print(f'{end - start} seconds for {n + 1} phonons')
        omegapart = np.append(omegapart, narray)
    omegapart = omegapart.reshape((phonons, len(omegarange)))
    omegadata = np.reshape(np.append(omegarange,omegapart),(phonons+1,len(omegarange)))
    label = '# First column is omega in [eV], second column is F1(omega) in [eV-2], third column is F2(omega) in [eV-3], etc.'
    np.savetxt(fn_path, omegadata.T,header=label)

    self.load_Fn(datadir, filename)

    return



##############################################################################
# Auxiliary functions for create_Fn_omega to call

def DoSintegrandn(self, omega, n):
    omegaminus = self.phonon_DoS[0][0]
    omegaplus = self.phonon_DoS[0][-1]
    def DoSintegrand(omegaintegrates):
        if len(omegaintegrates) == n - 1:
            if omegaminus < omega - (np.sum(omegaintegrates)) < omegaplus:
                result = self.DoS_interp(omega - np.sum(omegaintegrates))/(omega - np.sum(omegaintegrates))
                for i in omegaintegrates:
                    result *= self.DoS_interp(i)/i
                return result
            else:
                return 0
        else:
             raise Exception('wrong n')
    return DoSintegrand


def vegasintegrated(self, omega, n):
    if (n > 1) and (isinstance(n, int)):
        integrationrange = (n - 1)*[[self.phonon_DoS[0][0],self.phonon_DoS[0][-1]]]
        integ = vegas.Integrator(integrationrange)
        integ(self.DoSintegrandn(omega=omega, n=n), nitn=10, neval=1000)
        result = integ(self.DoSintegrandn(omega=omega, n=n), nitn=10, neval=1000)
        # print(result.summary())
        return gvar.mean(result)
    elif n == 1:
        if self.phonon_DoS[0][0] < omega < self.phonon_DoS[0][-1]:
            return self.DoS_interp(omega)/omega
        else:
            return 0
    else:
        raise Exception('n must be a nonnegative integer')

def structurefactornomegapart(self, omega, n):
    minimum = np.floor(omega/self.phonon_DoS[0][-1]) + 1
    if n >= minimum:
        integralterm = self.vegasintegrated(omega, n)
    else:
        integralterm = 0
    return integralterm/factorial(n)
