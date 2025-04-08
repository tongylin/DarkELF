import numpy as np
from numpy import linspace, sqrt, array, pi, cos, sin, dot, exp, sinh, log, log10, cosh, sinh
from math import factorial
from scipy.special import erf, erfc, gamma, gammaincc, exp1
from scipy.interpolate import interp1d, interp2d
from scipy import integrate
#import vegas
#import gvar
import time
import sys, os, glob
import pandas as pd


def debye_waller(self, q):
    '''Debye Waller factor exp(-2 W(q)) where W(q) = q^2 omega / (4 A mp)
    Inputs
    ------
    q: array in units of eV. For each q, gives the Debye-Waller factor for each atom '''

    one_over_q2_char = self.omega_inverse_bar/(2*self.Avec*self.mp)[None,...]
    q = q[...,None]
    return np.where(np.less(one_over_q2_char*q**2, 0.03), 1, exp(-one_over_q2_char*q**2))


##############################################################################
# Calculates the C_ld's using both multiphonon expansion and impulse approximation.

def C_ld(self, qrange, omega, d, q_IA_factor = 2):
    """
    Calculates the auto-correlation function C_ld(q, omega), which is independent of the lattice site.
    For q < q_IA_factor*sqrt(2 m_d omega_bar_d), it uses the multiphonon expansion from Fn(omega) files.
    For q >= q_IA_factor*sqrt(2 m_d omega_bar_d), it uses the impulse approximation.

    The function checks whether the qrange is physical, but does not check if single phonon analysis
    should be used instead of the multiphonon expansion.

    Inputs
    ------
    qrange: numpy array
        list of q values
    omega: float
        single energy
    d: int
        integer specifying atom in the crystal. index is same as used for Avec
    """

    if omega > self.omegaDMmax:
        return 0

    qmin = self.qmin(omega)
    qmax = self.qmax(omega)

    assert qrange[0] >= qmin and qrange[-1] <= qmax, "the range of q's is unphysical"


    # for q>q_IA_cut, the impulse approximation is used
    q_IA_cut = q_IA_factor * sqrt(2*self.Avec[d]*self.mp*self.omega_bar[d])
    # max([ q_IA_factor * sqrt(2*self.Avec[i]*self.mp*self.omega_bar[i]) for i in range(len(self.atoms))])

    if q_IA_cut >= qrange[-1]:
        q_IA_cut_index = len(qrange)
    else:
        q_IA_cut_index = next(x for x, val in enumerate(qrange) if val > q_IA_cut)


    q_multiphonon = qrange[0:q_IA_cut_index]
    q_IA = qrange[q_IA_cut_index::]

    # n_min: minimum number of phonons 
    # n_max_plus_one: maximum number of phonons + 1 (since range below goes up to n_max - 1)
    n_min = int(np.ceil(omega / self.dos_omega_range[1]))
    n_max_plus_one = len(self.phonon_Fn[d])

    # Calculation of the c_ld's via multiphonon expansion
    if q_multiphonon.size == 0:
        cld_multiphonon = np.empty(0)
    else:
        cld_multiphonon=np.zeros(len(q_multiphonon))
        debye_waller_factor = self.debye_waller(q_multiphonon).T
        if(n_min < n_max_plus_one):
            for n in range(n_min, n_max_plus_one):
                # Debye-Waller now included in qpart
                qpart = q_multiphonon**(2*n) * debye_waller_factor[d]
                # Notes:
                #  1. The atom multiplicity function is not included here, it should be included elsewhere.
                #  2. The 1/n! factor is found in the Fn function
                #  3. This is c_ld divided by (2 pi/ V)
                cld_multiphonon += (1/(2*self.Avec[d]*self.mp))**n * qpart * self.Fn_interpolations[d][n](omega)


    # calculation of the c_ld's via the impulse approximation
    if q_IA.size == 0:
        cld_IA = np.empty(0)
    else:
        # Width of gaussian
        deltaq = sqrt(self.omega_bar[d] / (2* self.Avec[d] * self.mp)) * q_IA

        # This is c_ld divided by (2 pi/ V)
        cld_IA = (1/(deltaq*sqrt(2*pi)))*exp(-(omega - q_IA**2/(2*self.Avec[d]*self.mp))**2/(2*deltaq**2))

    cld = np.concatenate((cld_multiphonon, cld_IA))
    return cld




##############################################################################
# Makes Fn(omega) files from an input DoS file

#Calculate Tn and Fn for given DOS data
def create_Fn_omega(self,datadir=None, dos_filename=None, phonons = 10, npoints=1000):
    """
    Function to create an array of Fn values for a given material.

    Uses recursive relation on Tn = n! * Fn and then divides by n! at the end for Fn

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

    # omega range for Fn files (determined by DoS range) - this could be expanded as needed.
    omegarange = np.linspace(self.dos_omega_range[0],
                                (phonons/2)*self.dos_omega_range[1], npoints)

    # omega array for each atom
    omega_d = [self.phonon_DoS[i][0] for i in range(len(self.atoms)) ]
    # Extract D(omega_n)/omega_n from DoS data
    T1_d =  [self.phonon_DoS[i][1]/self.phonon_DoS[i][0] for i in range(len(self.atoms)) ]

    # Interpolated T1 function
    T1_d_interp = [interp1d(DoS[0],DoS[1]/DoS[0],fill_value=0,bounds_error=False) for DoS in self.phonon_DoS]

    for atom, pdos in enumerate(dos_filename):
        fn_path = datadir + self.target+'/'+ pdos.replace('_pDoS','_Fn')

        # Create array that stores each T1 or F1 function over the entire omega range of interest
        # Add list of F2, F3,... up to FN to this array and return it
        Tn_array = np.array([T1_d_interp[atom](omegarange)])
        Fn_array = np.array([Tn_array[0]])

        T_n_minus_1_interp = T1_d_interp[atom]

        for n in range(1,phonons):
            Tn_array = np.append(Tn_array, \
                                  [ [np.trapz(T1_d[atom]*T_n_minus_1_interp(W-omega_d[atom]), omega_d[atom]) for W in omegarange] ], axis=0)
            Fn_array = np.append(Fn_array, [Tn_array[-1]/factorial(n+1)],axis=0)

            # Update the T_(n-1) function using the last computed integral
            #print(omegarange)
            #print(Tn_array)
            #print(Tn_array[-1])
            T_n_minus_1_interp = interp1d(omegarange, Tn_array[-1],fill_value=0,bounds_error=False,kind='linear')

        Fndata = np.append([omegarange],Fn_array,axis=0)
        label = '# First column is omega in [eV], second column is F1(omega) in [eV-2], third column is F2(omega) in [eV-3], etc.'
        np.savetxt(fn_path, Fndata.T,header=label)
        print("result saved in "+fn_path)

    self.load_Fn(datadir, dos_filename)

    return

############################################################################################

# Function to load density of states
def load_phonon_dos(self,datadir,filename):

    dos_paths = [datadir + self.target+'/'+ fi for fi in filename]
    self.phonon_DoS = []

    for file in dos_paths:
        if not os.path.exists(file):
            print(f"Warning, {file} does not exist! Density of states not loaded. Need to set dos_filename for all atoms.")
            self.phonon_dos_loaded=False
        else:
            (self.phonon_DoS).append(np.loadtxt(file).T)
            print("Loaded " + file + " for partial densities of states")
            self.phonon_dos_loaded=True

    if self.phonon_dos_loaded:       
        self.DoS_interp = np.array([interp1d(i[0],i[1],kind='linear', fill_value = 0, bounds_error=False) for i in self.phonon_DoS])
        self.dos_omega_range = np.array([ self.phonon_DoS[0][0][0], self.phonon_DoS[0][0][-1] ])
        # Assuming same omega range for all pDOS!

        self.omega_bar = np.array([np.trapz(i[1]*i[0], x=i[0]) for i in self.phonon_DoS])
        self.omega_inverse_bar = np.array([np.trapz([i[1][j]/i[0][j] if i[0][j] != 0 else 0 for j in range(len(i[0]))],
                                                x=i[0]) for i in self.phonon_DoS])
        # if else statement in second line so that there's no divide by 0 error at omega = 0
    else:
        self.dos_omega_range=np.array([0,0.1]) # arbitrary, not used

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
            tempdict[n] = interp1d(Fn[0], Fn[n], fill_value=0, bounds_error=False, kind='linear')
        self.Fn_interpolations[i] = tempdict

    return



############################################################################################


# Old functions using vegas to evaluate multiphonon integrals -- very slow.

def create_Fn_omega_vegas(self, datadir=None, dos_filename=None, phonons = 10, npoints=250):
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


