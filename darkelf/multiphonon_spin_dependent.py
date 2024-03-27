import numpy as np
from numpy import linspace, sqrt, array, pi, cos, sin, dot, exp, sinh, log, log10, cosh, sinh
from scipy.special import erf, erfc, gamma, gammaincc, exp1
from scipy.interpolate import interp1d, interp2d
from scipy import integrate
import sys, os

############################################################################################
#  Cross section reach for kg-yr exposure for for multi-phonon excitations,
#  obtained by integrating the structure factor with approximations as in arXiv:2205.02250
#  The density of states is obtained from phonon_filename, and Fn is calculated from the DoS.


# There are three important functions:

# An internal function for plotting differential rates assuming incoherent spin dependent multiphonon interactions:
#        _dR_domega_multiphonons_SD,

# A complete rate without coherent single phonon and coherent single phonon only:
#       R_multiphonons_SD

# The final cross sections corresponding to 3 events/kg/yr:
#       sigma_multiphonons_SD


def _R_multiphonons_prefactor_SD(self, sigman, SD_op):
    # Input sigman in cm^2; output is the rate pre-factor in cm^2
    # Currently operators Of3 (scalar DM) and Of4 (psuedoscalar DM) have been implemented

    totalmass = sum(self.Amult*self.mvec)
    spin_independent_factor = sigman*((1/totalmass)* (self.rhoX*self.eVcm**3)/(2*self.mX*(self.muxnucleon)**2))*((1/self.eVcm**2)*(self.eVtoInvYr/self.eVtokg))

    if SD_op == 'Of3':
        return spin_independent_factor * 32 * (self.muxnucleon)**2 / self.mX**2 / self.q0**2 * self.mp**2
    elif SD_op == 'Of4':
        return spin_independent_factor * 192 * 0.25 * (self.muxnucleon)**2 / self.mX**2 / self.q0**2 * self.mp**2
    else:
        raise Exception("This spin dependent operator has not yet been defined")



def sigma_multiphonons_SD(self, threshold, SD_op='Of3'):
    '''
    returns DM-proton cross-section [cm^2] corresponding to 3 events/kg/yr
    Inputs
    ------
    threshold: float
      experimental threshold, in eV
    SD
    '''

    rate = self.R_multiphonons_SD(threshold, SD_op=SD_op)
    if rate != 0:
        return (3.0*1e-38)/rate
    else:
        return float('inf')


def R_multiphonons_SD(self, threshold, sigman=1e-38, SD_op='Of3'):
    """
    Returns rate for DM scattering with a harmonic lattice, including multiphonon contributions but excluding the coherent single phonon contribution

    Inputs
    ------
    threshold: float in [eV]
    sigma_n: float
        DM-nucleon cross section in [cm^2], defined with respect to the reference momentum of q0. (q0 is specified by the 'update_params' function)
        DM-nucleus cross section assumed to be coherently enhanced by A^2 by default (if dark photon flag not set)
    dark_photon: boole
        If set to True, a dark photon mediator is assumed, by setting f_d(q) = Z_d(q), with Z_d(q) the momentum dependent effective charges. If set to False, darkELF sets f_d=A_d, which corresponds to a scalar mediator with coupling to nuclei.

    Outputs
    -------
    rate as function of threshold, in [1/kg/yr]
    """

    if threshold > self.omegaDMmax:
        return 0
    else:
        npoints = 1000
        # For better precision, we use linear sampling for omega < max phonon energy and log sampling for omega > max phonon energy.
        if(threshold<self.dos_omega_range[-1]):
            omegarange_linear=np.linspace(threshold,np.min([self.dos_omega_range[-1],self.omegaDMmax]), npoints)
            dR_linear=[self._dR_domega_multiphonons_SD(omega, sigman=sigman, SD_op=SD_op) for omega in omegarange_linear]
            R_linear=np.trapz(dR_linear, omegarange_linear)
        else:
            R_linear=0.0
        if(self.omegaDMmax>self.dos_omega_range[-1]):
            omegarange_log=np.logspace(np.max([np.log10(self.dos_omega_range[-1]),np.log10(threshold)]),\
                                     np.log10(self.omegaDMmax), npoints)
            dR_log=[self._dR_domega_multiphonons_SD(omega, sigman=sigman, SD_op=SD_op) for omega in omegarange_log]
            R_log=np.trapz(dR_log, omegarange_log)
        else:
            R_log=0

        return R_linear+R_log



# Multiphonon_expansion term

def _dR_domega_multiphonons_SD(self, omega, sigman=1e-38, SD_op='Of3', npoints=200):

    if omega > self.omegaDMmax:
        return 0

    qmin = self.qmin(omega)

    qmax = self.qmax(omega)

    if qmin >= qmax:
        return 0

    qrange = np.linspace(qmin, qmax, npoints)

    # Choice of effective coupling
    try:
        fd = np.tile(np.array([self.f_d_vec]),(npoints, 1)).T
    except NameError:
        print('Please make sure the yaml file includes the f_d information in the unit_cell dictionary for this material')


    formfactorsquared = self.Fmed_nucleus_SD(qrange, SD_op=SD_op)**2

    S = 0
    for d in range(len(self.atoms)):
        # This is structure factor divided by (2 pi/ omega_c)
        S += self.Amult[d] * fd[d]**2 / (self.mvec[d])**2 * qrange**2 * self.S_d_squared[d] * self.C_ld(qrange, omega, d)

    # add contributions from all atoms
    dR_domega_dq = S * qrange * formfactorsquared * self.etav((qrange/(2*self.mX)) + omega/qrange)

    dR_domega = np.trapz(dR_domega_dq, qrange)

    return self._R_multiphonons_prefactor_SD(sigman, SD_op) * dR_domega
