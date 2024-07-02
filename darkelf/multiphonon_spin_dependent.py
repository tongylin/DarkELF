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


def _R_multiphonons_prefactor_SD(self, sigman):
    # Input sigman in cm^2; output is the rate pre-factor in cm^2
    # Currently operators Of3 (scalar mediator) and Of4 (psuedoscalar mediator) have been implemented

    totalmass = sum(self.Amult*self.mvec)
    spin_independent_factor = sigman*((1/totalmass)* (self.rhoX*self.eVcm**3)/(2*self.mX*(self.muxnucleon)**2))*((1/self.eVcm**2)*(self.eVtoInvYr/self.eVtokg))

    if self.SD_op == 'Of3':
        return spin_independent_factor * 32 / self.q0**2 #* (self.muxnucleon)**2 / self.mX**2 #* self.mp**2
    elif self.SD_op == 'Of4':
        return spin_independent_factor * 192 * 0.25 / self.q0**2 #* (self.muxnucleon)**2 / self.mX**2 / self.q0**2 #* self.mp**2
    elif self.SD_op == 'Of8':
        return spin_independent_factor / 12 * 0.25 #* (self.muxnucleon)**2 / self.mX**2
    else:
        raise Exception("This spin dependent operator has not yet been defined")



def sigma_multiphonons_SD(self, threshold, nucleon='p'):
    '''
    returns DM-proton cross-section [cm^2] corresponding to 3 events/kg/yr
    Inputs
    ------
    threshold: float
      experimental threshold, in eV
    nucleon: string
      Choose from p or n. If this matches the defined ratio of g_p and g_n, the
      calculation will proceed. Otherwise, you will be prompted to reset the ratio
    '''
    if ((nucleon == 'p') & (self.gp_gn_ratio == 'g_n/g_p' )) or ((nucleon == 'n') & (self.gp_gn_ratio == 'g_p/g_n' )):
        pass
    else:
        print('Chosen nucleon must match ratio of g_p and g_n. Reset that ratio if this nucleon is desired.')
        return np.nan

    rate = self.R_multiphonons_SD(threshold)
    if rate != 0:
        return (3.0*1e-38)/rate
    else:
        return float('inf')


def R_multiphonons_SD(self, threshold, sigman=1e-38):
    """
    Returns rate for DM scattering with a harmonic lattice, including multiphonon contributions but excluding the coherent single phonon contribution

    Inputs
    ------
    threshold: float in [eV]
    sigma_n: float
        DM-nucleon cross section in [cm^2], defined with respect to the reference momentum of q0. (q0 is specified by the 'update_params' function)
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
            dR_linear=[self._dR_domega_multiphonons_SD(omega, sigman=sigman) for omega in omegarange_linear]
            R_linear=np.trapz(dR_linear, omegarange_linear)
        else:
            R_linear=0.0
        if(self.omegaDMmax>self.dos_omega_range[-1]):
            omegarange_log=np.logspace(np.max([np.log10(self.dos_omega_range[-1]),np.log10(threshold)]),\
                                     np.log10(self.omegaDMmax), npoints)
            dR_log=[self._dR_domega_multiphonons_SD(omega, sigman=sigman) for omega in omegarange_log]
            R_log=np.trapz(dR_log, omegarange_log)
        else:
            R_log=0

        return R_linear+R_log



def _dR_domega_multiphonons_SD(self, omega, sigman=1e-38, npoints=200):

    if self.haxton:
        return self._dR_domega_Haxton_SD(omega, sigman)

    if omega > self.omegaDMmax:
        return 0

    qmin = self.qmin(omega)

    qmax = self.qmax(omega)

    if qmin >= qmax:
        return 0

    qrange = np.linspace(qmin, qmax, npoints)

    formfactorsquared = self.Fmed_nucleus_SD(qrange)**2

    S = 0
    for d in range(len(self.atoms)):
        # This is structure factor divided by (2 pi/ omega_c)
        # S += self.Amult[d] * fd[d]**2 / (self.mvec[d])**2 * qrange**2 * self.S_d_squared[d] * self.C_ld(qrange, omega, d)
        if (self.SD_op == 'Of3') or (self.SD_op == 'Of4'):
            S += self.Amult[d] * self.isotope_averaged_factors[d] * qrange**2 * self.C_ld(qrange, omega, d) #isotope averaging over S_N.f_d/m_d
        elif self.SD_op == 'Of8':
            S += self.Amult[d] * self.isotope_averaged_factors[d] * self.C_ld(qrange, omega, d) #isotope averaging over S_N.f_d
        else:
            raise Exception("This spin dependent operator has not yet been defined")
    # add contributions from all atoms
    dR_domega_dq = S * qrange * formfactorsquared * self.etav((qrange/(2*self.mX)) + omega/qrange)

    dR_domega = np.trapz(dR_domega_dq, qrange)

    return self._R_multiphonons_prefactor_SD(sigman) * dR_domega


def _dR_domega_Haxton_SD(self, omega, sigman=1e-38):

    if omega > self.omegaDMmax:
        return 0

    totalmass = sum(self.Amult*self.mvec)
    mT = self.mvec
    muT = mT * self.mX / (self.mX + mT)
    qrange = np.sqrt(2 * mT * omega)
    vmin = qrange / 2 / muT
    formfactor = (self.q0**2 + self.mMed**2)/(qrange**2 + self.mMed**2)

    if self.SD_op == 'Of4':
        dR_domega = sigman * self.NTkg * (self.rhoX * self.eVcm**3) / self.mX**3 * 8 / self.q0**4 * sum(self.Amult * formfactor**2 * qrange**4 * self.etav(vmin) * mT * self.isotope_averaged_factors_haxton) * self.mX**2 / (self.muxnucleon)**2
    elif self.SD_op == 'Of8':
        dR_domega = sigman * self.NTkg * (self.rhoX * self.eVcm**3) / self.mX**3 / 6 * sum(self.Amult * formfactor**2 * self.etav(vmin) * mT * self.isotope_averaged_factors_haxton) * self.mX**2 / (self.muxnucleon)**2
    else:
        raise Exception("This spin dependent operator has not yet been defined")

    return dR_domega * (self.eVcm**-2) * (self.eVtoInvYr)
