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


# There are five important functions:

# Two internal functions for plotting differential rates without coherent single phonon and coherent single phonon only:
#        _dR_domega_multiphonons_no_single, _dR_domega_coherent_single

# Two complete rates without coherent single phonon and coherent single phonon only:
#       R_multiphonons_no_single, R_single_phonon

# Note that in R_single_phonon and _dR_domega_coherent_single, the optical phonon part
#   assumes zincblende crystal structures only.

# The final cross sections corresponding to 3 events/kg/yr:
#       sigma_multiphonons



### Useful functions for multiphonon calculations
def _debye_waller_scalar(self, q):
    # Debye Waller factor exp(-2 W(q)) where W(q) = q^2 omega / (4 A mp) for a given atom
    # Debye-Waller factor set to 1 when q^2 small relative to characteristic q, for numerical convenience
    one_over_q2_char = self.omega_inverse_bar/(2*self.Avec*self.mp)
    return np.where(np.less(one_over_q2_char*q**2, 0.03), 1, exp(-one_over_q2_char*q**2))


def debye_waller(self, q):
    '''Debye Waller factor exp(-2 W(q)) where W(q) = q^2 omega / (4 A mp)
    Inputs
    ------
    q: float or array in units of eV. For each q, gives the Debye-Waller factor for each atom '''
    if (isinstance(q,(np.ndarray,list)) ):
        return np.array([self._debye_waller_scalar(qi) for qi in q])
    elif(isinstance(q,float)):
        return self._debye_waller_scalar(q)
    else:
        print("Warning! debye_waller function given invalid quantity ")
        return 0.0

def _R_multiphonons_prefactor(self, sigman):
    # Input sigman in cm^2; output is the rate pre-factor in cm^2

    totalmass = self.mp * sum(self.Amult*self.Avec)
    return sigman*((1/totalmass)*
                (self.rhoX*self.eVcm**3)/(2*self.mX*(self.muxnucleon)**2))*((1/self.eVcm**2)*
                (self.eVtoInvYr/self.eVtokg))



def sigma_multiphonons(self, threshold, dark_photon=False):
    '''
    returns DM-proton cross-section [cm^2] corresponding to 3 events/kg/yr
    Inputs
    ------
    threshold: float
      experimental threshold, in eV
    dark_photon: Bool
      If set to True, a dark photon mediator is assumed, by setting f_d(q) = Z_d(q), with Z_d(q) the momentum dependent effective charges. If set to False, darkELF sets f_d=A_d, which corresponds to a scalar mediator with coupling to nuclei.
    '''
    if dark_photon:
      assert self.fd_loaded, "Error: effective charge not loaded. Cannot perform calculation for dark photon mediator."

    rate = self.R_multiphonons_no_single(threshold, dark_photon=dark_photon) + self.R_single_phonon(threshold, dark_photon=dark_photon)
    if rate != 0:
        return (3.0*1e-38)/rate
    else:
        return float('inf')


def R_multiphonons_no_single(self, threshold, sigman=1e-38, dark_photon=False):
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
    if dark_photon:
      assert self.fd_loaded, "Error: effective charge not loaded. Cannot perform calculation for dark photon mediator."

    if threshold > self.omegaDMmax:
        return 0
    else:
        npoints = 1000
        # For better precision, we use linear sampling for omega < max phonon energy and log sampling for omega > max phonon energy.
        if(threshold<self.dos_omega_range[-1]):
            omegarange_linear=np.linspace(threshold,np.min([self.dos_omega_range[-1],self.omegaDMmax]), npoints)
            dR_linear=[self._dR_domega_multiphonons_no_single(omega, sigman=sigman, dark_photon=dark_photon) for omega in omegarange_linear]
            R_linear=np.trapz(dR_linear, omegarange_linear)
        else:
            R_linear=0.0
        if(self.omegaDMmax>self.dos_omega_range[-1]):
            omegarange_log=np.logspace(np.max([np.log10(self.dos_omega_range[-1]),np.log10(threshold)]),\
                                     np.log10(self.omegaDMmax), npoints)
            dR_log=[self._dR_domega_multiphonons_no_single(omega, sigman=sigman, dark_photon=dark_photon) for omega in omegarange_log]
            R_log=np.trapz(dR_log, omegarange_log)
        else:
            R_log=0

        return R_linear+R_log



# Multiphonon_expansion term

def _dR_domega_multiphonons_no_single(self, omega, sigman=1e-38, dark_photon=False, npoints=200):

    '''dR_domega single-phonon coherent removed'''
    if(dark_photon): # check if effective charges are loaded
        assert self.fd_loaded, "Error: effective charge not loaded. Cannot perform calculation for dark photon mediator."

    if omega > self.omegaDMmax:
        return 0

    if (omega > self.dos_omega_range[-1]):
        qmin = self.qmin(omega)
    else: ## For q<qBZ and omega<max phonon energy, we use the single phonon rate.
        qmin = max(self.qmin(omega), self.qBZ)

    qmax = self.qmax(omega)

    if qmin >= qmax:
        return 0
    
    qrange = np.linspace(qmin, qmax, npoints)

    # Choice of effective coupling
    if dark_photon:
        fd = np.array([self.fd_darkphoton[i](qrange) for i in range(len(self.atoms))])
    else:
        fd = np.tile(self.Avec,(npoints, 1)).T

    formfactorsquared = self.Fmed_nucleus(qrange)**2

    S = 0
    for d in range(len(self.atoms)):
        # This is structure factor divided by (2 pi/ omega_c)
        S += self.Amult[d] * fd[d]**2 * self.C_ld(qrange, omega, d)

    # add contributions from all atoms
    dR_domega_dq = S * qrange * formfactorsquared * self.etav((qrange/(2*self.mX)) + omega/qrange)

    dR_domega = np.trapz(dR_domega_dq, qrange)
    
    return self._R_multiphonons_prefactor(sigman) * dR_domega
 
############################################################################################
#
# Single phonon coherent term

# !TL: functions wanted:
#      _dR_domega_coherent_single -- internal function used for plotting
#      R_coherent_single_phonon -- total rate, obtained analytically

def _dR_domega_coherent_single(self, omega, sigman=1e-38, dark_photon=False):
#   internal function used for plotting, takes in single float values of omega
#   Note! don't integrate over omega here as there are artificial sharp peaks
    if(dark_photon): # check if effective charges are loaded
        assert self.fd_loaded, "Error: effective charge not loaded. Cannot perform calculation for dark photon mediator."

    if omega > self.omegaDMmax:
        return 0

    qmin = self.qmin(omega)
    qmax = min(self.qmax(omega), self.qBZ)

    if qmin > qmax:
        return 0

    # following stuff is doing the acoustic part analytically
    if dark_photon:
        fd = np.array([self.fd_darkphoton[i](omega/self.cLA) for i in range(len(self.atoms))])*sqrt(self.debye_waller(omega/self.cLA))
    else:
        fd = self.Avec

    formfactorsquared = self.Fmed_nucleus(omega/self.cLA)**2

    if (omega < 2*self.mX*self.cLA*(self.vmax - self.cLA)) and (omega < self.cLA*self.qBZ) and (omega < self.dos_omega_range[-1]):
        # !TL: multiplicity of atoms added here in sum over fd, and in 1/(total mass)
        acoustic_part = ((np.sum(fd*self.Amult)**2/np.sum(self.Amult*self.Avec))*(1/(2*self.mp))*((omega/self.cLA)**2/self.cLA**2)*
                        formfactorsquared*
                        self.etav((omega/self.cLA)/(2*self.mX) + omega/(omega/self.cLA)))
    else:
        acoustic_part = 0

    #### Optical part is only included if we have two atoms in the unit cell;
    ####   otherwise, we don't have a simple analytic expression

    def dR_domega_dq_optical(q):

        # This function only works if we have two atoms in the unit cell, so let's check the two
        #   cases where that holds and assign the couplings appropriately.
        if( list(self.Amult) == [2]):
            Aoptical = np.array([self.Avec[0], self.Avec[0]])
            if dark_photon:
                fd = np.array([self.fd_darkphoton[0](q) for i in range(2)])*sqrt(self.debye_waller(q))
            else:
                fd = Aoptical*sqrt(self.debye_waller(q))
        elif( list(self.Amult) == [1,1]):
            Aoptical = self.Avec
            if dark_photon:
                fd = np.array([self.fd_darkphoton[i](q) for i in range(2)])*sqrt(self.debye_waller(q))
            else:
                fd = self.Avec*sqrt(self.debye_waller(q))
        else:
            return 0

        if q > self.qBZ:
            return 0
        else:
            pass

        formfactorsquared = self.Fmed_nucleus(q)**2

        width = 0.5e-3 # giving the delta functions finite width of 0.5 meV

        optical_factor1 = ((self.lattice_spacing/self.eVtoA0)**2/(2*self.LOvec[0]*self.mp))
        optical_factor2 = (fd[0]**2*Aoptical[1]/Aoptical[0] + fd[1]**2*Aoptical[0]/Aoptical[1] - 2*fd[0]*fd[1] +
                                fd[0]*fd[1]*q**2/16)/(np.sum(Aoptical))
        optical_part = q**3*optical_factor1*optical_factor2*(1/(width*sqrt(2*pi)))*exp(-(1/2)*(omega - self.LOvec[0])**2/(width)**2)
        return formfactorsquared*self.etav(q/(2*self.mX) + omega/q)*(optical_part)

    return self._R_multiphonons_prefactor(sigman)*(integrate.quad(lambda q: dR_domega_dq_optical(q),
                    qmin, qmax)[0] + acoustic_part)



def R_single_phonon(self, threshold, sigman=1e-38, dark_photon=False):
    return  (self._R_single_optical(threshold,sigman,dark_photon) + \
            self._R_single_acoustic(threshold,sigman,dark_photon) )


def _R_single_optical(self, threshold, sigman=1e-38, dark_photon=False):

    ###############################
    # Optical part (only applies if two atoms in the unit cell)

    if not hasattr(self, 'LOvec'):
        return 0

    # Returns 0 if LOvec not specified

    if(dark_photon): # check if effective charges are loaded
        assert self.fd_loaded, "Error: effective charge not loaded. Cannot perform calculation for dark photon mediator."

    if (self.LOvec[0] < threshold) or (self.mX*self.vmax**2/2 < self.LOvec[0]):
        return 0

    qmin = self.qmin(self.LOvec[0])
    qmax = min(self.qBZ, self.qmax(self.LOvec[0]))

    if qmin > qmax:
        return 0

    npoints = 100
    qrange = np.linspace(qmin, qmax, npoints)

    if( list(self.Amult) == [2]):
        Aoptical = np.array([self.Avec[0], self.Avec[0]])
        if dark_photon:
            fd = np.array([self.fd_darkphoton[0](qrange) for i in range(2)])*sqrt(self.debye_waller(qrange)).T
        else:
            fd = np.tile(np.array([Aoptical]),(npoints, 1)).T*sqrt(self.debye_waller(qrange)).T
    elif( list(self.Amult) == [1,1]):
        Aoptical = self.Avec
        if dark_photon:
            fd = np.array([self.fd_darkphoton[i](qrange) for i in range(2)])*sqrt(self.debye_waller(qrange)).T
        else:
            fd = np.tile(np.array([self.Avec]),(npoints, 1)).T*sqrt(self.debye_waller(qrange)).T
    else:
        return 0

    dR_dq_optical = np.zeros(npoints)

    formfactorsquared = self.Fmed_nucleus(qrange)**2


    optical_factor1 = ((self.lattice_spacing/self.eVtoA0)**2/(2*self.LOvec[0]*self.mp))
    optical_factor2 = (fd[0]**2*Aoptical[1]/Aoptical[0] + fd[1]**2*Aoptical[0]/Aoptical[1] - 2*fd[0]*fd[1] +
                                    fd[0]*fd[1]*qrange**2/16)/(np.sum(Aoptical))

    optical_part = qrange**3*optical_factor1*optical_factor2

    velocity_part = self.etav(qrange/(2*self.mX) + self.LOvec[0]/qrange)

    dR_dq_optical = optical_part*velocity_part*formfactorsquared

    optical_rate = np.trapz(dR_dq_optical, qrange)

    return self._R_multiphonons_prefactor(sigman)*optical_rate


def _R_single_acoustic(self, threshold, sigman=1e-38, dark_photon=False):
    ###############################
    # Acoustic part

    omegamin = threshold
    omegamax = min(2*self.mX*self.cLA*(self.vmax - self.cLA), self.cLA*self.qBZ, self.mX*self.vmax**2/2, self.omega1ph_max)

    if omegamax < omegamin:
        return 0

    npoints = 100
    omegarange = np.linspace(omegamin, omegamax, npoints)
    dR_domega_acoustic = np.zeros(npoints)

    formfactorsquared = self.Fmed_nucleus(omegarange/self.cLA)**2

    # !TL - atom multiplicity added in fd definitions below
    if dark_photon:
        fd = np.array([self.fd_darkphoton[i](omegarange/self.cLA)*self.Amult[i] for i in range(len(self.atoms))])*sqrt(self.debye_waller(omegarange/self.cLA)).T
    else:
        fd = np.tile(np.array([self.Avec*self.Amult]),(npoints, 1)).T*sqrt(self.debye_waller(omegarange/self.cLA)).T

    dR_domega_acoustic = (np.sum(fd,axis=0))**2*((1/(self.mp*np.sum(self.Avec*self.Amult)))*((omegarange/self.cLA)**2/self.cLA**2)*
                formfactorsquared*self.etav((omegarange/self.cLA)/(2*self.mX)
                                            + omegarange/(omegarange/self.cLA)))

    acoustic_rate = np.trapz(dR_domega_acoustic, omegarange)

    return self._R_multiphonons_prefactor(sigman)*acoustic_rate

###############################################################################################
# Loading in dark photon fd
#

def load_fd_darkphoton(self,datadir,filename):

    # Loads momentum dependent effective charges

    fd_paths = [datadir + self.target + '/' + fi for fi in filename]
    self.fd_darkphoton = []

    for file in fd_paths:
        if not os.path.exists(file):
            print(f"Warning, {file} does not exist! Must load in effective charges for each atom if dark photon calculations are desired.")
            self.fd_loaded=False
        else:
            self.fd_loaded=True
            fd = np.loadtxt(file).T
            (self.fd_darkphoton).append( interp1d(fd[0],fd[1],kind='linear', fill_value = (fd[1][0], fd[1][-1]),bounds_error=False) )

    self.fd_loaded=True
    return
