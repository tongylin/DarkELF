import numpy as np
from numpy import linspace, sqrt, array, pi, cos, sin, dot, exp, sinh, log, log10, cosh, sinh
from scipy.special import erf, erfc, gamma, gammaincc, exp1
from scipy.interpolate import interp1d, interp2d
from scipy import integrate
import sys, os, glob

############################################################################################
#  Cross section reach for kg-yr exposure for for multi-phonon excitations,
#  obtained by integrating the structure factor with approximations as in (paper)
#  density of states is obtained from phonon_filename,
#  fn must be calculated from density of states
#  important parameters from yaml files


# These are basically the only important functions, everything else is just used to calculate these


def R_multiphonons(self, threshold, sigman=1e-38, dark_photon=False):
    """
    Returns rate for DM off harmonic lattice, including multiphonons

    Inputs
    ------
    threshold: float in [eV]
    sigma_n: float
        DM-nucleon cross section in [cm^2], defined at reference momentum of q0.
        DM-nucleus cross section assumed to be coherently enhanced by A^2 by default (if dark photon flag not set)
    dark_photon: Bool to set f_d(q) = Z_d(q) atomic charges

    Outputs
    -------
    rate as function of En, in [1/kg/yr/eV]
    """
    '''Full rate in events/kg/yr, dm-nucleon cross-section 1e-38 [cm^2]'''

    prefactor = ((1/(self.A*self.mp + self.A*self.mp))*
                (self.rhoX*self.eVcm**3)/(2*self.mX*(self.muxnucleon)**2))

    if threshold > (1/2)*self.mX*(self.vmax)**2:
        return 0
    else:

        # !TL Can replace with R_multiphonons_no_single

        omegarange = np.logspace(np.log10(threshold), np.log10((1/2)*self.mX*(self.vmax**2)), 500)
        # Sometimes the number of points matters, increase if noise, decrease if too slow

        dr_domega = [self.dR_domega_multiphonons_no_single(omega, sigman=sigman, dark_photon=dark_photon) for omega in omegarange]

        return (np.trapz(dr_domega, omegarange) + self.R_single_phonon(threshold, sigman=sigman, dark_photon=dark_photon))


def sigma_multiphonons(self, threshold, dark_photon=False):
    '''DM-proton cross-section [cm^2] corresponding to 3 events/kg/yr '''
    rate = self.R_multiphonons(threshold, dark_photon=dark_photon)
    if rate != 0:
        return (3*1e-38)/rate
    else:
        return float('inf')


# !TL: Simplify this function and include it as internal function _dRdomega_multiphonons
def dRdomega_multiphonons(self, omega, sigman=1e-38, dark_photon=False):
    """
    Returns dR_domega in events/kg/yr/eV, this should be used for plotting ONLY since it includes single phonon as a narrow gaussian
    Note: don't integrate over this for the total rate, since the single-phonon coherent rate is modeled by a very sharp gaussian

    Inputs
    ------
    omega: float in [eV]
    sigma_n: float
        DM-nucleon cross section in [cm^2], defined at reference momentum of q0.
        DM-nucleus cross section assumed to be coherently enhanced by A^2 by default (if dark photon flag not set)
    dark_photon: Bool to set f_d(q) = Z_d(q) atomic charges

    Output
    ------
    dR_domega in events/kg/yr/eV
    """
    #!TL: The prefactors can be included all inside the separate dRdomegas, and this function can be just simplified to be basically
    #       dR_domega_multiphonons_no_single + _dR_domega_coherent_single
    #!TL: Can also define some internal prefactor function with all prefactors/units, _R_multiphonons_prefactor
    prefactor = sigman*((1/(self.A*self.mp + self.A*self.mp))*
                (self.rhoX*self.eVcm**3)/(2*self.mX*(self.muxnucleon)**2))
    total_dR_domega = prefactor*(self.dR_domega_multiphonon_expansion(omega, sigman=sigman, dark_photon=dark_photon)
                + self.dR_domega_impulse_approx(omega, sigman=sigman, dark_photon=dark_photon)
                + self.dR_domega_coherent_single(omega, sigman=sigman, dark_photon=dark_photon))
    return ((1/self.eVcm)**2)*(self.eVtoInvYr/self.eVtokg)*total_dR_domega

def dR_domega_multiphonons_no_single(self, omega, sigman=1e-38, dark_photon=False):
    '''dR_domega single-phonon coherent removed'''
    # (useful just for intermediate calcs since single-ph coherent integrated analytically)
    prefactor = sigman*((1/(self.A*self.mp + self.A*self.mp))*
                (self.rhoX*self.eVcm**3)/(2*self.mX*(self.muxnucleon)**2))
    total_dR_domega = prefactor*(self.dR_domega_multiphonon_expansion(omega, sigman=sigman, dark_photon=dark_photon)
                + self.dR_domega_impulse_approx(omega, sigman=sigman, dark_photon=dark_photon))
    return ((1/self.eVcm)**2)*(self.eVtoInvYr/self.eVtokg)*total_dR_domega


def R_multiphonons_no_single(self, threshold, sigman=1e-38, dark_photon=False):
    '''Full rate in events/kg/yr, dm-nucleon cross-section 1e-38 [cm^2]'''

    # !TL: prefactor not used here
    prefactor = ((1/(self.A*self.mp + self.A*self.mp))*
                (self.rhoX*self.eVcm**3)/(2*self.mX*(self.muxnucleon)**2))
    if threshold > (1/2)*self.mX*(self.vmax)**2:
        return 0
    else:
        omegarange = np.logspace(np.log10(threshold), np.log10((1/2)*self.mX*(self.vmax)**2), 500)
            # integrates trapezoidally over this logspace since any sharp peaks are at small omega
            # can make a number of points higher in case concern of very sharp optical peaks
            # !EV need to check this again, some small amount of numerical noise comes from this choice of range

        dr_domega = [self.dR_domega_multiphonons_no_single(omega, sigman=sigman, dark_photon=dark_photon) for omega in omegarange]
        return np.trapz(dr_domega, omegarange)

###############################################################################################
# Loading in dark photon fd
#
#
#

def load_fd_darkphoton(self,datadir,filename):

    fd_path = datadir + self.target+'/'+ filename

    if( not os.path.exists(fd_path)):
        # !TL: warning/text out of date
        print("Warning! Form factor not loaded. Need to set form_factor_filename if needed. Otherwise defaults to massive mediator ")
        self.fd_loaded=False
    else:
        self.fd_loaded=True
        self.fd_data = np.loadtxt(fd_path).T
        print("Loaded " + filename + " for form factor")
        self.fd_darkphoton = interp1d(self.fd_data[0],self.fd_data[1],kind='linear',
                                fill_value=(self.fd_data[0][0], self.fd_data[0][-1]),bounds_error=False)
        self.fd_range = [ self.fd_data[0][0], self.fd_data[0][-1] ]

    return


###############################################################################################
# Auxiliary functions
##########################
#
#
#
# Multiphonon_expansion term

def dR_domega_dq_multiphonon_expansion(self, q, omega, dark_photon=False):


    if ((q < self.qBZ) and (omega < self.dos_omega_range[1])) or (q > 2*sqrt(2*self.A*self.mp*self.omega_bar)):
        return 0
    else:
        pass

    if dark_photon:
        if self.fd_loaded:
            fd = self.fd_darkphoton(q)
        else:
            fd = 0
    else:
        fd = self.A

    #!TL: replace with Fmed_nucleus ?
    formfactorsquared = self.form_factor(q)**2

    otherpart = 0

    for n in range(1, len(self.phonon_Fn)):
        # Debye-Waller factor set to 1 when q^2 small relative to characteristic q, for numerical convenience
        if self.one_over_q2_char*q**2 < 0.03:
            qpart = q**(2*n + 1)
        else:
            qpart = q**(2*n + 1)*exp(-self.one_over_q2_char*q**2)

        otherpart += (1/(2*self.A*self.mp))**n*qpart*self.Fn_interpolations[n](omega)

    return (fd**2 + fd**2)*formfactorsquared*otherpart*self.etav((q/(2*self.mX)) + omega/q)


# !TL: move the sigman/etc prefactors into the individual dR functions.
def dR_domega_multiphonon_expansion(self, omega, sigman=sigman, dark_photon=False):

    if self.vmax**2 < 2*omega/self.mX:
        return 0

    if (omega > self.dos_omega_range[1]):
        qmin = self.mX*(self.vmax - sqrt(self.vmax**2 - (2*omega/self.mX)))
    else:
        qmin = max(self.mX*(self.vmax - sqrt(self.vmax**2 - (2*omega/self.mX))), self.qBZ)

    qmax = min(self.mX*(self.vmax + sqrt(self.vmax**2 - (2*omega/self.mX))), 2*sqrt(2*self.A*self.mp*self.omega_bar))

    if qmin >= qmax:
        return 0
    
    qrange = np.linspace(qmin, qmax, 100)
    
    dR_domega_dq = [self.dR_domega_dq_multiphonon_expansion(q, omega, dark_photon=dark_photon) for q in qrange]
    
    return np.trapz(dR_domega_dq, qrange)

# Need check numerics for multiphonons at q < self.qBZ, I have it currently set to integrate over omega > end of interpolation range
# (rather than omega_LO), should maybe change to omega_LO or cLA*self.qBZ
# a bit noisy in small omega..
# TL: This seems OK to me.

############################################################################################
#
# Impulse approximation term

# !TL: Is it possible to just combine this with  dR_domega_dq_multiphonon_expansion into one function?
#      Similarly, we can integrate that to combine into dR_domega_multiphonons_no_single
#       dR_domega_multiphonon_expansion -> not needed?
#       dR_domega_impulse_approx -> not needed?
#     could this all be one function?
def dR_domega_dq_impulse_approx(self, q, omega, dark_photon=False):
    if q < 2*sqrt(2*self.A*self.mp*self.omega_bar):
        return 0

    if dark_photon:
        if self.fd_loaded:
            fd = self.fd_darkphoton(q)
        else:
            fd = 0
    else:
        fd = self.A

    formfactorsquared = self.form_factor(q)**2

    deltaq = sqrt(q**2*self.one_over_q2_char)

    structurefactor = q*(1/(deltaq*sqrt(2*pi)))*exp(-(omega - q**2/(2*self.A*self.mp))**2/(2*deltaq**2))

    return (fd**2 + fd**2)*formfactorsquared*structurefactor*self.etav((q/(2*self.mX)) + omega/q)

def dR_domega_impulse_approx(self, omega, dark_photon=False):

    if self.vmax**2 < 2*omega/self.mX:
        return 0

    qmin = max(self.mX*(self.vmax - sqrt(self.vmax**2 - (2*omega/self.mX))), 2*sqrt(2*self.A*self.mp*self.omega_bar))
    qmax = self.mX*(self.vmax + sqrt(self.vmax**2 - (2*omega/self.mX)))
    if qmin >= qmax:
        return 0
    qrange = np.linspace(qmin, qmax, 100)
    dR_domega_dq = [self.dR_domega_dq_impulse_approx(q, omega, dark_photon=dark_photon) for q in qrange]
    return np.trapz(dR_domega_dq, qrange)


############################################################################################
#
# Single phonon coherent term

# !TL: functions wanted:  
#      _dR_domega_coherent_single -- internal function used for plotting
#      R_coherent_single_phonon -- total rate, obtained analytically

# !TL: can this be merged in dR_domega_coherent_single?
def dR_domega_dq_coherent_single(self, q, omega, dark_photon=False):
    '''no acoustic here, since it's a delta in q, omega'''

    if q > self.qBZ:
        return 0
    else:
        pass

    if dark_photon:
        if self.fd_loaded:
            formfactorsquared = self.fd_darkphoton(q)**2
        else:
            # print('Form factor not loaded, load with form_factor_filename, defaulted to massive mediator')
            formfactorsquared = 1
    else:
        if mediator == 'massive':
            formfactorsquared = 1
        else:
            formfactorsquared = (self.mX*self.v0/q)**4

    if dark_photon:
        # removing the mass number pre-factor if using custom form factor
        prefactor = 1/(self.A**2 + self.A**2)
    else:
        prefactor = 1

    width = 0.5e-3 # giving the delta functions finite width of 0.5 meV
    x = (1/(2*self.mp*self.A))*self.omega_inverse_bar
    if (x*q**2 < 0.03):
        debye_waller = 1
    else:
        debye_waller = exp(-x*q**2)

    #acoustic_part = ((self.A+self.A)/(2*self.mp))*(q**2/self.cLA)*(1/(width*sqrt(2*pi)))*exp(-(1/2)*(omega - self.cLA*q)**2/(width)**2)
    optical_factor1 = ((self.lattice_spacing/self.eVtoA0)**2/(32*self.LOvec[0]*self.mp))
    optical_factor2 = (self.A*self.A)/(self.A + self.A)
    optical_part = q**5*optical_factor1*optical_factor2*(1/(width*sqrt(2*pi)))*exp(-(1/2)*(omega - self.LOvec[0])**2/(width)**2)
    return prefactor*formfactorsquared*self.etav(q/(2*self.mX) + omega/q)*debye_waller*(optical_part)

def dR_domega_coherent_single(self, omega, dark_photon=False):

    # following stuff is doing the acoustic part analytically
    if dark_photon:
        if self.fd_loaded:
            formfactorsquared = self.fd_darkphoton(omega/self.cLA)**2
        else:
            # print('Form factor not loaded, load with form_factor_filename, defaulted to massive mediator')
            formfactorsquared = 1
    else:
        if mediator == 'massive':
            formfactorsquared = 1
        else:
            formfactorsquared = (self.mX*self.v0/(omega/self.cLA))**4

    if dark_photon:
        # removing the mass number pre-factor if using custom form factor
        prefactor = 1/(self.A**2 + self.A**2)
    else:
        prefactor = 1


    x = (1/(2*self.mp*self.A))*self.omega_inverse_bar
    if (x*(omega/self.cLA)**2 < 0.03):
        debye_waller = 1
    else:
        debye_waller = exp(-x*(omega/self.cLA)**2)

    if self.vmax**2 < 2*omega/self.mX:
        return 0
    if (omega < 2*self.mX*self.cLA*(self.vmax - self.cLA)) and (omega < self.cLA*self.qBZ):
        acoustic_part = (((self.A+self.A)/(2*self.mp))*((omega/self.cLA)**2/self.cLA**2)*
                        formfactorsquared*debye_waller*self.etav((omega/self.cLA)/(2*self.mX) + omega/(omega/self.cLA)))
    else:
        acoustic_part = 0

    qmin = self.mX*(self.vmax - sqrt(self.vmax**2 - (2*omega/self.mX)))
    qmax = min(self.mX*(self.vmax + sqrt(self.vmax**2 - (2*omega/self.mX))), self.qBZ)

    return integrate.quad(lambda q: self.dR_domega_dq_coherent_single(q, omega, mediator=mediator, dark_photon=dark_photon),
                    qmin, qmax)[0] + acoustic_part*prefactor


def R_single_phonon(self, threshold, sigman=1e-38, dark_photon=False):

    ###############################
    # Optical part

    if (self.LOvec[0] < threshold) or (self.mX*self.vmax**2/2 < self.LOvec[0]):
        optical_rate = 0
    else:

        qmin = self.mX*(self.vmax - sqrt(self.vmax**2 - 2*self.LOvec[0]/self.mX))
        qmax = min(self.qBZ, self.mX*(self.vmax + sqrt(self.vmax**2 - 2*self.LOvec[0]/self.mX)))

        if qmin > qmax:
            optical_rate = 0

        else:

            npoints = 100
            qrange = np.linspace(qmin, qmax, npoints)

            dR_dq_optical = np.zeros(npoints)

            optical_factor1 = ((self.lattice_spacing/self.eVtoA0)**2/(32*self.LOvec[0]*self.mp))
            optical_factor2 = 1/(2*(self.A + self.A))

            #!TL: could be vectorized rather than loop?
            for i, q in enumerate(qrange):

                formfactorsquared = self.form_factor(q)**2

                if dark_photon:
                    if self.fd_loaded:
                        fd = self.fd_darkphoton(q)
                    else:
                    # print('Form factor not loaded, load with form_factor_filename, defaulted to massive mediator')
                        fd = 0
                else:
                    fd = self.A

                #!TL define some debye-waller function?
                if self.one_over_q2_char*q**2 < 0.03:
                    debye_waller = 1
                else:
                    debye_waller = exp(-self.one_over_q2_char*q**2)

                optical_part = q**5*optical_factor1*optical_factor2

                velocity_part = self.etav(q/(2*self.mX) + self.LOvec[0]/q)

                dR_dq_optical[i] = (fd**2 + fd**2)*optical_part*velocity_part*debye_waller*formfactorsquared

            optical_rate = np.trapz(dR_dq_optical, qrange)

    ###############################
    # Acoustic part

    omegamin = threshold
    omegamax = min(2*self.mX*self.cLA*(self.vmax - self.cLA), self.cLA*self.qBZ, self.mX*self.vmax**2/2)

    if omegamax < omegamin:
        acoustic_rate = 0
    else:
        npoints = 100
        omegarange = np.linspace(omegamin, omegamax, npoints)
        dR_domega_acoustic = np.zeros(npoints)

        for i, omega in enumerate(omegarange):

            if dark_photon:
                if self.fd_loaded:
                    fd = self.fd_darkphoton(omega/self.cLA)
                else:
                    # print('Form factor not loaded, load with form_factor_filename, defaulted to massive mediator')
                    fd = 0
            else:
                fd = self.A

            formfactorsquared = self.form_factor(omega/self.cLA)**2

            if (self.one_over_q2_char*(omega/self.cLA)**2 < 0.03):
                debye_waller = 1
            else:
                debye_waller = exp(-self.one_over_q2_char*(omega/self.cLA)**2)

            dR_domega_acoustic[i] = (fd**2 + fd**2)*((1/(2*self.mp*self.A))*((omega/self.cLA)**2/self.cLA**2)*
                        formfactorsquared*debye_waller*self.etav((omega/self.cLA)/(2*self.mX) + omega/(omega/self.cLA)))

        acoustic_rate = np.trapz(dR_domega_acoustic, omegarange)

    prefactor = ((1/(self.A*self.mp + self.A*self.mp))*
                (self.rhoX*self.eVcm**3)/(2*self.mX*(self.muxnucleon)**2))

    return prefactor*sigman*((1/self.eVcm)**2)*(self.eVtoInvYr/self.eVtokg)*(optical_rate + acoustic_rate)
