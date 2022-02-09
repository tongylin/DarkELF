import numpy as np
from numpy import linspace, sqrt, array, pi, cos, sin, dot, exp, sinh, log, log10, cosh, sinh
from scipy.special import erf, erfc, gamma, gammaincc, exp1
from scipy.interpolate import interp1d, interp2d
from scipy import integrate


############################################################################################
#  Cross section reach for kg-yr exposure for for multi-phonon excitations,
#  obtained by integrating the structure factor with approximations as in (paper)
#  density of states is obtained from phonon_filename,
#  fn must be calculated from density of states
#  important parameters from yaml files


# These are basically the only important functions, everything else is just used to calculate these


def R_multiphonons(self, mdm, omegathreshold, mediator='massive', sigman=1e-38, custom_form_factor=False):
    '''Full rate in events/kg/yr, dm-nucleon cross-section 1e-38 [cm^2]'''
    if omegathreshold > (1/2)*mdm*(self.vesc + self.veavg)**2:
        return 0
    reduced_mass = mdm*self.mp/(mdm + self.mp)
    prefactor = ((1/(self.A*self.mp + self.A*self.mp))*
                (self.rhoX*self.eVcm**3)/(2*mdm*(reduced_mass)**2))
    if omegathreshold > (1/2)*mdm*(self.vesc + self.veavg)**2:
        return 0
    else:
        if (1/2)*mdm*(self.vesc + self.veavg)**2 < omegathreshold:
            return 0
        omegarange = np.logspace(np.log10(omegathreshold), np.log10((1/2)*mdm*(self.vesc + self.veavg)**2), 250)
            # integrates trapezoidally over this logspace since any sharp peaks are at small omega
            # can make a number of points higher in case concern of very sharp optical peaks

        dr_domega = [self.dR_domega_multiphonons_no_single(mdm, omega, mediator=mediator, custom_form_factor=custom_form_factor) for omega in omegarange]
        return (np.trapz(dr_domega, omegarange) + prefactor*sigman*((1/self.eVcm)**2)*
                (self.eVtoInvYr/self.eVtokg)*self.coherent_single_phonon_rate(mdm, omegathreshold, mediator=mediator, custom_form_factor=custom_form_factor))


def sigma_nucleon(self, mdm, omegathreshold, mediator='massive', custom_form_factor=False):
    '''DM-nucleon cross-section [cm^2] corresponding to 3 events/kg/yr '''
    rate = self.R_multiphonons(mdm, omegathreshold, mediator=mediator, custom_form_factor=custom_form_factor)
    if rate != 0:
        return (3*1e-38)/self.R_multiphonons(mdm, omegathreshold, mediator=mediator, custom_form_factor=custom_form_factor)
    else:
        return float('inf')

def dR_domega_multiphonons(self, mdm, omega, mediator='massive', sigman=1e-38, custom_form_factor=False):
    """
    Returns dR_domega in events/kg/yr/eV

    Inputs
    ------
    mdm: float
        dark matter mass in [eV]
    form_factor: loaded in or choose 'massive' or 'massless'
         default is 'massive'
    Output
    ------
    dR_domega in events/kg/yr/eV
    ------
    Note: don't integrate over this, since the single-phonon coherent rate
    is modeled by a very sharp gaussian
    """
    reduced_mass = mdm*self.mp/(mdm + self.mp)
    prefactor = sigman*((1/(self.A*self.mp + self.A*self.mp))*
                (self.rhoX*self.eVcm**3)/(2*mdm*(reduced_mass)**2))
    total_dR_domega = prefactor*(self.dR_domega_multiphonon_expansion(mdm, omega, mediator=mediator, custom_form_factor=custom_form_factor)
                + self.dR_domega_impulse_approx(mdm, omega, mediator=mediator, custom_form_factor=custom_form_factor)
                + self.dR_domega_coherent_single(mdm, omega, mediator=mediator, custom_form_factor=custom_form_factor))
    return ((1/self.eVcm)**2)*(self.eVtoInvYr/self.eVtokg)*total_dR_domega

def dR_domega_multiphonons_no_single(self, mdm, omega, mediator='massive', sigman=1e-38, custom_form_factor=False):
    '''dR_domega single-phonon coherent removed'''
    # (useful just for intermediate calcs since single-ph coherent integrated analytically)
    reduced_mass = mdm*self.mp/(mdm + self.mp)
    prefactor = sigman*((1/(self.A*self.mp + self.A*self.mp))*
                (self.rhoX*self.eVcm**3)/(2*mdm*(reduced_mass)**2))
    total_dR_domega = prefactor*(self.dR_domega_multiphonon_expansion(mdm, omega, mediator=mediator, custom_form_factor=custom_form_factor)
                + self.dR_domega_impulse_approx(mdm, omega, mediator=mediator, custom_form_factor=custom_form_factor))
    return ((1/self.eVcm)**2)*(self.eVtoInvYr/self.eVtokg)*total_dR_domega


###############################################################################################
# Auxiliary functions
##########################
#
#
#
# Multiphonon_expansion term

def dR_domega_dq_multiphonon_expansion(self, mdm, q, omega, mediator='massive', custom_form_factor=False):

    qBZ = (2*pi/self.lattice_spacing)*self.eVtoA0

    if ((q < qBZ) and (omega < self.dos_omega_range[1])) or (q > 2*sqrt(2*self.A*self.mp*self.omega_bar)):
        return 0
    else:
        pass

    if custom_form_factor:
        if self.form_factor_loaded:
            formfactorsquared = self.form_factor_func(q)**2
        else:
            # print('Form factor not loaded, load with form_factor_filename, defaulted to massive mediator')
            formfactorsquared = 1
    else:
        if mediator == 'massive':
            formfactorsquared = 1
        else:
            formfactorsquared = (mdm*self.v0/q)**4


    x = (1/(2*self.A*self.mp))*self.omega_inverse_bar

    if custom_form_factor:
        prefactor = 1
    else:
        prefactor = (self.A**2 + self.A**2)

    otherpart = 0
    for n in range(1, len(self.phonon_Fn)):
        if x*q**2 < 0.03:
            qpart = q**(2*n + 1)
        else:
            qpart = q**(2*n + 1)*exp(-x*q**2)
        if (omega > self.phonon_Fn[0][-1]) or (omega < self.phonon_Fn[0][0]):
            return 0
        else:
            otherpart += (1/(2*self.A*self.mp))**n*qpart*self.Fn_interpolations[n](omega)

    return prefactor*formfactorsquared*otherpart*self.etav((q/(2*mdm)) + omega/q)

def dR_domega_multiphonon_expansion(self, mdm, omega, mediator='massive', custom_form_factor=False):
    qBZ = (2*pi/self.lattice_spacing)*self.eVtoA0
    vmax = self.vesc + self.veavg
    if vmax**2 < 2*omega/mdm:
        return 0

    if (omega > self.dos_omega_range[1]):
        qmin = mdm*(vmax - sqrt(vmax**2 - (2*omega/mdm)))
    else:
        qmin = max(mdm*(vmax - sqrt(vmax**2 - (2*omega/mdm))), qBZ)
    qmax = min(mdm*(vmax + sqrt(vmax**2 - (2*omega/mdm))), 2*sqrt(2*self.A*self.mp*self.omega_bar))
    if qmin >= qmax:
        return 0
    qrange = np.linspace(qmin, qmax, 100)
    dR_domega_dq = [self.dR_domega_dq_multiphonon_expansion(mdm, q, omega, mediator=mediator, custom_form_factor=custom_form_factor) for q in qrange]
    return np.trapz(dR_domega_dq, qrange)

############################################################################################
#
# Impulse approximation term

def dR_domega_dq_impulse_approx(self, mdm, q, omega, mediator='massive', custom_form_factor=False):
    if q < 2*sqrt(2*self.A*self.mp*self.omega_bar):
        return 0
    else:
        pass

    if custom_form_factor:
        if self.form_factor_loaded:
            formfactorsquared = self.form_factor_func(q)**2
        else:
            # print('Form factor not loaded, load with form_factor_filename, defaulted to massive mediator')
            formfactorsquared = 1
    else:
        if mediator == 'massive':
            formfactorsquared = 1
        else:
            formfactorsquared = (mdm*self.v0/q)**4

    if custom_form_factor:
        prefactor = 1
    else:
        prefactor = (self.A**2 + self.A**2)

    structurefactor = q*(1/(self.deltafunc(q)*sqrt(2*pi)))*exp(-(omega - q**2/(2*self.A*self.mp))**2/(2*self.deltafunc(q)**2))
    return prefactor*formfactorsquared*structurefactor*self.etav((q/(2*mdm)) + omega/q)

def dR_domega_impulse_approx(self, mdm, omega, mediator='massive', custom_form_factor=False):
    vmax = self.vesc + self.veavg
    if vmax**2 < 2*omega/mdm:
        return 0

    qmin = max(mdm*(vmax - sqrt(vmax**2 - (2*omega/mdm))), 2*sqrt(2*self.A*self.mp*self.omega_bar))
    qmax = mdm*(vmax + sqrt(vmax**2 - (2*omega/mdm)))
    if qmin >= qmax:
        return 0
    qrange = np.linspace(qmin, qmax, 100)
    dR_domega_dq = [self.dR_domega_dq_impulse_approx(mdm, q, omega, mediator=mediator, custom_form_factor=custom_form_factor) for q in qrange]
    return np.trapz(dR_domega_dq, qrange)

def deltafunc(self, q):
    # be careful here, there's another variable called delta
    return sqrt((q**2)*(self.omega_bar)/(2*self.A*self.mp))


############################################################################################
#
# Single phonon coherent term

def dR_domega_dq_coherent_single(self, mdm, q, omega, mediator='massive', custom_form_factor=False):
    '''no acoustic here, since it's a delta in q, omega'''
    qBZ = (2*pi/self.lattice_spacing)*self.eVtoA0
    if q > qBZ:
        return 0
    else:
        pass

    if custom_form_factor:
        if self.form_factor_loaded:
            formfactorsquared = self.form_factor_func(q)**2
        else:
            # print('Form factor not loaded, load with form_factor_filename, defaulted to massive mediator')
            formfactorsquared = 1
    else:
        if mediator == 'massive':
            formfactorsquared = 1
        else:
            formfactorsquared = (mdm*self.v0/q)**4

    if custom_form_factor:
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

    vmax = self.vesc + self.veavg

    #acoustic_part = ((self.A+self.A)/(2*self.mp))*(q**2/self.cLA)*(1/(width*sqrt(2*pi)))*exp(-(1/2)*(omega - self.cLA*q)**2/(width)**2)
    optical_factor1 = ((self.lattice_spacing/self.eVtoA0)**2/(32*self.LOvec[0]*self.mp))
    optical_factor2 = (self.A*self.A)/(self.A + self.A)
    optical_part = q**5*optical_factor1*optical_factor2*(1/(width*sqrt(2*pi)))*exp(-(1/2)*(omega - self.LOvec[0])**2/(width)**2)
    return prefactor*formfactorsquared*self.etav(q/(2*mdm) + omega/q)*debye_waller*(optical_part)

def dR_domega_coherent_single(self, mdm, omega, mediator='massive', custom_form_factor=False):

    # following stuff is doing the acoustic part analytically
    if custom_form_factor:
        if self.form_factor_loaded:
            formfactorsquared = self.form_factor_func(omega/self.cLA)**2
        else:
            # print('Form factor not loaded, load with form_factor_filename, defaulted to massive mediator')
            formfactorsquared = 1
    else:
        if mediator == 'massive':
            formfactorsquared = 1
        else:
            formfactorsquared = (mdm*self.v0/(omega/self.cLA))**4

    if custom_form_factor:
        # removing the mass number pre-factor if using custom form factor
        prefactor = 1/(self.A**2 + self.A**2)
    else:
        prefactor = 1


    x = (1/(2*self.mp*self.A))*self.omega_inverse_bar
    if (x*(omega/self.cLA)**2 < 0.03):
        debye_waller = 1
    else:
        debye_waller = exp(-x*(omega/self.cLA)**2)

    qBZ = (2*pi/self.lattice_spacing)*self.eVtoA0

    vmax = self.vesc + self.veavg
    if vmax**2 < 2*omega/mdm:
        return 0
    if (omega < 2*mdm*self.cLA*(vmax - self.cLA)) and (omega < self.cLA*qBZ):
        acoustic_part = (((self.A+self.A)/(2*self.mp))*((omega/self.cLA)**2/self.cLA**2)*
                        formfactorsquared*debye_waller*self.etav((omega/self.cLA)/(2*mdm) + omega/(omega/self.cLA)))
    else:
        acoustic_part = 0
    return integrate.quad(lambda q: self.dR_domega_dq_coherent_single(mdm, q, omega, mediator=mediator, custom_form_factor=custom_form_factor),
                    mdm*(vmax - sqrt(vmax**2 - (2*omega/mdm))), mdm*(vmax + sqrt(vmax**2 - (2*omega/mdm))))[0] + acoustic_part*prefactor

def dR_domega_coherent_acoustic(self, mdm, omega, mediator='massive', custom_form_factor=False):

    # following stuff is doing the acoustic part analytically
    if custom_form_factor:
        if self.form_factor_loaded:
            formfactorsquared = self.form_factor_func(omega/self.cLA)**2
        else:
            # print('Form factor not loaded, load with form_factor_filename, defaulted to massive mediator')
            formfactorsquared = 1
    else:
        if mediator == 'massive':
            formfactorsquared = 1
        else:
            formfactorsquared = (mdm*self.v0/(omega/self.cLA))**4

    if custom_form_factor:
        # removing the mass number pre-factor if using custom form factor
        prefactor = 1/(self.A**2 + self.A**2)
    else:
        prefactor = 1

    x = (1/(2*self.mp*self.A))*self.omega_inverse_bar
    if (x*(omega/self.cLA)**2 < 0.03):
        debye_waller = 1
    else:
        debye_waller = exp(-x*(omega/self.cLA)**2)

    qBZ = (2*pi/self.lattice_spacing)*self.eVtoA0

    vmax = self.vesc + self.veavg
    if vmax**2 < 2*omega/mdm:
        return 0
    if (omega < 2*mdm*self.cLA*(vmax - self.cLA)) and (omega < self.cLA*qBZ):
        acoustic_part = (((self.A+self.A)/(2*self.mp))*((omega/self.cLA)**2/self.cLA**2)*
                        formfactorsquared*debye_waller*self.etav((omega/self.cLA)/(2*mdm) + omega/(omega/self.cLA)))
    else:
        acoustic_part = 0

    return prefactor*acoustic_part

def R_coherent_acoustic(self, mdm, omegathreshold, mediator='massive', custom_form_factor=False):

    qBZ = (2*pi/self.lattice_spacing)*self.eVtoA0

    vmax = self.vesc + self.veavg
    omegamin = omegathreshold
    omegamax = min(2*mdm*self.cLA*(vmax - self.cLA), self.cLA*qBZ, mdm*vmax**2/2)
    if omegamax < omegamin:
        return 0
    else:
        pass
    return integrate.quad(lambda omega: self.dR_domega_coherent_acoustic(mdm, omega, mediator=mediator, custom_form_factor=custom_form_factor),
                        omegamin, omegamax)[0]

def dR_dq_coherent_optical(self, mdm, q, mediator='massive', custom_form_factor=False):

    # following stuff is doing the acoustic part analytically
    if custom_form_factor:
        if self.form_factor_loaded:
            formfactorsquared = self.form_factor_func(q)**2
        else:
            # print('Form factor not loaded, load with form_factor_filename, defaulted to massive mediator')
            formfactorsquared = 1
    else:
        if mediator == 'massive':
            formfactorsquared = 1
        else:
            formfactorsquared = (mdm*self.v0/q)**4

    if custom_form_factor:
        # removing the mass number pre-factor if using custom form factor
        prefactor = 1/(self.A**2 + self.A**2)
    else:
        prefactor = 1

    x = (1/(2*self.mp*self.A))*self.omega_inverse_bar
    if (x*q**2 < 0.03):
        debye_waller = 1
    else:
        debye_waller = exp(-x*q**2)

    qBZ = (2*pi/self.lattice_spacing)*self.eVtoA0

    vmax = self.vesc + self.veavg

    if vmax**2 < 2*self.LOvec[0]/mdm:
        return 0

    qmin = mdm*(vmax - sqrt(vmax**2 - 2*self.LOvec[0]/mdm))
    qmax = min(qBZ, mdm*(vmax + sqrt(vmax**2 - 2*self.LOvec[0]/mdm)))


    if qmax > qmin:
        return 0

    optical_factor1 = ((self.lattice_spacing/self.eVtoA0)**2/(32*self.LOvec[0]*self.mp))
    optical_factor2 = (self.A*self.A)/(self.A + self.A)

    if qmin < q < qmax:
        optical_part = q**5*optical_factor1*optical_factor2

    return prefactor*optical_part*etav(q/(2*mdm) + self.LOvec[0]/q)*debye_waller*formfactorsquared

def R_coherent_optical(self, mdm, omegathreshold, mediator='massive', custom_form_factor=False):
    if omegathreshold < self.LOvec[0]:
        pass
    else:
        return 0

    qBZ = (2*pi/self.lattice_spacing)*self.eVtoA0
    vmax = self.vesc + self.veavg

    if vmax**2 < 2*self.LOvec[0]/mdm:
        return 0
    else:
        pass

    qmin = mdm*(vmax - sqrt(vmax**2 - 2*self.LOvec[0]/mdm))
    qmax = min(qBZ, mdm*(vmax + sqrt(vmax**2 - 2*self.LOvec[0]/mdm)))

    return integrate.quad(lambda q: self.dR_dq_coherent_optical(mdm, q, mediator=mediator, custom_form_factor=custom_form_factor),
                        qmin, qmax)[0]

def coherent_single_phonon_rate(self, mdm, omegathreshold, mediator='massive', custom_form_factor=False):
    return (self.R_coherent_optical(mdm, omegathreshold, mediator=mediator, custom_form_factor=custom_form_factor) +
            self.R_coherent_acoustic(mdm, omegathreshold, mediator=mediator, custom_form_factor=custom_form_factor))
