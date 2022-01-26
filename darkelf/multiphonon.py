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


# This is basically the only important function, everything else is just used to calculate this one

def sigma_nucleon(self, mdm, omegathreshold, mediator='massive'):
    """
    Returns reach in sigma_nucleon for 3 events/kg/yr

    Inputs
    ------
    mdm: float
        dark matter mass in [eV]
    omegathreshold: float
        experimental energy threshold in [eV]
    mediator: string
        'massive' or 'massless', default is 'massive'
    Output
    ------
    sigma_nucleon in [cm^2]
    """
    reduced_mass = mdm*self.mp/(mdm + self.mp)
    prefactor = ((1/(self.A*self.mp + self.A*self.mp))*
                (self.rhoX*self.eVcm**3)/(2*mdm*(reduced_mass)**2))
    total_rate = prefactor*(self.rate_integrated(mdm, omegathreshold, mediator)
                + self.impulse_rate(mdm, omegathreshold, mediator)
                + self.coherent_single_phonon_rate(mdm, omegathreshold, mediator))
    if total_rate != 0:
        return ((3*self.eVtokg/self.eVtoInvYr)/total_rate)*self.eVcm**2
    else:
        return float('inf')

############################################################################################
#
# Auxiliary functions to call

def multiphononintegrand(self, mdm, v, omegathreshold, mediator='massive'):
    '''This is the multiphonon integrand,
    momentum range restricted to between 1 Brillouin zone and 2*sqrt(2 mN omegabar),
    the momentum here has been integrated out analytically

    inputs:
    mdm, float in [eV]
    v, float in units of c
    mediator, string either "massive" or "massless"
    output:
    array in omega to be integrated over

    IMPORTANT NOTE: The Fn grid must be fairly fine because these numerical calculations
                    use sampled integrations.
    '''
    qBZ = (2*pi/(self.lattice_spacing))*self.eVtoA0
    x = (1/(2*self.A*self.mp))*self.omega_inverse_bar
    velocitypart = self._fv_1d_scalar(v)/v
    otherpart = np.zeros(len(self.phonon_Fn[0]))
    for i, omega in enumerate(self.phonon_Fn[0]):
        if (v**2 > 2*omega/mdm) and (omega > omegathreshold):
            qminus = max(mdm*(v - sqrt(v**2 - 2*omega/mdm)), qBZ)
            qplus = min(mdm*(v + sqrt(v**2 - 2*omega/mdm)), 2*sqrt(2*self.A*self.mp*self.omega_bar))
            if qplus > qminus:
                for n in range(1, len(self.phonon_Fn)):
                    qdefinite = self.integratedqpart(qplus, x, n, mediator) - self.integratedqpart(qminus, x, n, mediator)
                    otherpart[i] += (1/(2*self.A*self.mp))**n*qdefinite*self.phonon_Fn[n][i]
            else: otherpart[i] = 0
        else:
            otherpart[i] = 0
    prefactor = (self.A**2 + self.A**2)
    if mediator == 'massive':
        factor = 1
    else:
        factor = (mdm*self.v0)**4
    return prefactor*factor*velocitypart*otherpart

def integratedqpart(self, q, x, n, mediator='massive'):
    '''Analytically integrated momentum (indefinite integral)'''
    if mediator == 'massive':
        return -(1/2)*x**(-(1+n))*gamma(1 + n)*gammaincc(1 + n, x*q**2)
    else:
        if n == 1:
            return -(1/2)*exp1(x*q**2)
        else:
            return -(1/2)*x**(-(n-1))*gamma(n - 1)*gammaincc(n - 1, x*q**2)

def rate_omega_integrated(self, mdm, v, omegathreshold, mediator='massive'):
    '''Multiphonon integrand with momentum and energy integrated out divided by sigma_nucleon'''
    integrated = np.trapz(self.multiphononintegrand(mdm, v, omegathreshold, mediator),
                            x=self.phonon_Fn[0])
    return integrated

def rate_integrated(self, mdm, omegathreshold, mediator='massive'):
    ''''Multiphonon integrated rate divided by sigma_nucleon'''

    vrange = np.linspace(np.sqrt(2*omegathreshold/mdm), self.vesc + self.veavg, 30)
    omega_int = np.zeros(30)

    for i, v in enumerate(vrange):
        omega_int[i] = self.rate_omega_integrated(mdm, v, omegathreshold, mediator)

    integrated = np.trapz(omega_int, vrange)

    return integrated

############################################################################################
#
# Impulse approximation

def impulse_rate(self, mdm, omegathreshold, mediator='massive'):
    '''integrates out q, v'''
    vmax = self.vesc + self.veavg
    return integrate.dblquad(lambda q, v: self.integrandimpulse(mdm, v, q, omegathreshold, mediator),
                sqrt(2*omegathreshold/mdm), vmax,
                lambda v: max(mdm*(v - sqrt(v**2 - 2*omegathreshold/mdm)),
                 2*sqrt(2*self.A*self.mp*self.omega_bar)) if v**2>2*omegathreshold/mdm else 0,
                lambda v: mdm*(v + sqrt(v**2 - 2*omegathreshold/mdm)) if v**2>2*omegathreshold/mdm else 0)[0]


# Functions to call (intermediate steps)

def deltafunc(self, q):
    # be careful here, there's another variable called delta
    return sqrt((q**2)*(self.omega_bar)/(2*self.A*self.mp))

def indefiniteintegratedomega(self, q, omega):
    '''Impulse approximation omega integrated out'''
    return -(1/2)*erf((q**2 - 2*self.A*self.mp*omega)/(2*sqrt(2)*self.A*self.mp*self.deltafunc(q)))

def definiteintegratedomega(self, mdm, v, q, omegathreshold):
    upperlimit = q*v - (q**2/(2*mdm))
    if upperlimit > omegathreshold:
        lowerlimit = omegathreshold
        return self.indefiniteintegratedomega(q, upperlimit) - self.indefiniteintegratedomega(q, lowerlimit)
    else:
        return 0

def integrandimpulse(self, mdm, v, q, omegathreshold, mediator='massive'):
    if v**2 - 2*omegathreshold/mdm > 0:
        qminus = max(mdm*(v - sqrt(v**2 - 2*omegathreshold/mdm)), 2*sqrt(2*self.A*self.mp*self.omega_bar))
        qplus = mdm*(v + sqrt(v**2 - 2*omegathreshold/mdm))
        if qminus < q < qplus:
            if mediator == 'massive':
                formfactor = 1
            else:
                formfactor = (mdm*self.v0/q)**4
            prefactor = q*(self.A**2 + self.A**2)
            return prefactor*formfactor*self.definiteintegratedomega(mdm, v, q, omegathreshold)*self._fv_1d_scalar(v)/v
        else:
            return 0
    else:
        return 0


############################################################################################
#
# Single phonon coherent

def coherent_single_phonon_rate(self, mdm, omegathreshold, mediator='massive'):
    '''integrates out v'''
    vmax = self.vesc + self.veavg
    return integrate.quad(lambda v: self.acoustic_integrand(mdm, v, omegathreshold, mediator) +
                                    self.optical_integrand(mdm, v, omegathreshold, mediator),
                                    sqrt(2*omegathreshold/mdm), vmax)[0]

def definite_integrand_acoustic(self, omega, x, mediator):
    if mediator == 'massive':
        return (1/(2*self.cLA**4))*(sqrt(pi)*erf(sqrt(x)*omega)/(4*x**(3/2))
                                        - (exp(-x*omega**2)*omega)/(2*x))
    else:
        return (1/2)*(-sqrt(pi*x)*erf(sqrt(x)*omega)
                                        - exp(-x*omega**2)/omega)


def acoustic_integrand(self, mdm, v, omegathreshold, mediator='massive'):
    qBZ = (2*pi/self.lattice_spacing)*self.eVtoA0
    omegaminus = omegathreshold
    omegaplus = min(mdm*v**2/2, self.cLA*qBZ, 2*mdm*self.cLA*(v-self.cLA))
    x = (1/(2*self.mp*self.A*self.cLA**2))*self.omega_inverse_bar
    factor = (self.A + self.A)/self.mp
    if mediator == 'massive':
        prefactor = 1
    else:
        prefactor = ((mdm*self.v0)**4)

    if omegaplus > omegaminus:
        return prefactor*factor*(self.definite_integrand_acoustic(omegaplus, x, mediator)
                - self.definite_integrand_acoustic(omegaminus, x, mediator))*self._fv_1d_scalar(v)/v
    else:
        return 0

def definite_integrand_optical(self, q, x, mediator):
    part1 = ((self.lattice_spacing/self.eVtoA0)**2/(64*self.LOvec[0]*self.mp))
    part2 = (self.A*self.A)/(self.A + self.A)
    if mediator == 'massive':
        if (x*q**2 < 0.03):
            # in this limit we ignore Debye-Waller for numerical reasons
            part3 = q**6/6
        else:
            part3 = -exp(-x*q**2)*(2 + 2*x*q**2 + x**2*q**4)/x**3
        return part1*part2*part3
    else:
        if (x*q**2 < 0.03):
            # in this limit we ignore Debye-Waller for numerical reasons
            part3 = q**2/2
        else:
            part3 = -exp(-x*q**2)/(2*x)
        return part1*part2*part3

def optical_integrand(self, mdm, v, omegathreshold, mediator='massive'):
    qBZ = (2*pi/self.lattice_spacing)*self.eVtoA0
    if (1/2)*mdm*v**2 > self.LOvec[0]:
        qminus = mdm*(v - sqrt(v**2 - 2*self.LOvec[0]/mdm))
        qplus = min(qBZ, mdm*(v + sqrt(v**2 - 2*self.LOvec[0]/mdm)))
    else:
        return 0

    x = (1/(2*self.mp*self.A))*self.omega_inverse_bar

    if mediator == 'massive':
        prefactor = 1
    else:
        prefactor = (mdm*self.v0)**4

    if (qplus > qminus) and (omegathreshold < self.LOvec[0] < (1/2)*mdm*v**2):
        return prefactor*(self.definite_integrand_optical(qplus, x, mediator)
                - self.definite_integrand_optical(qminus, x, mediator))*self._fv_1d_scalar(v)/v
    else:
        return 0
