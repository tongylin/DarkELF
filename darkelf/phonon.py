import numpy as np
from numpy import linspace, sqrt, array, pi, cos, sin, dot, exp, sinh, log, log10, cosh, sinh
from scipy.interpolate import interp1d, interp2d
from scipy import integrate

############################################################################################
# Rate for phonon excitations, obtained by numerically integrating the loss function
#  epsilon(omega) is obtained from phonon_filename and it is assumed that the k-dependence
#  is negligible when performing the k integral. This rate is therefore only robust for
#  sub-MeV dark matter.

def dRdomegadk_phonon(self,omega,k,sigmae=1e-38):
    """
    Returns double  differential phonon excitation rate dR/domega/dk in 1/kg/yr/eV^2
    It should only be used in massless mediator limit, which is enforced here no matter what mMed is

    Inputs
    ------
    omega: float
        electron excitation energy in [eV]
    k: float
        momentum transfer in [eV]
    sigmae: float
        cross section in [cm^2]
    """
    etav_val = self.etav(self.vmin(omega,k))
    temp_eps1=self.eps1(omega,k,method="phonon")
    temp_eps2=self.eps2(omega,k,method="phonon")
    if(etav_val > 0.0):
        fac =  self.rhoX/self.mX * 1/self.rhoT * self.eVtoInvYr * \
            1/(2*pi)**2 * sigmae/self.eVcm**2/ self.muXe**2 * \
            k**3/(2.0*self.alphaEM)*(self.alphaEM*self.me/k)**4 * temp_eps2/(temp_eps1**2 + temp_eps2**2)
        return fac*etav_val*1000 # 1000 to convert from 1/gram to 1/kilogram
    else:
        return 0.0

# integrate the function above WRT k to get dR/domega in units of 1/kg/year
def dRdomega_phonon(self,omega, sigmae=1e-38):
    """
    Returns differential phonon excitation rate dR/domega in 1/kg/yr/eV
    It should only be used in massless mediator limit, which is enforced here no matter what mMed is

    Inputs
    ------
    omega: float or list
        energy in eV
    sigmae: float
        DM-electron cross section in [cm^2]
    """
    assert self.phonon_ELF_loaded == True, "Phonon ELF data is not loaded!"

    # integrate over k...
    scalar_input = np.isscalar(omega)
    omega = np.atleast_1d(omega)
    dRdomega = np.zeros_like(omega)
    for i in range(len(omega)):
        kmin = self.qmin(omega[i])
        kmax = self.qmax(omega[i])
        if(kmin >= kmax): # accounts for kmin=kmax=0 when omega is too high (kinematics)
            continue
        dRdomega[i] = integrate.quad(lambda x: self.dRdomegadk_phonon(omega[i],x,sigmae,)\
                                     /self.eVtoInvYr, kmin, kmax, limit=50)[0] * \
                    self.eVtoInvYr

    if(scalar_input):
        return dRdomega[0]
    else:
        return dRdomega

def R_phonon(self,threshold=-1.0,sigmae=1e-38):
    """
    Returns phonon excitation rate in 1/kg/yr by numerically integrating Im(-1/eps(omega)),
    where eps(omega) is obtained by interpolating data specified in phonon_filename

    It should only be used in massless mediator limit, which is enforced here no matter what mMed is

    Inputs
    ------
    threshold: float
        energy threshold in eV
    sigmae: float
        DM-electron cross section in [cm^2]
    """
    assert self.phonon_ELF_loaded == True, "Phonon ELF data is not loaded!"

    if (threshold < self.omph_range[0]):
        # Integrate over available energy rate of phonon eps data file
        olist=np.linspace(self.omph_range[0],self.omph_range[1],300)
    elif (threshold > self.omph_range[1]):
        return 0.0
    else:
        # Integrate from threshold up to maximum energy of phonon eps data file
        olist=np.linspace(threshold,self.omph_range[1],200)

    return integrate.trapz(self.dRdomega_phonon(olist,sigmae=sigmae),x=olist)


############################################################################################
# Total rate for Frohlich-like optical phonon interaction, approximation in the k=0 limit

def R_phonon_Frohlich(self,sigmae=1e-38,LOvec=0,TOvec=0):
    """
    Returns phonon excitation rate in 1/kg/yr by using Frohlich analytic approximation
    Only applies for massless mediators, since it assumes approximate k-independence

    Inputs
    ------
    sigmae: float
        DM-electron cross section in [cm^2]
    LOvec: array
        vector of LO phonon frequencies in eV
    TOvec: array
        vector of TO phonon frequencies in eV; must be ordered to pair with LOvec
    """
    if(LOvec == 0):
        LOvec = self.LOvec
    if(TOvec == 0):
        TOvec = self.TOvec

    rt = 0
    for j in range(len(LOvec)):
        omLO = LOvec[j]
        omTO = TOvec[j]
        foo = self._R_phonon_Frohlich_branch(omLO,omTO)
        for k in range(len(LOvec)):
            if( k != j):
                foo = foo * (TOvec[k]**2 - LOvec[j]**2)/(LOvec[k]**2 - LOvec[j]**2)
        rt = rt + foo
    return rt*sigmae  # 1/kg/day


def _R_phonon_Frohlich_branch(self,omegaLO,omegaTO,rho=0,eps_inf=0):
    """
    Returns phonon excitation rate in 1/kg/yr by using Frohlich analytic approximation
    Only applies for massless mediators. This function does a calculation for a single branch
    with energies omegaLO, omegaTO, while R_phonon_Frohlich includes proper sum over all branches
    """

    vv = np.linspace(1e-6,self.vesc+self.veavg,200)  # in units of c
    vint = np.zeros_like(vv)

    if(rho == 0):
        rho = self.rhoT  # gram/cm^3
    rhokg =  rho*1e-3

    fac = self.eVtoInvYr*self.rhoX/self.mX/rhokg/(2*np.pi)
    # Multiply by Qx^2/sigmae where sigmae is in cm^2
    fac = fac * (self.alphaEM*self.me)**4/(16*np.pi*self.muXe**2*self.alphaEM**2)/self.eVcm**2

    if(eps_inf == 0):
        eps_inf = self.eps_inf

    CFrohsq = self.eEMparticle**2*(omegaLO**2 - omegaTO**2)/(2*eps_inf*omegaLO)

    for i in range(len(vv)):
        vi = vv[i]
        ki = self.mX*vi
        kf = self.mX*np.sqrt(abs(vi**2 - 2*omegaLO/self.mX))
        Qmin = abs(ki-kf)
        Qmax = ki + kf
        if(Qmin > Qmax or vi**2  < 2*omegaLO/self.mX):
            continue
        vint[i] = self.fv_1d(vi)/(vi)*np.log(Qmax/Qmin)*CFrohsq

    return fac*integrate.trapz(vint,x=vv)
