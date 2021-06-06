import numpy as np
from numpy import linspace, sqrt, array, pi, cos, sin, dot, exp, sinh, log, log10, cosh, sinh
from scipy.interpolate import interp1d, interp2d
from scipy import integrate

############################################################################################

def dRdomegadk_electron(self,omega,k,sigmae=1e-38,withscreening=True, method="grid"):
    """
    Returns double differential rate for DM-electron scattering in 1/kg/year/eV^2

    Inputs
    ------
    omega: float 
        electron excitation energy in [eV]
    k: float 
        momentum transfer in [eV]    
    sigmae: float
        cross section in [cm^2]
    withscreening: Boolean
        whether to include the 1/|epsilon|^2 factor in the scattering rate
    method: ["grid","Lindhard"]
        use interpolated grid of epsilon, or Lindhard analytic epsilon
    """
    etav_val = self.etav(self.vmin(omega,k))
    temp_eps1=self.eps1(omega,k,method=method)
    temp_eps2=self.eps2(omega,k,method=method)

    if(method!="grid" and method!="Lindhard"):
        print("Error: unknown method. Please choose `Lindhard` or `grid`")
        return 0.0
    
    # units of 1/kg/year/eV^2
    dR = etav_val * self.rhoX/self.mX * 1000./self.rhoT * self.eVtoInvYr \
            * 1/(2*pi)**2 * sigmae/self.eVcm**2/ self.muXe**2 \
            * 1.0/(2.0*self.alphaEM) * k**3 * self.Fmed_electron(k)**2 * temp_eps2
    if(withscreening):
            dR = dR/(temp_eps1**2 + temp_eps2**2)
    return dR


def dRdomega_electron(self,omega, sigmae=1e-38, kcut=0, withscreening=True, method="grid"):
    """
    Returns differential rate for DM-electron scattering in 1/kg/yr/eV

    Inputs
    ------
    omega: float or array
        electron excitation energy in [eV]
    sigmae: float
        cross section in [cm^2]
    kcut: float
        option to include a maximum k value in the integration (helpful if you
        wish to avoid to using ELF in high k regime where it may be more uncertain)
        if kcut=0 (default), the integration is cut off at the highest k-value of the grid at hand
    withscreening: Boolean
        whether to include the 1/|epsilon|^2 factor in the scattering rate
    method: ["grid","Lindhard"]
        use interpolated grid of epsilon, or Lindhard analytic epsilon
    """
    
    # integrate over k range where integrand is nonzero, and with possible user-specified max k
    if(kcut == 0):
        kcut = self.kmax

    # Integration over k
    scalar_input = np.isscalar(omega)
    omega = np.atleast_1d(omega)
    dRdomega = np.zeros_like(omega)
    for i in range(len(omega)):
        # Kinematic limits on k range
        kmin = self.qmin(omega[i])
        kmax = self.qmax(omega[i])
        if(method == "Lindhard"):  
            # Accounts for the finite range where eps2 is nonzero in Lindhard.
            kmin1 =  max( np.sqrt( 2*self.me * omega[i] + self.kF**2 ) - self.kF, kmin ) 
            kmax1 =  min( np.sqrt( 2*self.me * omega[i] + self.kF**2 ) + self.kF, kmax ) 
            kmin, kmax = kmin1, kmax1
        else:
            # Set max of k range based on grid range if using grid
            kmax = min(kmax,kcut)

        if(kmin >= kmax): # accounts for kmin=kmax=0 when omega is too high (kinematics)
            continue

        # Note: division and multiplication by self.eVtoInvYr in the integrand improves performance of quad
        dRdomega[i] = self.eVtoInvYr * integrate.quad(lambda x: self.dRdomegadk_electron(omega[i],x,sigmae, \
                    withscreening=withscreening,method=method)/self.eVtoInvYr, kmin, kmax, limit=50)[0] 
                     # units of 1/kg/yr/eV
    
    if(scalar_input):
        return dRdomega[0]
    else:
        return dRdomega
        

def R_electron(self,threshold=-1.0,Emax=-1.0,sigmae=1e-38, kcut = 0, withscreening=True, method="grid"):
    """
    Returns total number of events per 1/kg/yr, normalized to a reference cross section sigmae.
    Inputs
    ------   
    sigmae: float
        reference cross section in [cm^2]
    threshold: float
        energy threshold in eV. Defaults to the 2e- threshold 
        when the average number of ionization electrons is available. If this information is not available,
        the default threshold is twice the bandgap.
    Emax: float
        max energy considered in reach. Will integrate over the minimum of 
          [max kinematically accessible energy, max of energy range in ELF grid, Emax]
    kcut: float
        option to include a maximum k value in the integration (helpful if you
        wish to avoid to using ELF in high k regime where it may be more uncertain)
        if kcut=0 (default), the integrating is cut off at the highest k-value of the grid at hand
    withscreening: Boolean
        whether to include the 1/|epsilon|^2 factor in the scattering rate
    method: ["grid","Lindhard"]
        use interpolated grid of epsilon, or Lindhard analytic epsilon
    """
    if (threshold<0.0):
          if(hasattr(self, "e0")):
            threshold=self.E_gap+self.e0
          else:
            threshold=np.max([2.0*self.E_gap,1e-3]) # prevent zero threshold for metals
  
    if (Emax < 1.0):
        Emax = np.min([self.ommax,0.5*(self.vesc+self.veavg)**2*self.mX])
    else:
        Emax = np.min([self.ommax,0.5*(self.vesc+self.veavg)**2*self.mX, Emax])

    olist=np.linspace(threshold,Emax,200)
    return integrate.trapz(self.dRdomega_electron(olist,sigmae=sigmae,kcut=kcut, \
        withscreening=withscreening,method=method), x=olist)
    

############################################################################################

# number of ionization electrons. Taken from 1509.01598
def electron_yield(omega):
    """
    Number of ionization electrons for a given energy omega
    """
    if(hasattr(self, "e0") and hasattr(self, "E_gap")):
        return 1+np.floor((omega-self.E_gap)/self.e0)
    else:
        print("This function is not available for "+self.target)
        return 0.0

# differential rate as function of nr of ionization electrons
def dRdQ_electron(self,Q, sigmae=1e-38, kcut = 0, withscreening=True,method="grid"):
    """
    Returns differential rate in terms of the charge yield for DM-electron scattering in 1/kg/yr. Available for select materials

    Inputs
    ------
    Q: integer or list of integers
        number of ionization electrons    
    sigmae: float
        cross section in [cm^2]
    kcut: float
        option to include a maximum k value in the integration (helpful if you
        wish to avoid to using ELF in high k regime where it may be more uncertain)
        if kcut=0 (default), the integrating is cut off at the highest k-value of the grid at hand
    withscreening: Boolean
        whether to include the 1/|epsilon|^2 factor in the scattering rate
    method: ["grid","Lindhard"]
        use interpolated grid of epsilon, or Lindhard analytic epsilon
    """

    assert(hasattr(self, "e0") and hasattr(self, "E_gap")), \
        "This function is not available for "+self.target+" due to missing e0 and E_gap"

    Qlist=np.atleast_1d(Q)
    scalar_input = np.isscalar(Q)
    dRdQ = np.zeros_like(Qlist)
    for i in range(len(Qlist)):
        if(Qlist[i]<1.0):
            print("invalid nr of ionization electrons")
            dRdQ[i]=0.0
        else:
            olist=np.linspace(self.E_gap+(Qlist[i]-1.0)*self.e0,self.E_gap+Qlist[i]*self.e0,20)
            deltao=olist[1]-olist[0]
            dRdQ[i]=integrate.trapz(self.dRdomega_electron(olist,sigmae=sigmae,kcut=kcut, \
                withscreening=withscreening,method=method), x=olist)
    if scalar_input:
        return dRdQ[0]
    else:
        return dRdQ
