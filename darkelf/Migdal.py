import numpy as np
from numpy import linspace, sqrt, array, pi, cos, sin, dot, exp, sinh, log, log10, cosh, sinh
from scipy.interpolate import interp1d, interp2d
from scipy.special import erfc, erf
from scipy import integrate
import sys, os, glob

############################################################################################


# Auxiliary functions
########################

# incomplete error function, better numerical stability by using erfc
def _incomErf(x,y):
  return erfc(x)-erfc(y)

# routine which loads and interpolates the atomic form factors from Ibe et al
def load_Migdal_FAC(self,datadir):
    # Load files for atomic Migdal effect
    FAC_fname = datadir+self.target+'/'+self.target+'_Migdal_FAC.dat'
    if (not os.path.exists(FAC_fname)):
        self.ibe_available=False
        print("Warning! Atomic Migdal calculation not present")
    else:
        self.ibe_available=True
        with open(FAC_fname) as f:
            lines = [line.rstrip() for line in f]
        # Columns are electron energy [eV] and differential probability dp/dE in units of 1/eV
        FAC_list = []
        FAC_listname = ['$n=1$', '$n=2, \ell = 0$', '$n=2, \ell=1$', '$n=3, \ell = 0$', \
            '$n=3, \ell=1$', '$n=3, \ell=2$', '$n=4, \ell = 0$', '$n=4, \ell=1$', '$n=4, \ell=2$', '$n=5, \ell=0$', '$n=5, \ell=1$']
        FAC_Nshells = int(len(lines)/(254))
        self.FAC_Nshells = FAC_Nshells
        for j in range(FAC_Nshells):
            foo = lines[3 + j*254: (j+1)*254]
            FAC_list = FAC_list + [np.transpose(np.array([ [float(fooi[0:16]), float(fooi[17:])] for fooi in foo]))]
        # define interpolated dp/dE for shells
        # Note -- The probabilities need to be rescaled by (qe/1 eV)^2 for qe ̸= 1 eV.
        #self._dpdomega_FAC = [ interp1d(FAC_list[0][0],  [ sum([FAC_list[j][1][i] for j in range(FAC_Nshells-2)] )  for i in range(len(FAC_list[0][0]))  ])
        self._dpdomega_FAC = [ interp1d(FAC_list[0][0] + self.Enl_list[j], FAC_list[j][1]) for j in range(FAC_Nshells) ]
    return

# elastic nuclear recoils, no ionization
################################################

def dRdEn_nuclear(self,En,sigma_n=1e-40):
    """ 
    Returns rate for elastic scattering of DM off free nucleus, without ionization

    Inputs
    ------
    En: float or array
        Nuclear recoil energy in [eV]
    sigma_n: float
        DM-nucleon cross section in [cm^2], defined at reference momentum of q0. 
        DM-nucleus cross section assumed to be coherently enhanced by A^2

    Outputs
    -------
    rate as function of En, in [1/kg/yr/eV]
    """
    scalar_input = np.isscalar(En)

    En = np.atleast_1d(En)

    q = np.sqrt(2*self.mN*En)
    vmin = q/(2.0*self.muxN)
    # rate in 1/kg/yr/eV
    rt = ( self.vesc + self.veavg - vmin) * \
        self.NTkg * self.rhoX/self.mX * self.A**2*sigma_n * self.c0cms * 86400 * 365.* \
        self.mN/(2*self.muxnucleon**2) * self.etav(vmin)*  \
        self.Fmed_nucleus(q)**2 # Form factor mediator
    if(scalar_input):
        return rt[0]
    else:
        return rt

# shake-off probabilities, in soft limit
################################################      
      
# dP/dkdomega
def dPdomegadk(self,omega,k,En,method="grid"):
    """ 
    Returns double differential ionization probability dP/domega dk, in the soft limit 

    Inputs
    ------
    omega: float or array
          energy deposited into electronic excitations
    k: float
          momentum deposited into electronic excitations      
    En: float
        Nuclear recoil energy in [eV]
    method: ["grid","Lindhard"]
        use interpolated grid of epsilon, or Lindhard analytic epsilon    
    Outputs
    -------
    dP/domega dk for the specified En, in [1/eV^2]
    """
    scalar_input = np.isscalar(omega)
    omega = np.atleast_1d(omega)
    dPdomegadk = np.zeros_like(omega)
    vN=np.sqrt(2.0*En/self.mN)
    
    for i in range(len(omega)):
      if(hasattr(self, "Zion") and self.electron_ELF_loaded):
        dPdomegadk[i]=(2.0*self.alphaEM*self.Zion**2*vN**2)/(3.0*np.pi**2*omega[i]**4)*k**2*self.elf(omega[i],k,method=method)
      else:
        if(i==0): print("This function is not available for "+self.target)
        dPdomegadk[i]=0.0
        
    if(scalar_input):
        return dPdomegadk[0]
    else:
        return dPdomegadk  
      
# dP/domega
def dPdomega(self,omega,En,method="grid",kcut=0,Nshell=0):
    """ 
    Returns differential ionization probability dP/domega, in the soft limit 

    Inputs
    ------
    omega: float or array
          energy deposited into electronic excitations
    En: float
        Nuclear recoil energy in [eV]
    method: ["grid","Lindhard","Ibe"]
        use interpolated grid of epsilon, Lindhard analytic epsilon or the atomic calculation by Ibe et al 1707.07258
    kcut: float
        option to include a maximum k value in the integration (helpful if you
        wish to avoid to using ELF in high k regime where it may be more uncertain)
        if kcut=0 (default), the integration is cut off at the highest k-value of the grid at hand.
        Only used if method="grid" is selected.
    Nshell: int
        Number of atomic shells included in the Ibe et. al. calculation. Only used if method="Ibe" is selected.
        By default, all available shells are included.   

    Outputs
    -------
    dP/domega for the specified En, in [1/eV]
    """
    scalar_input = np.isscalar(omega)
    omega = np.atleast_1d(omega)
    dPdomega = np.zeros_like(omega)
    
    # protect against capitalization typos in the flags
    if method=="lindhard":
      method="Lindhard"
    if method=="ibe":
      method="Ibe"
    
    if method=="grid" or method=="Lindhard":
      for i in range(len(omega)):
        if(hasattr(self, "Zion") and self.electron_ELF_loaded):
          if(kcut == 0):
            kcut = self.kmax
          # perform integral in log-space, for better convergence  
          dPdomega[i]=integrate.quad(lambda logk: 10**logk*np.log(10.)*self.dPdomegadk(omega[i],10**logk,En,method=method),1.0,np.log10(kcut),limit=100,full_output=1)[0]
        else:
          if(i==0): print("This function is not available for "+self.target)
          dPdomega[i]=0.0
          
    if method=="Ibe":
      if(self.ibe_available==False):
        print("This function is not available for "+self.target)
        return 0
      else:
        prefac=1.0/(2.0*pi)*En*2.0*self.me**2/(self.mN*1.0**2) # see (91) of Ibe et al
        if(Nshell == 0 or Nshell > self.FAC_Nshells):
          Nshell = self.FAC_Nshells
        for i in range(len(omega)):  
          dPdomega[i] = 0.0
          for j in range(Nshell):
            # Ibe et al. only goes down to 1 eV in E_e. So for E_e between 0 and 1 eV, just use the 1 eV value
            if(omega[i] <= self.Enl_list[j] + 1) :
              if(omega[i] >= self.Enl_list[j]):
                dPdomega[i] = dPdomega[i] + prefac*self._dpdomega_FAC[j](1.0 + self.Enl_list[j])
            else:
              dPdomega[i] = dPdomega[i] + prefac*self._dpdomega_FAC[j](omega[i])
                  
    if(scalar_input):
        return dPdomega[0]
    else:
        return dPdomega   
      

# Migdal rates, in soft limit
################################################

# auxiliary function I(omega)=1/En dP/domega              
def _I(self,omega,method,kcut,Nshell):
    if(hasattr(self, "Zion") and self.electron_ELF_loaded):
        return self.dPdomega(omega,1.0,method=method,kcut=kcut,Nshell=Nshell)
    else:
        print("This function is not available for "+self.target)
        return 0.0

    
# pretabulate "I(omega)" and store as an interpolated function. This can be used to speed up the computation of the rate
# the user can call this function to overwrite the stored function with different settings      
def tabulate_I(self,method="grid",kcut=0,Nshell=0):
    """ 
    tabulates and interpolates I(omega)=1/En dP/domega and stores the result as an internal function, which can be used to speed up the rate calculations 

    Inputs
    ------
    method: ["grid","Lindhard","Ibe"]
        use interpolated grid of epsilon, Lindhard analytic epsilon or the atomic calculation by Ibe et al 1707.07258
    kcut: float
        option to include a maximum k value in the integration (helpful if you
        wish to avoid to using ELF in high k regime where it may be more uncertain)
        if kcut=0 (default), the integration is cut off at the highest k-value of the grid at hand.
        Only used if method="grid" is selected.
    Nshell: int
        Number of atomic shells included in the Ibe et. al. calculation. Only used if method="Ibe" is selected.
        By default, all available shells are included.   

    Outputs
    -------
    None
    """
    if(hasattr(self, "Zion") and self.electron_ELF_loaded):
      omlist=np.linspace(np.max([2.0*self.E_gap,1.0]),self.ommax,50) # start the integral at twice the band gap, as GPAW data is noisy for E_gap < omega < 2 E_gap. Require omega > 1.0 eV, to avoid NaN for materials with E_gap=0.0.
      Ilist=self._I(omlist,method,kcut,Nshell)
      self.I_tab=interp1d(omlist,Ilist,fill_value=0.0,bounds_error=False)
  
      
# auxiliary function J(v,omega)=\int dEn En dsigma_eq/dEn. 
# Enth is the low energy threshold on En, which needs to be set to avoid the breakdown of the various approximations        
def _J(self,v,omega,approximation,Enth,sigma_n):
  # auxiliary parameter
  prefactor=2.0*np.pi**2*self.A**2*sigma_n/(v*self.muxnucleon**2)
  
  if approximation=="free":
    # boundary conditions
    qmax=v*self.muxN*(1+np.sqrt(1.0-2.0*omega/(v**2*self.muxN)))
    qmin=np.max([v*self.muxN*(1-np.sqrt(1.0-2.0*omega/(v**2*self.muxN))),np.sqrt(2.0*self.mN*Enth)])
    
    if(self.mMed>10.0*v*self.mX): # general formula is numerically unstable in the massive mediator limit, switch to limiting case.
      return prefactor*(qmax**4-qmin**4)/(32.0*np.pi**2*self.mN*v)*(qmin<qmax)
    else:
      return prefactor*(self.q0**2+self.mMed**2)**2/(16.0*np.pi**2*v*self.mN)*(self.mMed**2/(qmax**2+self.mMed**2)-self.mMed**2/(qmin**2+self.mMed**2)+np.log((qmax**2+self.mMed**2)/(qmin**2+self.mMed**2)))*(qmin<qmax)
  
  elif approximation=="impulse":
    
    if(not hasattr(self, "ombar")):
      print("No phonon frequency found for this material. Specify ombar in yaml file or use the free approximation.")
      sys.exit()# abort evaluation
    else:

      # auxiliary parameters
      Delta=np.sqrt(self.ombar*self.mN)
      prefactor_imp=1.0/(32.0*np.pi**2.5*v*self.mN)
    
      # boundary conditions
      qNmax=(lambda q: np.sqrt(2.0*self.mN*(v*q-q**2/(2*self.mX)-omega)))
      qNmin=np.sqrt(2.0*self.mN*Enth)
      qmax=v*self.mX*(1.0+np.sqrt(1-2.0*omega/(v**2*self.mX)))
      qmin=v*self.mX*(1.0-np.sqrt(1-2.0*omega/(v**2*self.mX)))
    
      #integrant
      fun=(lambda q, qN: 2.0*Delta*(q**2-q*qN+qN**2+Delta**2)*np.exp(-(q+qN)**2/Delta**2)-2.0*Delta*(q**2+q*qN+qN**2+Delta**2)*np.exp(-(q-qN)**2/Delta**2)+np.pi**0.5*q*(2.0*q**2+3.0*Delta**2)*_incomErf((q-qN)/Delta,(q+qN)/Delta))
      integrant=(lambda q: self.Fmed_nucleus(q)**2*(fun(q,qNmax(q))-fun(q,qNmin))*(qNmax(q)>qNmin))
    
      #integral
      return prefactor*prefactor_imp*integrate.quad(lambda q: integrant(q),qmin,qmax)[0]
    
  else:
    print("unknown approximation flag, please use 'free' or 'impulse'.")
    return 0.0
    

# Migdal rate    
def dRdomega_migdal(self,omega,Enth=-1.0,sigma_n=1e-38,method="grid",approximation="free",kcut=0,Nshell=0,fast=False):
    """ 
    Returns differential rate for ionization from the Migdal effect, in 1/kg/yr/eV

    Inputs
    ------
    omega: float or array
        electron excitation energy in [eV]
    sigma_n: float
        DM-nucleon reference cross section in [cm^2]
    Enth: float
        lower bound on nuclear recoil energy, enables the user to exclude the soft nuclear recoil part of the phase space, where the impulse and free approximations are invalid. The default value is set to 4 times the average phonon frequency, specified by the ombar parameter in the yaml file.
    method: ["grid","Lindhard","Ibe"]
        use interpolated grid of epsilon, Lindhard analytic epsilon or the atomic calculation by Ibe et al 1707.07258
    approximation: ["free","impulse"] 
        use impulse approximation or free ion approximation
    kcut: float
        option to include a maximum k value in the integration (helpful if you
        wish to avoid to using ELF in high k regime where it may be more uncertain)
        if kcut=0 (default), the integration is cut off at the highest k-value of the grid at hand
    Nshell: int
        Number of atomic shells included in the Ibe et. al. calculation. Only used if method="Ibe" is selected.
        By default, all available shells are included.
    fast: boole 
        If set to "True", darkELF will use the pretabulated shake-off probability, to speed up the computation. The pretabulated shake-off probability can be updated by calling the "tabulate_I" function. The default for this flag is "False". If fast is set to "True", the "method", "kcut" and "Nshell" flags are ignored.
    """
    scalar_input = np.isscalar(omega)
    omega = np.atleast_1d(omega)
    dRdomega = np.zeros_like(omega)
    
    if(Enth<0.0):
      if(hasattr(self, "ombar")):
        Enth=4.0*self.ombar
      else:
        print("No phonon frequency found for this material. Setting Enth=0.1 eV")
        Enth=0.1
    
    for i in range(len(omega)):
      if(hasattr(self, "Zion")):
        vmin = np.sqrt(2*omega[i]/self.muxN)
        vmax=self.vesc + self.veavg
        if( vmin > vmax):
          dRdomega[i]=0
        else:
          vint=integrate.quad( lambda v:v*self.fv_1d(v)*self._J(v,omega[i],approximation=approximation,Enth=Enth,sigma_n=sigma_n),vmin,vmax)[0]
          if(fast): # use pretabulated shake-off probability, to speed up computation
            dRdomega[i]=self.rhoX/(self.mN*self.mX)*self.I_tab(omega[i])*vint*self.c0cms*self.yeartosec/self.eVtokg
          else:
            dRdomega[i]=self.rhoX/(self.mN*self.mX)*self._I(omega[i],method=method,kcut=kcut,Nshell=Nshell)*vint*self.c0cms*self.yeartosec/self.eVtokg
      else:
        if(i==0): print("This function is not available for "+self.target)
        dRdomega[i]=0.0

    if(scalar_input):
        return dRdomega[0]
    else:
        return dRdomega  
    
    
    
    
def R_migdal(self,threshold=-1.0,sigma_n=1e-38,Enth=-1.0,method="grid",approximation="free",kcut=0,Nshell=0,fast=False):
    """ 
    Returns integrated rate for ionization from the Migdal effect, in 1/kg/yr

    Inputs
    ------
    threshold: energy threshold in [eV]. 
        Defaults to the 2e- threshold when the average number of ionization electrons is available. If this information is not available,
        the default threshold is twice the bandgap.
    sigma_n: float
        DM-nucleon reference cross section in [cm^2]
    Enth: float
        lower bound on nuclear recoil energy, enables the user to exclude the soft nuclear recoil part of the phase space, where the impulse and free approximations are invalid. The default value is set to 4 times the average phonon frequency, specified by the ombar parameter in the yaml file.
    method: ["grid","Lindhard","Ibe"]
        use interpolated grid of epsilon, Lindhard analytic epsilon or the atomic calculation by Ibe et al 1707.07258
    approximation: ["free","impulse"] 
        use impulse approximation or free ion approximation    
    kcut: float
        option to include a maximum k value in the integration (helpful if you
        wish to avoid to using ELF in high k regime where it may be more uncertain)
        if kcut=0 (default), the integration is cut off at the highest k-value of the grid at hand
    Nshell: int
        Number of atomic shells included in the Ibe et. al. calculation. Only used if method="Ibe" is selected.
        By default, all available shells are included.
    fast: boole 
        If set to "True", darkELF will use the pretabulated shake-off probability, to speed up the computation. The pretabulated shake-off probability can be updated by calling the "tabulate_I" function. The default for this flag is "False". If fast is set to "True", the "method", "kcut" and "Nshell" flags are ignored.      
    """
    if (threshold<0.0):
          if(hasattr(self, "e0")):
            threshold=self.E_gap+self.e0
          else:
            threshold=2.0*self.E_gap
    olist=np.linspace(threshold,self.ommax,200)
    return integrate.trapz(self.dRdomega_migdal(olist,Enth=Enth,sigma_n=sigma_n,method=method,approximation=approximation,kcut=kcut,Nshell=Nshell,fast=fast), x=olist)    
    
    
    
    
    

# stuff below is obsolete    
#######################    
    
    
    
# auxiliary function J(v,omega)=\int dEn En dsigma_eq/dEn        
def _Jold(self,v,omega,mediator,approximation,Enth,sigma_n):
  if approximation=="free":
    
    # check and update if needed.
    ERmax = self.muxN**2*v**2/self.mN * (1 - omega/(self.muxN*v**2) + np.sqrt(1 - 2*omega/(self.muxN*v**2)  ) )
    ERmin = max(Enth, self.muxN**2*v**2/self.mN * (1.0 - omega/(self.muxN*v**2) - np.sqrt(1 - 2*omega/(self.muxN*v**2)) ) )
    
    if mediator=="massive":
      return (self.A**2*sigma_n*self.mN)/(4.0*v**2*self.muxnucleon**2)*(ERmax**2-ERmin**2)*(ERmax>ERmin)
    if mediator=="massless":
      return (self.A**2*sigma_n*self.q0**4)/(8.0*v**2*self.muxnucleon**2*self.mN)*np.log(ERmax/ERmin)*(ERmax>ERmin)
    
  elif approximation=="impulse":
    Delta=np.sqrt(self.ombar*self.mN)
    Enmax=0.5*v**2*self.mX-omega
    Enmin=Enth
    if Enmax<Enmin:
      return 0.0
    else:
      qmax_fun=(lambda En: v*self.mX*(1+np.sqrt(1-2.0*En/(v**2*self.mX)-2.0*omega/(v**2*self.mX))))
      qmin_fun=(lambda En: v*self.mX*(1-np.sqrt(1-2.0*En/(v**2*self.mX)-2.0*omega/(v**2*self.mX))))
      if mediator=="massive":
        return (self.A**2*sigma_n*self.mN)/(4.0*v**2*self.muxnucleon**2)*\
      integrate.quad(lambda En:En*(erf((qmax_fun(En)-np.sqrt(2.0*En*self.mN))/Delta)
                     -erf((qmax_fun(En)+np.sqrt(2.0*En*self.mN))/Delta)
                     -erf((qmin_fun(En)-np.sqrt(2.0*En*self.mN))/Delta)
                     +erf((qmin_fun(En)+np.sqrt(2.0*En*self.mN))/Delta)),Enmin,Enmax)[0]
      if mediator=="massless":
        return integrate.dblquad(lambda q, En: np.exp((-2.0*En*self.mN+q**2)/Delta**2)*En**0.75/q**4*np.sinh(2.0*np.sqrt(2.0*En*self.mN)*q/Delta**2), Enmin, Enmax, qmin_fun, qmax_fun)[0]    
    
    
    
    
    
    

# old functions
##################

# Crystal form factor, in impulse approximation
################################################

def _Fcrystal(self,q):
  """
  F(q) crystal form factor, in the isotropic impulse approximation, see appendix A and B of 2011.09496 
  """
  return (4.0*np.pi/(self.mN*self.ombar))**0.75*np.exp(-q**2/(2.0*self.mN*self.ombar))


#
def dRdER_elastic_nucleus(self,ER,sigma_n=1e-40,q0=1e3):
    """ 
    Returns rate for DM nucleus elastic scattering, without ionization

    Inputs
    ------
    ER: float or array
        Nuclear recoil energy in [eV]
    sigma_n: float
        DM-nucleon cross section in [cm^2], defined at reference momentum of q0. 
        DM-nucleus cross section assumed to be coherently enhanced by A^2
    q0: float
        reference momentum in [eV]

    Outputs
    -------
    rate as function of ER, in [1/kg/yr/eV]
    """
    scalar_input = np.isscalar(ER)

    ER = np.atleast_1d(ER)

    q = np.sqrt(2*self.A*self.mp*ER)
    vmin = q/(2.0*self.muxN)
    # rate in 1/kg/yr/eV
    rt = ( self.vesc + self.veavg - vmin) * \
        self.NTkg * self.rhoX/self.mX * self.A**2*sigma_n * self.c0cms * 86400 * 365.* \
        self.A*self.mp/(2*self.muxnucleon**2) * self.etav(vmin)*  \
        self.Fmed_nucleus(q,q0=q0)**2  # Form factor mediator

    if(scalar_input):
        return rt[0]
    else:
        return rt

############################################################################################
# ----- return differential rate for ionization signals from DM nucleus scattering
#
# dR/domega/dk/dE_R

# this one is not used?
def dRdomegadkdER_nucleus(self,omega,k,ER,sigma_n=1e-40,mediator='massive'):
    # momentum transfer is set by NR
    qN = np.sqrt(self.mN*ER)
    etav_val = self.etav(self.vmin(omega + ER,qN))
    if(etav_val > 0.0):
        fac = self.NTkg * self.rhoX/self.mX * self.A**2*sigma_n * self.c0cms * 86400 * 365. # 1/kg/yr
        facER = ER/(2*self.muxnucleon**2)  # mN/(2muxn^2) * ER/mN --- units of 1/eV
        facomegak = 8*self.alphaEM*self.Zion**2*(2*pi)*1./3.*k**2/(2*pi)**3/(omega**4) # --- units of 1/eV^2
        return fac * facER * facomegak * etav_val * self.elf(omega,k)
    else:
        return 0.0

# integral of E_R*dE_R for massive mediator
def _nucleus_ERintegral_massive(self,v,omega=1.0,ERth=0.1):
    if( np.sqrt(2*omega/self.muxN) > v):
        return 0
    else:
        ERmax = self.muxN**2*v**2/(self.A*self.mp) * (1 - omega/(self.muxN*v**2) + np.sqrt(1 - 2*omega/(self.muxN*v**2)  ) )
        ERmin = max(ERth, self.muxN**2*v**2/(self.A*self.mp) * (1 - omega/(self.muxN*v**2) - np.sqrt(1 - 2*omega/(self.muxN*v**2)) ) )
        #print(ERmin,ERmax)
        return (ERmax > ERmin)*0.5*(ERmax**2 - ERmin**2)

# integral of 1/E_R*dE_R for massless mediator
def _nucleus_ERintegral_massless(self,v,omega=1.0,ERth=0.1):
    if( np.sqrt(2*omega/self.muxN) > v):
        return 0
    else:
        ERmax = self.muxN**2*v**2/(self.A*self.mp) * (1 - omega/(self.muxN*v**2) + np.sqrt(1 - 2*omega/(self.muxN*v**2)  ))
        ERmin = max(ERth, self.muxN**2*v**2/(self.A*self.mp) * (1 - omega/(self.muxN*v**2) - np.sqrt(1 - 2*omega/(self.muxN*v**2)  )) )
        return (ERmax > ERmin)*np.log(ERmax/ERmin)/(4*self.A**2*self.mp**2)

# Differential rate spectrum. q0 is reference momentum [eV] for massless mediator scenario
def dRdomega_nucleus(self,omega,kcut = 0,ERth=0.1,sigma_n=1e-40,mediator='massive',q0=1000.):
    if(kcut == 0):
        kcut = self.kmax
    vmin = np.sqrt(2*omega/self.muxN)
    if( vmin > (self.vesc + self.veavg)):
        return 0
    else:
        fac = 2.* self.NTkg * self.rhoX/self.mX * self.A**2*sigma_n * self.c0cms * 86400 * 365. # 1/kg/yr
        #JK: Note added factor of 2 above
        if(mediator == 'massive'):
            facvER = 1/(2*self.muxnucleon**2) * integrate.quad( lambda x: self.fv_1d(x)/x*self._nucleus_ERintegral_massive(x,omega=omega,ERth=ERth), vmin , 3e-3 )[0]  # dim-less
        else:
            facvER = 1/(2*self.muxnucleon**2) * pow(q0,4) * integrate.quad( lambda x: self.fv_1d(x)/x*self._nucleus_ERintegral_massless(x,omega=omega,ERth=ERth), vmin , 3e-3 )[0]  # dim-less 
        fack = 8*self.alphaEM*self.Zion**2*(2*pi)*1./3.*1.0/(2*pi)**3/(omega**4) * integrate.quad(lambda k: self.elf(omega,k)*k*k, 1.0, kcut)[0]  # -- 
        #units of 1/eV
        return fack*fac*facvER  # 1/kg/yr/eV

# Midgal rate from Ibe et al.
def dRdomega_Ibe(self,omega,ERth=0.1,sigma_n=1e-40,mediator='massive',q0=1000.,Nshell=0):
    vmin = np.sqrt(2*omega/self.muxN)
    if( vmin > (self.vesc + self.veavg)):
        return 0
    else:
        fac = self.NTkg * self.rhoX/self.mX * self.A**2*sigma_n * self.c0cms * 86400 * 365. # 1/kg/yr
        if(mediator == 'massive'):
            facvER = 1/(2*self.muxnucleon**2) * integrate.quad( lambda x: self.fv_1d(x)/x*self._nucleus_ERintegral_massive(x,omega=omega,ERth=ERth), vmin , 3e-3 )[0]  # dim-less 
        else:
            facvER = 1/(2*self.muxnucleon**2) * pow(q0,4) * integrate.quad( lambda x: self.fv_1d(x)/x*self._nucleus_ERintegral_massless(x,omega=omega,ERth=ERth), vmin , 3e-3 )[0]  # dim-less 
        fack = 1.0/(2.0*pi)*self.dpdomega_FAC(omega,Nshell=Nshell)*2.0*(self.me/1.0)**2 # -- units of 1/eV
        return fack*fac*facvER  # 1/kg/yr/eV
    
      
# Rates for crystal including form factor

def crystal_ERintegral_massive(self,v,omega=1.0,ERth=0.1):
    if( np.sqrt(2*omega/self.mX) > v):
        return 0
    else:
        mN=self.mp*self.A #Nucleus mass
        
        if ERth<0.5*self.mX*v**2-omega:
            
            denom = np.sqrt(mN*self.ombar)
            integral=integrate.quad( lambda x: 0.5*x*(erf((self.mX*v+np.sqrt(2.*self.mX*(0.5*self.mX*v**2 - x - omega))-np.sqrt(2.*mN*x))/denom) + erf((self.mX*v-np.sqrt(2.*self.mX*(0.5*self.mX*v**2 - x - omega))+np.sqrt(2.*mN*x))/denom) - erf((self.mX*v-np.sqrt(2.*self.mX*(0.5*self.mX*v**2 - x - omega))-np.sqrt(2.*mN*x))/denom) - erf((self.mX*v+np.sqrt(2.*self.mX*(0.5*self.mX*v**2 - x - omega))+np.sqrt(2.*mN*x))/denom)), ERth,0.5*self.mX*v**2-omega)[0]
        else:
            integral = 0
        return integral
        
def dRdomega_crystal(self,omega,kcut = 0,ERth=0.1, sigma_n=1e-40,mediator='massive',q0=1000.):
    if(kcut == 0):
        kcut = self.kmax
    vmin = np.sqrt(2*(omega+ERth)/self.mX)
    if( vmin > (self.vesc + self.veavg)):
        return 0
    else:
        fac = 2.*self.NTkg * self.rhoX/self.mX * self.A**2*sigma_n * self.c0cms * 86400 * 365. # 1/kg/yr
        if(mediator == 'massive'):
            facvER = 1/(2*self.muxnucleon**2) * integrate.quad( lambda x: self.fv_1d(x)/x*self.crystal_ERintegral_massive(x,omega=omega,ERth=ERth), vmin , 3e-3 )[0]  # dim-less
        else:
            #Not implemented yet; return 0 for now
            facvER = 0
#                facvER = 1/(2*self.muxnucleon**2) * pow(q0,4) * integrate.quad( lambda x: self.fv_1d(x)/x*self.nucleus_ERintegral_massless(x,omega=omega,ERth=ERth), vmin , 3e-3 )[0]  # dim-less
        fack = 8*self.alphaEM*self.Zion**2*(2*pi)*1./3.*1.0/(2*pi)**3/(omega**4) * integrate.quad(lambda k: self.elf(omega,k)*k*k, 1.0, kcut)[0]  # -- units of 1/eV
        return fack*fac*facvER  # 1/kg/yr/eV
        
        

        
############################################################################################

        
      
# define Migdal probability WRT the full detector energy and specifying number of shells
def dpdomega_FAC(self,omega,Nshell=0): 
    if(Nshell == 0 or Nshell > self.FAC_Nshells):
        Nshell = self.FAC_Nshells
    foo = 0.0
    for j in range(Nshell):
        # Ibe et al. only goes down to 1 eV in E_e. So for E_e between 0 and 1 eV, just use the 1 eV value
        if(omega <= self.Enl_list[j] + 1) :
            if(omega >= self.Enl_list[j]):
                foo = foo + self._dpdomega_FAC[j](1.0 + self.Enl_list[j])
        else:
            foo = foo + self._dpdomega_FAC[j](omega)
    return foo      

