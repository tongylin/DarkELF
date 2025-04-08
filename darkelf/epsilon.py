



import numpy as np
from numpy import linspace, sqrt, array, pi, cos, sin, dot, exp, sinh, log, log10, cosh, sinh
from scipy.interpolate import interp1d, interp2d
from scipy import integrate
import sys, os, glob
import pandas as pd

############################################################################################

# Function to load eps data and set eps1_grid and eps2_grid in electron recoil regime
def load_epsilon_grid(self,datadir,filename):

    fname = datadir +str(self.target)+"/" + filename

    if (not os.path.exists(fname)):
        self.electron_ELF_loaded=False
        print("Warning! Epsilon grid in electron regime does not exist.")

        om = np.arange(10)
        k = np.arange(10)
        e1 = np.zeros((10,10))
        e2 = np.zeros((10,10))

        self.kmin = 0.0
        self.kmax = 1.0e5
        self.ommax = 100.0

        return om, k, e1, e2
    else:
        self.electron_ELF_loaded=True
        print("Loaded " + filename + " for epsilon in electron regime")

    with open(fname) as f:
      citation = f.readline().replace("\n","")
      print("electronic ELF taken or calculated from "+citation)

    data = pd.read_csv(fname, delim_whitespace=True,header=None,skiprows=1,
                names=['omega', 'k', 'eps1', 'eps2'])

    data.fillna(inplace=True,method='bfill')# fill in some NaN values

    # Reshapes to array in omega, k
    eps1df = data.pivot(index='omega', columns='k', values='eps1')
    eps1df.interpolate()  # Interpolates any NaNs that might be present
    eps2df = data.pivot(index='omega', columns='k', values='eps2')
    eps2df.interpolate()  # Interpolates any NaNs that might be present
    # energies in eV
    omall = eps1df.index.values
    # momenta in eV
    kall = eps1df.columns.values

    eps1df = np.array(eps1df)
    eps2df = np.array(eps2df)
    self.eps1_grid = interp2d(omall,kall,eps1df.T, \
                fill_value=1.0,bounds_error=False) # set eps_1 to 1 when going outside the grid
    self.eps2_grid = interp2d(omall,kall,eps2df.T, \
                fill_value=0.0,bounds_error=False) # set eps_2 to zero when going outside the grid
    self.kmin = min(kall)
    self.kmax = max(kall)
    self.ommax = max(omall)

    return

def load_epsilon_phonon(self,datadir,filename):

    phonon_path = datadir + self.target+'/'+ filename

    if( not os.path.exists(phonon_path)):
        print("Warning! eps for phonon frequencies not loaded. Need to set phonon_filename to perform data-driven, single phonon calculations")
        self.phonon_ELF_loaded=False
    else:
        self.phonon_ELF_loaded=True

        with open(phonon_path) as f:
          citation = f.readline().replace("\n","")
          print("phonon ELF taken or calculated from "+citation)

        phonondat = np.loadtxt(phonon_path,skiprows=1).T
        print("Loaded " + filename + " for epsilon in phonon regime")
        self.eps1_phonon = interp1d(phonondat[0],phonondat[1],\
            fill_value=(phonondat[1][0],phonondat[1][-1]),bounds_error=False)
        self.eps2_phonon = interp1d(phonondat[0],phonondat[2],fill_value=0.0,bounds_error=False)
        self.omph_range = [ min(phonondat[0]), max(phonondat[0]) ]

    return

def load_eps_electron_opticallimit(self,datadir,filename):
  
  optical_path = datadir + self.target+'/'+ filename
  
  if (not os.path.exists(optical_path)):
    print(f"Warning, {filename} does not exist! dielectric function in optical limit not loaded. Needed for absorption calculations in superconductors.")
    self.eps_electron_opticallimit_loaded=False
  else:
    self.eps_electron_opticallimit_loaded=True
    
    with open(optical_path) as f:
      citation = f.readline().replace("\n","")
      print("optical data taken or calculated from "+citation)
    
      opticaldat = np.loadtxt(optical_path,skiprows=1).T
      print("Loaded " + filename + " for absorption calculations in superconductors")
      
      self.eps1_electron_optical = interp1d(opticaldat[0],opticaldat[1],\
            fill_value=(opticaldat[1][0],opticaldat[1][-1]),bounds_error=False)
      self.eps2_electron_optical=interp1d(opticaldat[0],opticaldat[2],\
                                          fill_value=0.0,bounds_error=False)
      self.om_electron_optical_range = [ min(opticaldat[0]), max(opticaldat[0]) ]
    
    
    
  
  

def load_Zion(self,datadir):

    Zion_path = datadir + self.target+'/'+ self.Zion_filename

    if( not os.path.exists(Zion_path)):
        print("Warning! Momentum dependent Zion for Migdal calculation not loaded. Using Z - number of valence electrons.")
        self.Zion_loaded=False
    else:
        self.Zion_loaded=True

        with open(Zion_path) as f:
          citation = f.readline().replace("\n","")
          print("Zion(k) for Migdal calculation taken or calculated from: "+citation)

        Ziondat = np.loadtxt(Zion_path,skiprows=1).T
        self.Zion_k = interp1d(Ziondat[0],Ziondat[1],\
            fill_value=(Ziondat[1][0],Ziondat[1][-1]),bounds_error=False)

    return

############################################################################################

# Electron gas dielectric function
def g(x):
    return (1 - x**2)*np.log( np.abs(1 + x)/np.abs(1 - x))

def eps1_electrongas(self,omega,k):
    u = omega/(k*self.vF)
    z = k/(2*self.me*self.vF)
    foo = 1 + 3*self.omegap**2/(self.vF**2)/k**2 *  ( 0.5 + 1/(8*z)* ( g(z - u) + g(z + u) ) )
    return foo


def eps2_electrongas(self,omega,k):
    u = omega/(k*self.vF)
    z = k/(2*self.me*self.vF)
    foo = 3*self.omegap**2/(self.vF**2)/k**2
    if( u + z <= 1):
        foo = foo*np.pi/2.0*u
    elif(u + z > 1):
        if(np.abs(z - u) >= 1):
            foo = foo*0
        elif(np.abs(z-u) < 1):
            foo = foo*np.pi/(8*z)*(1 - (z-u)**2 )
    return foo

############################################################################################

def eps1(self,omega,k,method="grid"):
    """
    Real part of epsilon(omega,k)

    Inputs
    ------
    omega: float or array
        energy in eV
    k: float
        energy in eV
    method = "grid" (using the grid loaded in filename), "Lindhard" (free electron gas), "phonon" for phonon absorption data, or "optical" of electron absorption data
    """

    scalar_input = np.isscalar(omega)
    omega = np.atleast_1d(omega)

    if method=="grid":
        # If k is smaller than grid kmin, we will extrapolate from lowest k point
        # This is implemented by changing all small k values to kmin
        k = k*(k >= self.kmin) + self.kmin*(k < self.kmin)
        eps1=self.eps1_grid(omega,k)
    elif(method=="Lindhard"):
        eps1=[self.eps1_electrongas(om,k) for om in omega]
    elif(method=='phonon'):
      if(hasattr(self, "eps1_phonon")==False):
        print("Error, eps for phonon frequencies not loaded. Need to set phonon_filename.")
        return 0
      else:
        eps1=[self.eps1_phonon(om) for om in omega]
    elif(method=='optical'):
      if(hasattr(self, "eps1_electron_optical")==False):
        print("Error, eps for electron frequencies not loaded. Need to set eps_electron_optical_filename.")
        return 0
      else:
        eps1=[self.eps1_electron_optical(om) for om in omega]    
    else:
        print("Error, unknown method. Please choose 'grid' or 'Lindhard'")
    if(scalar_input):
      return eps1[0]
    else:
      return np.array(eps1)

def eps2(self,omega,k,method="grid"):
    """
    Imaginary part of epsilon(omega,k)

    Inputs
    ------
    omega: float or array
        energy in eV
    k: float
        energy in eV
    method = "grid" (using the grid loaded in filename), "Lindhard" (free electron gas), "phonon" for phonon absorption data, or "optical" of electron absorption data
    """
    scalar_input = np.isscalar(omega)
    omega = np.atleast_1d(omega)

    if method=="grid":
        # If k is smaller than grid kmin, we will extrapolate from lowest k point
        # This is implemented by changing all small k values to kmin
        k = k*(k >= self.kmin) + self.kmin*(k < self.kmin)
        eps2=self.eps2_grid(omega,k)
    elif(method=="Lindhard"):
        eps2=[self.eps2_electrongas(om,k) for om in omega]
    elif(method=='phonon'):
      if(hasattr(self, "eps2_phonon")==False):
        print("Error, eps for phonon frequencies not loaded. Need to set phonon_filename.")
        return 0
      else:
        eps2=[self.eps2_phonon(om) for om in omega]
    elif(method=='optical'):
      if(hasattr(self, "eps2_electron_optical")==False):
        print("Error, eps for electron frequencies not loaded. Need to set eps_electron_optical_filename.")
        return 0
      else:
        eps2=[self.eps2_electron_optical(om) for om in omega]        
    else:
        print("Error, unknown method. Please choose 'grid' or 'Lindhard'")

    if(scalar_input):
      return eps2[0]
    else:
      return np.array(eps2)



# define function that returns the loss function Im(-1/eps)
def elf(self,omega,k,method="grid"):
    """
    Energy loss function Im(-1/eps(omega,k))

    Inputs
    ------
    omega: float or array
        energy in eV
    k: float
        energy in eV
    method = "grid" (using the grid loaded in filename), "Lindhard" (free electron gas), "phonon" for phonon absorption data, or "optical" of electron absorption data
    """
    return self.eps2(omega,k,method=method)/(self.eps1(omega,k,method=method)**2 + self.eps2(omega,k,method=method)**2)
  
