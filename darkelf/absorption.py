import numpy as np

############################################################################################


def R_absorption(self,kappa=1e-15):
    """
    Returns rate for dark photon absorption, events per 1/kg/yr. Will use electronic ELF for mX > electronic band gap and phonon ELF otherwise. If the required ELF is not loaded or if mX is outside the range of the grid, the function will return zero.
    Inputs
    ------
    kappa: float
        kinetic mixing parameter for dark photon model
    Outputs
    -------
    absorption rate [1/kg/yr]
    """
    foo = 1.0/self.rhoT*self.rhoX*1e3*self.eVtoInvYr # eV/g -> counts/kg-year
    if self.mX < self.E_gap:
      if self.phonon_ELF_loaded:
        return kappa**2*foo*self.elf(self.mX,self.mX,method="phonon")
      else:
        return 0.0
    else:
      if self.electron_ELF_loaded:
        return kappa**2*foo*self.elf(self.mX,self.mX,method="grid")
      else:
        return 0.0
    