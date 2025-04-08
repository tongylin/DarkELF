import numpy as np
from numpy import linspace, sqrt, array, pi, cos, sin, dot, exp, sinh, log, log10, cosh, sinh
from scipy.interpolate import interp1d, interp2d
from scipy import integrate
from scipy.special import erf, erfc
import pandas as pd
import sys, os
import yaml


class darkelf(object):
    def __init__(self, mX = 1e5, mMed = -1, vesckms = 500, v0kms = 220, vekms = 240, delta = 0.0, q0=0.0, gp_gn_ratio_val=1, gp_gn_ratio='g_n/g_p', nuclear_recoil=False,
        target='Ge',targetyaml='',filename="", phonon_filename="",
        eps_data_dir = os.path.dirname(__file__)+"/../data/",
        dos_filename="",fd_filename="",Zion_filename="",eps_electron_optical_filename=""):

        # Useful units and constants
        self.eVtoK = 11604.5221
        self.eVtoA0 = 1973.2
        self.eVtokg =  1.672621*1e-27/(0.94*1e9)
        self.eVcm = 1.9732*1e-5  # eV*cm
        self.mp = 0.94*1e9       # eV
        self.me = 511e3          # eV
        self.THztoeV = 0.00413566766 # per THz
        self.eVtoInvS = 1.5192674e15 # eV to 1/s
        self.yeartosec=86400 * 365.
        self.eVtoInvYr = self.eVtoInvS * self.yeartosec
        self.eVtokg=1.78e-36
        self.alphaEM = 1./137
        self.eEM = np.sqrt(self.alphaEM) # Note: cgs units! careful with particle phys conventions.
        self.eEMparticle = np.sqrt(4*np.pi*self.alphaEM)  # Particle phys convention for e
        self.A0tocm = 1e-8
        self.mom_autoeV=3728.95
        self.rhoX = 0.4e9  # eV/cm^3

        self.c0 = 2.99792458e5            # km/s
        self.c0cms = self.c0*1e5          # cm/s

        # read in various material-dependent quantities from yaml
        self.target = target
        if(targetyaml == ''):
            self.targetyaml = self.target
        else:
            self.targetyaml = targetyaml
        configfilename=eps_data_dir+'/'+target+'/'+self.targetyaml+'.yaml'
        if (not os.path.exists(configfilename)):
            print("Configuration file for "+target+" is absent: ", configfilename)
        else:
            with open(configfilename) as file:
                variable_list = yaml.load(file, Loader=yaml.FullLoader)
                for k, v in variable_list.items():
                    setattr(self, k, v)

        # !TL - dictionary of unique atoms in unit cell
        self.atoms = list(self.unitcell)
        # mass of atoms in unit cell
        self.Avec = np.array([self.unitcell[ai]['A'] for ai in self.atoms])
        self.Amult = np.array([self.unitcell[ai]['mult'] for ai in self.atoms])

        # Modified unit cell dictionary used for anisotropic calculations
        # Check that a material has anisotropic files available by checking if has unitcell_anisotropic attribute
        if hasattr(self,'unitcell_anisotropic'):
            self.atoms_anisotropic = list(self.unitcell_anisotropic)
            # mass of atoms in unit cell
            self.Avec_anisotropic = np.array([self.unitcell_anisotropic[ai]['A'] for ai in self.atoms_anisotropic])
            self.Amult_anisotropic = np.array([self.unitcell_anisotropic[ai]['mult'] for ai in self.atoms_anisotropic])
            self.mN_vector = self.Avec_anisotropic*self.mp
        else:
            print('Anisotropic calculations unavailable for ' + self.target)
        
        # Mass vector for atoms in unit cell, used for spin-dependent scattering only
        # This is updated in update_params to sum over the isotopes weighted by their mass fraction
        self.mvec = self.Avec * self.mp


        # !TL - these will become a vector more generally
        # nucleon mass used for Migdal calculation, regular nuclear recoils
        self.mN=self.A*self.mp
        self.NTkg = 5.9786e26/self.A  # number of targets per kg

        self.NUCkg = 5.9786e26 / sum(self.Avec * self.Amult) # number of unitcells per kg

        # Sound speeds in units of c
        self.cLA = self.cLAkms/self.c0

        # Brillouin zone in eV (isotropic approximation)
        self.qBZ = (2*pi/self.lattice_spacing)*self.eVtoA0

        # Parameters for electron gas approximation
        self.vF = pow(3*pi*self.omegap**2/(4*(1./137)*self.me**2), 1./3)
        self.kF = self.vF*self.me

        # select the default DOS
        if(dos_filename==""):
            dos_filename = [ai+'_pDoS.dat' for ai in self.atoms]
        elif(type(dos_filename) == str):
            dos_filename = [dos_filename]
        #if(dos_filename=="" and self.n_atoms==1):
        #  dos_filename=target+'_DoS.dat'
        #if(dos_filename=="" and self.n_atoms==2):
        #  dos_filename=[self.atoms[0]+'_pDoS.dat',self.atoms[1]+'_pDoS.dat']

        # select the default fd file (here taken to be fd for dark photon mediator)
        if(fd_filename==""):
            fd_filename = [ai+'_atomic_Zion.dat' for ai in self.atoms]
        elif(type(fd_filename) == str):
            fd_filename = [fd_filename]
        #if(fd_filename=="" and self.n_atoms==1):
        #  fd_filename=target+'_atomic_Zion.dat'
        #if(fd_filename=="" and self.n_atoms==2):
        #  fd_filename=[self.atoms[0]+'_atomic_Zion.dat',self.atoms[1]+'_atomic_Zion.dat']


        # File for electron-regime ELF
        if(filename==""):
          if(target=="Ge" or target=="Si"):
            filename=target+"_gpaw_withLFE.dat"
          else:
            filename=target+"_mermin.dat"

        # File for phonon-regime ELF
        if(phonon_filename==""):
          if(target=="Al2O3"):
            phonon_filename="Al2O3_epsphonon_o.dat"
          elif(target=="GaAs"):
            phonon_filename="GaAs_epsphonon_data10K.dat"
          elif(target=="GaN"):
            phonon_filename="GaN_epsphonon_300K.dat"
          elif(target=="Ge"):
            phonon_filename="Ge_epsphonon_data2K.dat"
          elif(target=="Si"):
            phonon_filename="Si_epsphonon_data6K.dat"
          elif(target=="SiC"):
            phonon_filename="SiC_epsphonon_3C.dat"
          elif(target=="ZnS"):
            phonon_filename="ZnS_epsphonon_300K.dat"
          else:
            phonon_filename="none" # no default phonon_filename

        # File for Zion used in Migdal calculation
        # !TL -- there is only an if statement for GaAs? For some of the other multi-atom unit cells
        #    we are using a different naming convention for the Zion files where we label it
        #    as, e.g. GaN_Zion.dat which may be confusing. For SiC, SiO2, we don't provide a file.
        #    To be updated when Migdal updated to multi-atom.
        if Zion_filename=="":
          if target=="GaAs":
            Zion_filename=target+"Ga_Zion.dat" # for now only using Ga in Migdal, needs to be generalized.
          else:
            Zion_filename=target+"_Zion.dat"
        
        if eps_electron_optical_filename=="":
            eps_electron_optical_filename=self.target+"_eps_electron_opticallimit.dat"  

        # Set parameters that load data files
        self.filename = filename
        self.phonon_filename = phonon_filename
        self.eps_data_dir = eps_data_dir
        self.dos_filename = dos_filename
        self.fd_filename = fd_filename
        self.Zion_filename = Zion_filename
        self.eps_electron_optical_filename=eps_electron_optical_filename

        # Default is to use tabulated dielectric functions, assuming they are available.
        print(" .... Loading files for " + self.target)

        # Load epsilon data in electron regime
        self.load_epsilon_grid(self.eps_data_dir,self.filename)
        # Load epsilon data in phonon regime
        self.load_epsilon_phonon(self.eps_data_dir,self.phonon_filename)
        # Load epsilon data for electrons in optical limit
        self.load_eps_electron_opticallimit(self.eps_data_dir,self.eps_electron_optical_filename)
        # Load Atomic Migdal calculation from Ibe et al.
        self.load_Migdal_FAC(self.eps_data_dir)
        # Load momentum dependent effective ion charge Zion(k) for Migdal
        self.load_Zion(self.eps_data_dir)
        # Load momentum dependent effective ion charge Zion(k) for multiphonon (REDUNDANCY, need to fix)
        self.load_fd_darkphoton(self.eps_data_dir,self.fd_filename)
        # load density of states
        self.load_phonon_dos(self.eps_data_dir,self.dos_filename)
        # Load precomputed Fn(omega) functions for multiphonon calculation
        self.load_Fn(self.eps_data_dir,self.dos_filename)
        # tabulate the shake-off probability for the Migdal calculation
        self.tabulate_I()

        # Only load anisotropic files if available for material
        if hasattr(self,'unitcell_anisotropic'):
            # load anisotropic density of states
            self.load_phonon_dos_anisotropic(self.eps_data_dir)
            # Load precomputed Fn(omega) functions for anisotropic multiphonon calculation
            self.load_Fn_anisotropic(self.eps_data_dir)

        # Set parameters that depend on DM properties
        self.update_params(mX=mX,delta=delta,setdelta=True,mMed=mMed,vesckms=vesckms,v0kms=v0kms,vekms=vekms,q0=q0,
                            gp_gn_ratio_val=gp_gn_ratio_val, gp_gn_ratio=gp_gn_ratio)

        # Characteristic momenta where many phonons become important (take maximum if two distinct atoms)
        if hasattr(self, 'omega_bar'):

            self.qchar = max([2*self.Avec[i]*self.mp*self.omega_bar[i] for i in range(len(self.atoms))] )
        else:
            self.qchar = 0

        # Set single-coherent phonon bounding box: if LOvec exists, set to LOvec
        # If not, set to edge of single-ph density of states

        if hasattr(self, 'LOvec'):
            self.omega1ph_max = max(self.LOvec)
        else:
            self.omega1ph_max = self.dos_omega_range[-1]




    ############################################################################################

    from .epsilon import load_epsilon_grid, load_epsilon_phonon, load_eps_electron_opticallimit 
    from .epsilon import load_Zion
    from .epsilon import eps1_electrongas, eps1, eps2_electrongas, eps2, elf

    from .fnomega import load_phonon_dos, load_Fn, create_Fn_omega
    from .fnomega import C_ld, debye_waller

    from .multiphonon_spin_independent import sigma_multiphonons_SI, R_multiphonons_SI, R_single_phonon
    from .multiphonon_spin_independent import _R_single_optical, _R_single_acoustic, _dR_domega_coherent_single
    from .multiphonon_spin_independent import _dR_domega_multiphonons_SI, _R_multiphonons_prefactor_SI
    from .multiphonon_spin_independent import load_fd_darkphoton

    from .multiphonon_spin_dependent import sigma_multiphonons_SD, R_multiphonons_SD
    from .multiphonon_spin_dependent import _dR_domega_multiphonons_SD, _R_multiphonons_prefactor_SD
    from .multiphonon_spin_dependent import _dR_domega_nuclear_recoil_SD

    from .fnomega_anisotropic import load_phonon_dos_anisotropic,F_n_d_precompute_anisotropic,F_n_omega_anisotropic
    from .fnomega_anisotropic import load_Fn_anisotropic, omega_bar_anisotropic_func, omega_bar_inverse_anisotropic_func

    from .multiphonon_anisotropic import debye_waller_anisotropic, debye_waller_vector_anisotropic, impulse_approximation_anisotropic
    from .multiphonon_anisotropic import structure_factor_anisotropic, R_multiphonons_anisotropic, sigma_multiphonons_anisotropic
    from .multiphonon_anisotropic import dR_dtheta_dphi_domega, kinematic_function_vector, _dR_domega_anisotropic
    from .multiphonon_anisotropic import modulation_fraction_anisotropic, num_events_modulation_anisotropic, sigma_modulation_anisotropic


    from .electron import R_electron, dRdomega_electron, dRdomegadk_electron
    from .electron import electron_yield, dRdQ_electron

    from .phonon import R_phonon_Frohlich, _R_phonon_Frohlich_branch
    from .phonon import R_phonon, dRdomega_phonon, dRdomegadk_phonon

    from .Migdal import load_Migdal_FAC, _I, _J, _incomErf
    from .Migdal import dPdomega, dPdomegadk, dRdEn_nuclear, dRdomega_migdal, R_migdal, tabulate_I

    from .absorption import R_absorption


    ############################################################################################

    def DM_params(self):
        """
        Show dark matter parameters currently used in the class.
        """
        print("mX = " + str(self.mX) + " eV")
        print("mMed = " + str(self.mMed) + " eV")
        print("delta = " + str(self.delta) + " eV")
        return


    # Function to update various DM model parameters
    # Initial call by class will set params to avoid 0 values! After that, update params
    #    by specifying the parameter to update
    def update_params(self, mX = 0, delta = 0, setdelta=False, mMed = -1,
                        vesckms = 0, v0kms = 0, vekms = 0, mediator = '', q0 = 0.0,
                        SD_op = "A'", gp_gn_ratio_val = 1, set_gp_gn_ratio_val = False,
                        gp_gn_ratio = 'g_n/g_p'):
        """
        Function to update dark matter parameters used in the class.
        If the value is set to zero or not set in the arguments, then that means no changes.

        Inputs
        ------
        mX: float
            Mass in [eV]
        delta: float
            Inelastic splitting in DM states [eV]
            Must set setdelta=True in this case to get delta=0
            Currently just used for DM-electron scattering
        mMed: float
            DM-SM mediator mass in eV
            If zero or not set, then default is mMed = mX (massive mediator for NR scattering)
        mediator: string
            options: "massless" or "massive"
            setting mMed overrides this option
        vesckms: float
            Set vesc in units of km/s
        v0kms: float
            Set vesc in units of km/s
        vekms: float
            Set vesc in units of km/s
        mediator: string 'massive' or 'massless'
            Specifies whether the massive or massless mediator limit is used. This flag is not used if the mediator mass is specified explicitly with the "mMed" flag
        q0: float
            choice of reference momentum to define the NR mediator form factor. Default is mX*v0 if not specified
        SD_op: string "phi" or "a" or "A'"
            Used in multiphonon_spin_dependent.py
            Selects an EFT operator from those implemented.
        gp_gn_ratio_val: float
            Used in multiphonon_spin_dependent.py
            Specifies the numerical ratio of g_p and g_n (relevant if not using the Odd Group Model for f_n)
        gp_gn_ratio: string 'g_n/g_p' or 'g_p/g_n'
            Used in multiphonon_spin_dependent.py
            Specifies whether the ratio has g_p or g_n as the numerator.
        """

        if(mX > 0):
            self.mX = mX
        if(setdelta or delta !=0):
            self.delta = delta

        # stuff for DM velocity distribution
        self.c0 = 2.99792458e5            # km/s
        self.c0cms = self.c0*1e5          # cm/s

        if(v0kms > 0):
            self.v0 = v0kms/self.c0   # km/s
        if(vesckms > 0):
            self.vesc = vesckms/self.c0 # km/s
        if(vekms > 0):
            self.veavg = vekms/self.c0   # km/s

        self.vmax = self.vesc+ self.veavg

        self.zz = self.vesc/self.v0
        zz = self.zz
        self.Nfv = (erf(zz) - 2*zz*exp(-zz*zz)/sqrt(pi))*pow(pi,1.5)*self.v0**3
        self.omegaDMmax = self.mX/2.0*(self.vesc + self.veavg)**2

        # Reduced mass
        self.muxN = self.mX*self.mN/(self.mX + self.mN)
        self.muxnucleon = self.mX*self.mp/(self.mX + self.mp)

        self.muXe = mX*self.me/(mX + self.me)

        # reference momentum used in mediator form factor for nuclear recoils
        if(q0==0.0):
            self.q0=self.mX*self.v0
        else:
            self.q0 = q0

        # Set massless or massive limit for mediator
        if(mediator == "massive"):
            self.mMed = 100*self.mX
        elif(mediator == "massless"):
            self.mMed = 0.0
        else:
            # Default mediator mass is equal to 100*mX, for massive mediator
            self.mMed = 100*self.mX

        # If mMed is numerically selected, then overrides the above
        if(mMed >= 0):
            self.mMed = mMed

        # Settings for spin-dependent nuclear scattering

        # sets the DM-Nucleon operator
        self.SD_op = SD_op

        # Sets the factor that is isotope averaged (f_d^2 <S^2> / m_d^2) based
        # on the ratio of g_p and g_n and which one is on top
        if self.unitcell[self.atoms[0]].get('isotopes') != None:
            if (self.SD_op == "phi" or self.SD_op == "a") and not set_gp_gn_ratio_val:
                def gn_over_gp(ma):
                    # !TL: fa = 1 TeV? Values are slightly off in commented code? 0.938/1e3 * 0.44 = 0.00041272, same for second coeff
                    #      ma in units of GeV [--> Updated to eV to be consistent with the rest of the code]
                    #gp_withoutcGG =  np.abs(0.000397682 - (7.88551*10**(-6))/(ma**2 - 0.0182188))
                    #gn_withoutcGG = np.abs(0.000398669 + (7.90508*10**(-6))/(ma**2 - 0.0182188))
                    maMeV = ma  / 1e6
                    gp_withoutcGG =  np.abs(0.00041272 - (8.07284081*10**(-6))/(maMeV**2 - 0.0182188))
                    gn_withoutcGG = np.abs(0.00039712 + (8.07284081*10**(-6))/(maMeV**2 - 0.0182188))
                    return gn_withoutcGG/gp_withoutcGG

                if gp_gn_ratio == 'g_n/g_p':
                    gp_gn_ratio_val = gn_over_gp(mMed)
                elif gp_gn_ratio == 'g_p/g_n':
                    gp_gn_ratio_val = 1 / gn_over_gp(mMed)
                else:
                    print('Invalid choice for gp_gn_ratio - must be g_n/g_p or g_p/g_n ')

            # mass vec accounting for isotopes.
            # !TL - technically, one should maybe average over lambda_d^2 * Jd (Jd + 1) * md^3 FIRST. 
            #       Given that we are not doing this and the corrections are small, we can also remove this for simplicity
            #       and just use mvec -> Avec * mp
            self.mvec = np.zeros(len(self.atoms))
            # vector containing averaged lambda_d^2 * Jd (Jd + 1) 
            self.isotope_averaged_factors = np.zeros(len(self.atoms))
            for i, ai in enumerate(self.atoms):
                for j in range(len(self.unitcell[ai]['isotopes'])):
                    frac = self.unitcell[ai]['isotopes'][j]['frac']
                    atomic_spin = self.unitcell[ai]['isotopes'][j]['atomic_spin']

                    if gp_gn_ratio == 'g_n/g_p':
                        lambda_d = self.unitcell[ai]['isotopes'][j]['f_p'] + gp_gn_ratio_val * self.unitcell[ai]['isotopes'][j]['f_n']
                    elif gp_gn_ratio == 'g_p/g_n':
                        lambda_d = self.unitcell[ai]['isotopes'][j]['f_n'] + gp_gn_ratio_val * self.unitcell[ai]['isotopes'][j]['f_p']
                    else:
                        print('Invalid choice for gp_gn_ratio - must be g_n/g_p or g_p/g_n ')
                        # In the Odd Group Model, only one of f_n or f_p will be nonzero, so the
                        # ratio of g_n and g_p should not matter. In the shell model, that is not
                        # necessarily the case.
                    self.mvec[i] += self.mp * frac * self.unitcell[ai]['isotopes'][j]['A']

                    self.isotope_averaged_factors[i] += frac * (atomic_spin * (1. + atomic_spin)) * lambda_d**2

            self.gp_gn_ratio = gp_gn_ratio
            self.gp_gn_ratio_val = gp_gn_ratio_val
        return


    def Fmed_nucleus_SI(self,q):
        return (self.q0**2 + self.mMed**2)/(q**2 + self.mMed**2)

    def Fmed_nucleus_SD(self, q):
        if (self.SD_op == "phi") or (self.SD_op == "a") or (self.SD_op == "A'") or (self.SD_op == "double A'"):
            return (self.q0**2 + self.mMed**2)/(q**2 + self.mMed**2)
        else:
            raise Exception("This spin dependent operator has not yet been defined")

    def Fmed_electron(self,q):
        return ((self.alphaEM*self.me)**2 + self.mMed**2)/(q**2 + self.mMed**2)

    # return (massless) dark photon cross section in terms of Qx
    def sigmaebar(self,Qx=1e-9,mX=0):
        if(mX==0):
            mX = self.mX
        return 16*pi*self.muXe**2*self.alphaEM**2*Qx**2/(self.alphaEM*self.me)**4 * self.eVcm**2


    ############################################################################################
    ########################### Dark matter velocity distributions #############################

    # boosted velocity integrand for isotropic approx - dimensionless
    def _fv_1d_scalar(self, v):
        if(v > self.vesc + self.veavg):
            return 0
        elif(v > self.vesc - self.veavg):
            a = 2*v*self.veavg/self.v0**2
            xmax = (self.vesc**2 - v**2 - self.veavg**2)/(2*v*self.veavg)
            return (2*pi*v**2/self.Nfv) * exp(- (v**2 + self.veavg**2)/self.v0**2 ) * (exp(a) - exp(-a*xmax))*1.0/a
        else:
            a = 2*v*self.veavg/self.v0**2
            return (2*pi*v**2/self.Nfv) * exp(- (v**2 + self.veavg**2)/self.v0**2 ) * 2*sinh(a)/a

    # velocity integrant to be called
    def fv_1d(self,v):
        """
        DM speed distribution in the lab (i.e., integrating fDM(v) over angles)

        Inputs
        ------
        v: float or array
            v in units where c = 1
        """
        if(isinstance(v,(np.ndarray,list)) ):
            return np.array([self._fv_1d_scalar(vi) for vi in v])
        elif(isinstance(v,float)):
            return self._fv_1d_scalar(v)
        else:
            print("Warning! fv function given invalid quantity ")
            return 0.0

    # vmin(omega, k) in units of c
    def vmin(self,omega,k):
        return omega/k + k/(2*self.mX) + self.delta/k

    # eta(v) function, acts only on scalar values
    def _etav_scalar(self,vmini):
        if(vmini > self.vesc+self.veavg):
            return 0
        elif(vmini > self.vesc-self.veavg):
            return pi*self.v0**2/(2*self.veavg * self.Nfv) * ( -2*exp(-self.vesc**2/self.v0**2)*(self.veavg + self.vesc - vmini) + \
                                                             np.sqrt(pi)*self.v0*(erf(self.vesc/self.v0) + erf((self.veavg - vmini)/self.v0) ) )
        else:
            return pi*self.v0**2/(2*self.veavg * self.Nfv) * ( -4*exp(-self.vesc**2/self.v0**2)*self.veavg + \
                np.sqrt(pi)*self.v0*erf( (self.veavg - vmini)/self.v0) + np.sqrt(pi)*self.v0*erf( (self.veavg + vmini)/self.v0) )

    # eta(v) function designed to be called
    def etav(self,vmini):
        """
        Integral of d^3v fDM(v)/v Theta(v - vmin)

        Inputs
        ------
        vmini: float or array
            vmin in units where c = 1
        """
        if(isinstance(vmini,(np.ndarray,list)) ):
            return np.array([self._etav_scalar(vminii) for vminii in vmini])
        elif(isinstance(vmini,float)):
            return self._etav_scalar(vmini)
        else:
            print("Warning! etav function given invalid quantity ")
            return 0.0

    # Minimum and maximum allowed q values (TOTAL momentum transfer), given omega (energy deposited) and other DM params
    def qmin(self,omega):
        if( omega + self.delta < self.omegaDMmax):
            return self.mX*self.vmax - np.sqrt(self.mX**2*self.vmax**2 \
                - 2 * (omega + self.delta) * self.mX)
        else:
            return 0

    def qmax(self,omega):
        if( omega + self.delta < self.omegaDMmax):
            return self.mX*self.vmax + np.sqrt(self.mX**2*self.vmax**2  \
                - 2 * (omega + self.delta) * self.mX)
        else:
            return 0
