import numpy as np
from numpy import linspace, sqrt, array, pi, cos, sin, dot, exp, sinh, log, log10, cosh, sinh
from scipy.special import erf, erfc, gamma, gammaincc, exp1
from scipy.interpolate import interp1d, interp2d
from scipy import integrate
import sys, os

############################################################################################
#  Cross section reach for kg-yr exposure for multi-phonon excitations,
#  obtained by integrating the structure factor.
#  The anisotropic density of states is obtained from phonon_filename, and Fn is calculated from the DoS.


# There are nine important functions:

# Two internal functions for plotting differential rates without coherent single phonon:
#        _dR_domega_anisotropic, dR_dtheta_dphi_domega

# Compute rate without coherent single phonon:
#       R_multiphonons_anisotropic

# Structure factor using Fn files and impulse approximation:
#       structure_factor_anisotropic, impulse_approximation_anisotropic

# Two functions for computing required cross sections. One for overall exposure corresponding to
# 3 events/kg/yr and the other for a modulating signal at the 2 sigma level:
#       sigma_multiphonons_anisotropic, sigma_modulation_anisotropic

# One function that computes the number of events to observe a modulating signal at the 2 sigma level
#       num_events_modulation_anisotropic

# One function that computes the modulation amplitude R(t)/<R> for a fixed time and threshold
#       modulation_fraction_anisotropic


### Useful functions for multi phonon calculations

# We parameterize q in spherical as (q,theta,phi)
# The debye waller function returns an array of values, each component for each atom d.
def debye_waller_anisotropic(self,q,theta,phi):
    
    # Assuming inputs are scalar
    q_cart = np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
    D_d_q_array = np.einsum('i...,dijk,j...->dk...',q_cart,self.D_d_ij_tensor,q_cart)
    #D_d^q(omega)

    W_d_array = np.array(q**2 / (4*self.mN_vector) * np.trapz(D_d_q_array/self.phonon_DoS_anisotropic[0],self.phonon_DoS_anisotropic[0]))

    return W_d_array

# This function allows for vector inputs, and spits out a multidimensional debye waller array
# indexed as d,q,theta,phi
def debye_waller_vector_anisotropic(self,q,theta,phi):
    '''Debye Waller factor exp(-2 W(q)) where W(q) = q^2 omega / (4 A mp)
    Inputs
    ------
    q: float or array in units of eV. For each q, gives the Debye-Waller factor for each atom 
    theta: float or array
    theta: array
    phi: array
    '''
    if (isinstance(q,(np.ndarray,list)) and isinstance(theta,(np.ndarray,list)) and isinstance(phi,(np.ndarray,list))):
        num_d = len(self.atoms_anisotropic) #number of unique atoms
        th,ph = np.meshgrid(theta,phi,indexing='ij')
        q_cart = np.array([np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th)]) #vector q in cartesian

        D_d_q_array = np.moveaxis(np.einsum('i...,dijk,j...->dk...',q_cart,self.D_d_ij_tensor,q_cart),1,-1) 
        # swap axes to ensure indexed properly -- move (d,omega,theta,phi) to (d,theta,phi,omega)
        
        # indexing is (d, q, theta, phi)
        W_d_array = np.array([[qi**2 / (4*self.mN_vector[d]) * np.trapz(\
            D_d_q_array[d]/self.phonon_DoS_anisotropic[0],self.phonon_DoS_anisotropic[0]) for qi in q] for d in range(num_d)])

        return W_d_array
    elif(isinstance(q,(np.ndarray,float)) and isinstance(theta,(np.ndarray,float)) and isinstance(phi,(np.ndarray,float))):
        return self.debye_waller_anisotropic(q,theta,phi)
    else:
        print("Warning! debye_waller function given invalid quantity ")
        return 0.0
    

# (roughly) ISOTROPIC impulse approximation (Actually, using theta,phi = 0 for direction of q)
# returns array indexed as q, omega.
def impulse_approximation_anisotropic(self,q_grid,omega_grid,dark_photon=False):
    """
    Calculates the structure factor over a grid of omega inputs for a specified
    magnitude of q using the impulse approximation.
    Returns 2d array indexed as q, omega

    Inputs:
    -----------
    q_grid: array
        grid of magnitude of q
    omega_grid: array
        grid of omega to get structure factor values for
    """
    totalmass = self.mp * sum(self.Amult_anisotropic*self.Avec_anisotropic)
    volume = totalmass/self.rhoT * (1000*self.eVtokg)
    prefactor = 1/(volume) * self.eVcm**3 #volume in cm^3, so convert to 1/ev^3
    mass_array = self.mN_vector

    q,om,mass = np.meshgrid(q_grid,omega_grid,mass_array,indexing='ij')
    delta_squared = q**2 * self.omega_bar_anisotropic/(2*mass)
    # Effective coupling (cannot perform dark_photon calculations for now)

    if dark_photon:
        fd_squared = np.zeros((len(self.atoms_anisotropic),len(q_grid)))
        for k in range(len(self.atoms)):
            for i in range(len(self.atoms_anisotropic)):
                # Check to see if anisotropic atom name corresponds to isotropic
                # unit cell, then set fd corresponding dark fd_darkphoton file.
                # Basically a work-around to not make new fd_darkphoton_anisotropic specific
                # files.
                if(self.atoms_anisotropic[i].partition('_')[0] == self.atoms[k]):
                    fd_squared[i] = (self.fd_darkphoton[k](q_grid))**2 * self.Amult_anisotropic[i]
    else:
        fd_squared = np.tile((self.Avec_anisotropic)**2 *self.Amult_anisotropic,(len(q_grid),1)).T

    summation = np.einsum('dq,qwd->qw', fd_squared,np.sqrt(2*np.pi/delta_squared) \
                    * np.exp(-(om - (q**2 / (2*mass)))**2 /(2*delta_squared)))

    return prefactor * summation


# Computes structure factor values over specified q,theta,phi,omega grids. 
# Returns 4d array indexed as q,theta,phi,omega
#C_ld_anisotropic
def structure_factor_anisotropic(self,q_grid,theta_grid,phi_grid,omega_grid,min_n,max_n,dark_photon=False):
    """
    Calculates the structure factor over a grid of omega inputs for a specifed
    q (given in spherical), and summing over a specified range of n

    Inputs:
    -----------
    q_grid: array
        grid of magnitude of q
    theta_grid: array
        grid of theta coordinates of q in spherical
    phi_grid: array
        grid of phi coordinates of q in spherical
    omega_grid: array
        grid of omega to get structure factor values for
    min_n: integer
        lowest n to sum from
    max_n: integer
        highest n to sum to
    """
    
    # Ensure that omega_grid and q_grid are arrays. If they are floats, then make 1 element arrays.
    if(type(omega_grid)==float):
        omega_grid = np.array([omega_grid])
    if(type(q_grid)==float):
        q_grid = np.array([q_grid])

    num_d = len(self.atoms_anisotropic)
    totalmass = self.mp * np.sum(self.Amult_anisotropic*self.Avec_anisotropic)
    volume = totalmass/self.rhoT * (1000*self.eVtokg)
    prefactor = 2*np.pi/(volume) * self.eVcm**3 #volume in cm^3, so convert to 1/ev^3
    mass_array = self.mN_vector
    di,th,ph,om = np.meshgrid(np.arange(num_d),theta_grid,phi_grid,omega_grid,indexing='ij')

    F_n_d = np.array([self.Fn_interpolations_anisotropic[n]((di,th,ph,om)) for n in range(min_n,max_n+1)])
    # note default meshgrid is 'xy' ordering so ordering will be q n d here.
    n,q,m = np.meshgrid(np.arange(min_n,max_n+1),q_grid,mass_array)

    # Effective coupling (cannot perform dark_photon calculations for now)

    if dark_photon:
        fd_squared = np.zeros((len(self.atoms_anisotropic),len(q_grid)))
        for k in range(len(self.atoms)):
            for i in range(len(self.atoms_anisotropic)):
                # Check to see if anisotropic atom name corresponds to isotropic
                # unit cell, then set fd corresponding dark fd_darkphoton file
                if(self.atoms_anisotropic[i].partition('_')[0] == self.atoms[k]):
                    fd_squared[i] = (self.fd_darkphoton[k](q_grid))**2 * self.Amult_anisotropic[i]
    else:
        fd_squared = np.tile((self.Avec_anisotropic)**2 *self.Amult_anisotropic,(len(q_grid),1)).T


    summation = np.einsum('dqtp,qnd,dq,ndtpw -> qtpw',\
            np.exp(-2*self.debye_waller_vector_anisotropic(q_grid,theta_grid,phi_grid)),\
                (q**2 / (2*m))**n,fd_squared,\
                    F_n_d,optimize='optimal')

    return prefactor * summation



########################################################################################################
# Functions to compute the full rate

def sigma_multiphonons_anisotropic(self,t,threshold,sigman=1.e-38,N_ev=3,dark_photon=False):
    '''
    returns DM-proton cross-section [cm^2] corresponding to N_ev (default 3) events/kg/yr
    Inputs
    ------
    t: float
      time of day
    threshold: float
      experimental threshold, in eV
    sigman: float
      Default DM-nucleon cross section in [cm^2] of 1e-38
    N_ev: Number of events per kg per year. Default 3
    dark_photon: Choice of dark photon
    '''
    # factor of 3 to account for assumption of 3 events/kg-year
    rate = self.R_multiphonons_anisotropic(t,threshold,sigman,dark_photon)
    if rate != 0.0:
        return N_ev*sigman/rate
    else:
        return float('inf')


# Calculates full rate integral for given parameters by integrating over omega
def R_multiphonons_anisotropic(self,t,threshold,N_angular=40,sigman=1.e-38,dark_photon=False):
    """
    Returns rate for DM scattering with a harmonic lattice with full anisotropic structure factor, 
    including multiphonon contributions but excluding the coherent single phonon contribution

    Inputs
    ------
    t: float in [hr]
      time of day, between 0 and 24
    threshold: float in [eV]
    N_angular: float
        Number of theta,phi sampling points in integral. Default of 40.
    sigma_n: float
        DM-nucleon cross section in [cm^2], defined with respect to the reference momentum of q0. (q0 is specified by the 'update_params' function)
    Outputs
    -------
    rate as function of threshold, in [1/kg/yr]
    """
    n_omega_points = 200
    omega_max = 0.499999*self.mX*((self.veavg+self.vesc)**2) 

    # Was getting negative values inside sqrt of q limits with 0.5 coefficient -> used 0.4999
    n_min = max(int(threshold/self.dos_omega_range_anisotropic[1]),1) # Ensure that we don't consider n=0
    n_max = len(self.Fn_interpolations_anisotropic) #number of phonons precomputed

    # angular sampling points
    n_points_angular = N_angular

    theta_limit_lower = 0.
    theta_limit_upper = np.pi
    theta_range = np.linspace(theta_limit_lower,theta_limit_upper,n_points_angular)#Use 20 sampling points for theta

    phi_limit_lower = 0.
    phi_limit_upper = 2*np.pi
    phi_range = np.linspace(phi_limit_lower,phi_limit_upper,n_points_angular)#Use 20 sampling points for phi

    ### Use linear sampling for omega < max phonon energy, log sampling for omega > max phonon energy
    if(threshold < self.dos_omega_range_anisotropic[1]):

        omega_grid_linear = np.linspace(threshold,min([self.dos_omega_range_anisotropic[1],omega_max]),n_omega_points)
        
        integrand_vals_dR_domega_dphi = np.einsum('tpw,t->tpw',self.dR_dtheta_dphi_domega(theta_range,phi_range,omega_grid_linear,t,sigman,n_min,n_max,dark_photon),\
                                        np.sin(theta_range),optimize='optimal') #full integrand vals
        
        integrand_vals_dR_domega = np.trapz(integrand_vals_dR_domega_dphi,theta_range,axis=0) #integrate wrt theta
        integrand_values_linear = np.trapz(integrand_vals_dR_domega,phi_range,axis=0) #integrate wrt phi
        R_linear = np.trapz(integrand_values_linear,omega_grid_linear) #integrate wrt omega

    else:
        R_linear = 0.0
    if(omega_max > self.dos_omega_range_anisotropic[1]):

        omega_grid_log = np.logspace(max(np.log10(self.dos_omega_range_anisotropic[1]),np.log10(threshold)),np.log10(omega_max),n_omega_points)
        integrand_vals_dR_domega_dphi = np.einsum('tpw,t->tpw',self.dR_dtheta_dphi_domega(theta_range,phi_range,omega_grid_log,t,sigman,n_min,n_max,dark_photon),\
                                        np.sin(theta_range),optimize='optimal') #full integrand vals

        integrand_vals_dR_domega = np.trapz(integrand_vals_dR_domega_dphi,theta_range,axis=0) #integrate wrt theta
        integrand_values_log = np.trapz(integrand_vals_dR_domega,phi_range,axis=0) #integrate wrt phi
        R_log = np.trapz(integrand_values_log,omega_grid_log) #integrate wrt omega
    else:
        R_log = 0

    return R_linear + R_log

# Not used by R_multiphonons_anisotropic for speed, but can be called to calculate dR_domega plots
def _dR_domega_anisotropic(self,omega,t,sigman=1.e-38,dark_photon=False):

    
    #n_min = max(int(threshold/self.dos_omega_range_anisotropic[1]),1) # Ensure that we don't consider n=0
    n_min = 1
    n_max = len(self.Fn_interpolations_anisotropic) 

    theta_limit_lower = 0.
    theta_limit_upper = np.pi
    theta_range = np.linspace(theta_limit_lower,theta_limit_upper,20)#Use 20 sampling points for theta

    phi_limit_lower = 0.
    phi_limit_upper = 2*np.pi
    phi_range = np.linspace(phi_limit_lower,phi_limit_upper,20)#Use 20 sampling points for phi

    integrand_vals_dR_domega_dphi = np.einsum('tpw,t->tpw',self.dR_dtheta_dphi_domega(theta_range,phi_range,omega,t,sigman,n_min,n_max,dark_photon),\
                                    np.sin(theta_range),optimize='optimal') #full integrand vals
    
    integrand_vals_dR_domega = np.trapz(integrand_vals_dR_domega_dphi,theta_range,axis=0) #integrate wrt theta
    return np.trapz(integrand_vals_dR_domega,phi_range,axis=0)

# Calculates R(t)/<R> for a given time of day and energy threshold. Returns a scalar.
def modulation_fraction_anisotropic(self,t,threshold,sigman=1e-38,dark_photon=False):
    rate_t = self.R_multiphonons_anisotropic(t,threshold,sigman,dark_photon)
    
    time_array = np.linspace(0,24,10)
    rates = np.array([self.R_multiphonons_anisotropic(t,threshold,sigman,dark_photon) for t in time_array])
    avg_rate = np.trapz(rates,time_array,axis=0)/24
    return rate_t/avg_rate

# Compute N_ev - number of events to see a modulation at the 2 sigma level.
def num_events_modulation_anisotropic(self,threshold,sigman=1e-38,dark_photon=False):
    """
    Computes the expected exposure N required to observe a daily modulation using cosine approximation
    of the modulation.
    """
    
    time_array = np.linspace(0,24,10)
    rates = np.array([self.R_multiphonons_anisotropic(t,threshold,dark_photon) for t in time_array])
    avg_rate = np.trapz(rates,time_array,axis=0)/24

    rate_max = rates[0]
    A_mod = np.abs(rate_max - avg_rate)/avg_rate
    N_ev = 4/(A_mod**2)

    return N_ev

# gives cross section to see a modulation at a 2 sigma level for fixed mass and energy threshold
# note: assumes kg-year exposure
def sigma_modulation_anisotropic(self,threshold,sigman=1e-38,dark_photon=False):
    time_array = np.linspace(0,24,10)
    rates = np.array([self.R_multiphonons_anisotropic(t,threshold,sigman,dark_photon) for t in time_array])
    avg_rate = np.trapz(rates,time_array,axis=0)/24
    rate_max = rates[0]
    A_mod = np.abs(rate_max - avg_rate)/avg_rate
    N_ev = 4/(A_mod**2)

    return N_ev*sigman/avg_rate


#############################################################################
# Internal functions used by the above functions


# Calculate dR_domega_dtheta_dphi over a grid of q values. Returns a 3d array indexed by theta, phi, omega
def dR_dtheta_dphi_domega(self,theta_grid,phi_grid,omega,t,sigman,n_min,n_max,dark_photon=False):
    n_points = 200
    if omega[0] > self.omegaDMmax:
        return np.zeros((len(theta_grid),len(phi_grid),len(omega)))
    
    constant = self.eVtoInvYr* self.rhoX*self.eVcm**3 * sigman * np.pi \
        /((self.rhoT/(1000*self.eVtokg))*self.eVcm**3 * self.mX*self.eVtokg * \
          (2*np.pi)**3 * (self.muxnucleon**2)*self.eVcm**2)
    
    omega_min = omega[0]
    # integration limits on q
    q_limit_lower=  max(self.qBZ,self.qmin(omega_min))
    q_limit_upper = self.qmax(omega_min)
    q_impulse_approx = max(4*np.sqrt(2*self.mN_vector/self.omega_bar_inverse_anisotropic))
    
    if q_limit_lower >= q_limit_upper:

        return np.zeros((len(theta_grid),len(phi_grid),len(omega)))

    # Choose the largest value across all m_d values, and use theta,phi = 0 for omega_bar_inverse.

    # Recursive Structure Factor Part:
    if(n_min >= n_max):
        #Since F_n only computed to n=10, if given a high enough E_thresh, the recursive part
        # starts to throw problems. For high enough E_thresh, just use impulse approximation.
        recursive_part = np.zeros((len(theta_grid),len(phi_grid),len(omega))) 
    else:
        q_range_recursive = np.linspace(q_limit_lower,min(q_limit_upper,q_impulse_approx),n_points) #q integration range for F_n part
        integrand_vals = np.einsum('q,qtpw,qtpw ->qtpw',(q_range_recursive**2)*(self.Fmed_nucleus(q_range_recursive))**2,\
                            self.structure_factor_anisotropic(q_range_recursive,theta_grid,phi_grid,omega,n_min,n_max,dark_photon),\
                            kinematic_function_vector(self,q_range_recursive,theta_grid,phi_grid,omega,t),optimize='optimal')

        recursive_part = np.trapz(integrand_vals,q_range_recursive,axis=0) 

    # Impulse approximation part of integral:
    if(q_impulse_approx > q_limit_upper):
        impulse_part=0
    else:
        q_range_impulse = np.linspace(q_impulse_approx,q_limit_upper,n_points)
        integrand_values_impulse = np.einsum('q,qtpw,qw -> qtpw',q_range_impulse**2 *(self.Fmed_nucleus(q_range_impulse))**2,\
                                        kinematic_function_vector(self,q_range_impulse,theta_grid,phi_grid,omega,t),
                                        self.impulse_approximation_anisotropic(q_range_impulse,omega,dark_photon),optimize='optimal')
     
        impulse_part = np.trapz(integrand_values_impulse,q_range_impulse,axis=0)

    return constant * (recursive_part + impulse_part)

# Helper function for kinematic function g(q,omega,t)
def v_minus_vector(self,q_grid,theta,phi,omega,t):
    """
    ________________________________
    Inputs:
    q: float
        momentum transfer
    theta: float
        q theta coordinate in polar
    phi: float
        q phi coordinate in polar
    omega: float
        energy
    t: float
        time of day in hours. (between 0 and 24)
    """

    q,th,p,w = np.meshgrid(q_grid,theta,phi,omega,indexing='ij')
    # shape: (3, q, theta, phi, omega)
    q_hat = np.array([np.sin(th)*np.cos(p),np.sin(th)*np.sin(p),np.cos(th)])

    v_minus_val = np.einsum('iqtpw,i-> qtpw', q_hat,v_earth_t(self,t),optimize='optimal') + q/(2*self.mX) + w/q

    return np.minimum(v_minus_val,np.tile(self.vesc,(len(q),len(theta),len(phi),len(omega))))

# Vector form of earth velocity
def v_earth_t(self,t):
    sin_theta = np.sin((42/180)*np.pi)
    cos_theta = np.cos((42/180)*np.pi)
    sin_phi = np.sin(2*np.pi*(t/24))
    cos_phi = np.cos(2*np.pi*(t/24))
    return self.veavg * np.array([sin_theta*sin_phi,sin_theta*cos_theta*(cos_phi - 1),cos_theta**2 + (sin_theta**2 *cos_phi)])


# Kinematic function g(q,omega,t) as described in arxiv.org/abs/2102.09567
def kinematic_function_vector(self,q_grid,theta,phi,omega,t):
    """
    Kinematic function g. Returns multidimensional array indexed 
    q, theta, phi, omega
    """
    q,th,p,w = np.meshgrid(q_grid,theta,phi,omega,indexing='ij')
    N_0 = ((self.v0)**3)*np.pi*(np.sqrt(np.pi)*erf(self.vesc/self.v0) \
         - 2*(self.vesc/self.v0)*np.exp(-1*(self.vesc/self.v0)**2)) 
        #velocity distribution normalization constant in natural units
    prefactor = np.pi * (self.v0)**2 / (N_0 * q)

    return np.einsum('qtpw,qtpw -> qtpw',prefactor,(np.exp(-v_minus_vector(self,q_grid,theta,phi,omega,t)**2 / self.v0**2)) - \
            np.tile(np.exp(-self.vesc**2 / self.v0**2),(len(q_grid),len(theta),len(phi),len(omega))),optimize='optimal')
