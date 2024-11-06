import numpy as np
from numpy import linspace, sqrt, array, pi, cos, sin, dot, exp, sinh, log, log10, cosh, sinh
from math import factorial
from scipy.special import erf, erfc, gamma, gammaincc, exp1
from scipy.interpolate import interp1d, interp2d, RegularGridInterpolator
from scipy import integrate
#import vegas
#import gvar
import time
import sys, os, glob
import pandas as pd


##############################################################################
# Function to load Fn(omega) data corresponding to density of states file
def load_Fn_anisotropic(self,datadir=None,dos_filename=None):
    if(datadir == None):
        datadir = self.eps_data_dir
    dos_filename = self.target + '_pDoS_anisotropic.dat' 

    file = dos_filename[0] #assume filename is 1 element array
    print(dos_filename)
    Fn_path = datadir + self.target+'/'+ dos_filename.replace('_pDoS_anisotropic','_Fn').replace('.dat','.npy') 
    print(Fn_path)
    if not os.path.exists(Fn_path):
        print(f"Warning! {dos_filename.replace('_pDoS','_Fn').replace('.dat','.npy')} does not exist! Need to calculate Fn(omega) from DoS. Use the function 'F_n_d_precompute' to produce these files ")
        return
    else:
        self.phonon_Fn_anisotropic = np.load(Fn_path) #dos_filename is a .npy file with a 5 dim array
        print("Loaded " + dos_filename.replace('_pDoS','_Fn').replace('.dat','.npy')  + " for Fn(omega)")

    
    # length of each dimension in F_n array. Used to compute dictionary below
    phonons = len(self.phonon_Fn_anisotropic)
    num_d = len(self.D_d_ij_tensor) #number of atoms
    num_theta = len(self.phonon_Fn_anisotropic[0][0])
    num_phi = len(self.phonon_Fn_anisotropic[0][0][0])
    num_omega = len(self.phonon_Fn_anisotropic[0][0][0][0])

    # These match the arrays used to precompute F_n
    theta = np.linspace(0,np.pi,num_theta)
    phi = np.linspace(0,2*np.pi,num_phi)
    omega = np.linspace(self.dos_omega_range_anisotropic[0],
                                (phonons/2)*self.dos_omega_range_anisotropic[1], num_omega)

    # dictionary for Fn functions in terms of number of phonons (NO offset of index to phonon number, ie Fn_interpolations[3] is N=3 phonons)
    self.Fn_interpolations_anisotropic = {}

    for n in range(len(self.phonon_Fn_anisotropic)):
        self.Fn_interpolations_anisotropic[n+1] = RegularGridInterpolator((np.arange(num_d),theta,phi,omega),self.phonon_Fn_anisotropic[n],fill_value=0,bounds_error=False)
    
    return

#Precompute a 5d array containing F_n values for different directions and atoms
# Indexed as: n, d, theta, phi, omega
def F_n_d_precompute_anisotropic(self,datadir=None,dos_filename=None,phonons=20,npoints_omega=1000,npoints_theta=40,npoints_phi=40):
    if(datadir == None):
        datadir = self.eps_data_dir
    if(dos_filename == None):
        dos_filename = self.target + '_pDoS_anisotropic.dat' 


    theta_grid = np.linspace(0,np.pi,npoints_theta)
    phi_grid = np.linspace(0,2*np.pi,npoints_phi)

    F_n_d_array = np.moveaxis(np.array([[F_n_omega_anisotropic(self,theta,phi,datadir,dos_filename,phonons,npoints_omega,)\
                        for phi in phi_grid] for theta in theta_grid]),[0,1],[2,3]) #move axis to index as n,d,theta,phi,omega
    
    fn_path = datadir + self.target+'/'+ dos_filename.replace('_pDoS_anisotropic','_Fn').replace('.dat','.npy')

    print(fn_path)
    np.save(fn_path,F_n_d_array)
    return

#Calculate Tn and Fn for given DOS data and specified direction q_hat 
def F_n_omega_anisotropic(self,theta,phi,datadir=None, dos_filename=None, phonons = 10, npoints_omega=1000):
    """
    Function to create an array of Fn values for a given material.

    Uses recursive relation on Tn = n! * Fn and then divides by n! at the end for Fn

    Inputs
    ------
    theta: float
        theta value in q hat
    phi: float
        phi value in q hat
    datadir: string
        directory with DoS file, default is self.eps_data_dir with all the other data
    dos_filename: list of strings
        DoS filename(s), default is self.dos_filename which is set when the class is instantiated
    phonons: int
        specifies up to how many phonons Fn is calculated for. Default value is 10.
    npoints_omega: int
        number of omega points to compute Fn grid, default is 1000
    """

    # omega range for Fn files (determined by DoS range) - this could be expanded as needed.
    omegarange = np.linspace(self.dos_omega_range_anisotropic[0],
                                (phonons/2)*self.dos_omega_range_anisotropic[1], npoints_omega)
    
    omega = self.phonon_DoS_anisotropic[0] #omega values in original pDOS data
 
    q = np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)]) #cartesian q.
    num_d = len(self.D_d_ij_tensor) #number of atoms

    D_d_q = np.einsum('i...,dijk,j...->dk...',q,self.D_d_ij_tensor,q) # projected density of states tensor in direction q hat

    T_1 = np.array([D_d_q[d]/omega for d in range(num_d)])

    T_1_func = {}
    for d in range(num_d):
        T_1_func[d] = interp1d(omega,T_1[d],fill_value=0,bounds_error=False)

    T_n_array = np.zeros((phonons,num_d,len(omegarange)))
    for d in range(num_d):
        T_n_array[0][d]=T_1_func[d](omegarange)   
         # Will add T_2,T_3, etc....

    if (phonons == 1):
        return T_n_array
    else:
        for n in range(1,phonons):
            T_n_minus_1_func = {}
            for d in range(num_d):
                T_n_minus_1_func[d] = interp1d(omegarange,T_n_array[n-1][d],fill_value=0,bounds_error=False)
                
                T_n_array[n][d] = np.array([[(np.trapz(np.multiply(T_1[d],\
                        T_n_minus_1_func[d](np.tile(W,len(omega))-omega)),\
                            omega)) for W in omegarange]])

    F_n_array = np.array([T_n_array[n]/(factorial(n+1)) for n in range(len(T_n_array))])

    return F_n_array



############################################################################################

# Function to load density of states and create D_d_ij tensor
def load_phonon_dos_anisotropic(self,datadir):
    # Load pDoS data
    file = self.target + '_pDoS_anisotropic.dat' 
    # Assuming only one composite file for anisotropic case

    dos_path = datadir + self.target+'/'+ file
    if not os.path.exists(dos_path):
        print(f"Warning, {dos_path} does not exist! Density of states not loaded. Need to set dos_filename.")
    else:
        self.phonon_DoS_anisotropic = np.loadtxt(dos_path).T
        print("Loaded " + dos_path + " for partial densities of states")

    #DOS data in form omega, [pDOS(x,y,z,x+y,x+z,y+z)] for each non-redudant atom

    self.dos_omega_range_anisotropic = np.array([ self.phonon_DoS_anisotropic[0][0], self.phonon_DoS_anisotropic[0][-1] ])
    # Warning: assuming same omega range for all pDOS!
    #----------------------------------------------------------------------
    # Create D_d_ij tensor

    data = self.phonon_DoS_anisotropic
    num_d = int((len(data)-1)/6)
    num_omega_vals = len(data.T)

    D_d_ij = np.zeros(shape=(num_d,3,3,num_omega_vals)) #Create empty array for tensor

    # Add components
    for d in range(num_d):
        xx = data[6*d +1]
        yy =data[6*d +2]
        zz= data[6*d +3]
        x_plus_y = data[6*d +4]
        x_plus_z = data[6*d +5]
        y_plus_z = data[6*d +6]

        #diagonals
        D_d_ij[d][0][0]=xx #xx
        D_d_ij[d][1][1]=yy #yy
        D_d_ij[d][2][2]=zz #zz

        #off diagonals
        D_d_ij[d][0][1] = x_plus_y - 0.5*(xx + yy) #xy
        D_d_ij[d][1][0] = D_d_ij[d][0][1] #xy
        D_d_ij[d][0][2] = x_plus_z - 0.5*(xx+zz) #xz
        D_d_ij[d][2][0] = D_d_ij[d][0][2] #xz
        D_d_ij[d][1][2] = y_plus_z - 0.5*(yy+zz)#yz
        D_d_ij[d][2][1] = D_d_ij[d][1][2] #yz

    self.D_d_ij_tensor = D_d_ij

     # Default set omega bar and omega bar inverse as along z axis.
    # Two functions exist below to calculate for arbitrary directions.
    self.omega_bar_anisotropic = self.omega_bar_anisotropic_func(0.,0.)
    self.omega_bar_inverse_anisotropic = self.omega_bar_inverse_anisotropic_func(0.,0.)

    return 

# Function to get pDOS for a specified q direction
# NOTE: It returns an array of d arrays, one for each atom in the unit cell.
def D_d_q(self,q):

    #q = q/np.linalg.norm(q) #Ensure q is normed
    D_d_q = np.einsum('i...,dijk,j...->dk...',q,self.D_d_ij_tensor,q)
    
    return D_d_q

# Compute omega_bar for given material in specified q direction
def omega_bar_anisotropic_func(self,theta,phi):
    omega = self.phonon_DoS_anisotropic[0]

    #If given array inputs, turn into multidimensional meshgrid. Makes q array below work
    if(type(theta)==np.ndarray and type(phi) == np.ndarray):
        theta,phi = np.meshgrid(theta,phi)

    q = np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
    directional_DOS = np.einsum('i...,dijk,j...->dk...',q,self.D_d_ij_tensor,q)
    return np.array(np.trapz(np.einsum('do...,o->do...',directional_DOS,omega),omega,axis=1)) # axis 0 is d, axis 1 is omega 

# Compute omega_bar_inverse for given material in specified q direction
def omega_bar_inverse_anisotropic_func(self,theta,phi):
    omega = self.phonon_DoS_anisotropic[0]
    
    #If given array inputs, turn into multidimensional meshgrid. Makes q array below work
    if(type(theta)==np.ndarray and type(phi) == np.ndarray):
        theta,phi = np.meshgrid(theta,phi)

    q = np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
    directional_DOS = np.einsum('i...,dijk,j...->dk...',q,self.D_d_ij_tensor,q)

    return np.array(np.trapz(np.einsum('do...,o->do...',directional_DOS,1/omega),omega,axis=1)) # axis 0 is d, axis 1 is omega 
