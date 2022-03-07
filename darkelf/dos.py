import numpy as np
from numpy import linspace, sqrt, array, pi, cos, sin, dot, exp, sinh, log, log10, cosh, sinh
from scipy.interpolate import interp1d, interp2d
from scipy import integrate
import sys, os, glob
import pandas as pd

############################################################################################

# Function to load density of states

# Note: If there are two distinct atoms,
# a partial density of states file is required for each atom

def load_phonon_dos(self,datadir,filename):

    if self.n_atoms == 2:
        # If there are two distinct atoms, dos_filename needs to be
        # set to two element list of filenames
        if len(filename) != 2:
            print("Warning! Density of states not loaded. dos_filename must be list of two partial DoS filenames")
            self.phonon_DoS_loaded=False
        else:
            partial_dos_1_path = datadir + self.target+'/'+ filename[0]
            partial_dos_2_path = datadir + self.target+'/'+ filename[1]

            if (not os.path.exists(partial_dos_1_path)) or (not os.path.exists(partial_dos_2_path)):
                print("Warning! Density of states not loaded. Need to set dos_filename for both atoms")
                self.phonon_DoS_loaded=False

            else:
                self.phonon_DoS_loaded=True
                dosdat_1 = np.loadtxt(partial_dos_1_path).T
                dosdat_2 = np.loadtxt(partial_dos_2_path).T
                print("Loaded " + filename[0] + " and " + filename[1] " for partial densities of states")
                self.phonon_DoS = [dosdat_1, dosdat_2]
                self.DoS_interp = [interp1d(self.phonon_DoS[i][0],self.phonon_DoS[i][1],kind='linear', fill_value = 0) for i in self.phonon_DoS]
                self.dos_omega_range = [[ phonon_DoS[i][0][0], phonon_DoS[i][0][-1] ] for i in self.phonon_DoS]
                self.omega_bar = [np.trapz(self.phonon_DoS[i][1]*self.phonon_DoS[i][0],
                                            x=self.phonon_DoS[0]) for i in self.phonon_DoS]
                self.omega_inverse_bar = [np.trapz(self.phonon_DoS[i][1]/self.phonon_DoS[i][0],
                                            x=self.phonon_DoS[0]) for i in self.phonon_DoS]

    else:
        dos_path = datadir + self.target+'/'+ filename

        if( not os.path.exists(dos_path)):
            print("Warning! Density of states not loaded. Need to set dos_filename ")
            self.phonon_DoS_loaded=False
        else:
            self.phonon_DoS_loaded=True
            dosdat = np.loadtxt(dos_path).T
            print("Loaded " + filename + " for density of states")
            self.phonon_DoS = dosdat
            self.DoS_interp = interp1d(self.phonon_DoS[0],self.phonon_DoS[1],kind='linear', fill_value = 0)
            self.dos_omega_range = [ dosdat[0][0], dosdat[0][-1] ]
            self.omega_bar = np.trapz(self.phonon_DoS[1]*self.phonon_DoS[0],
                                        x=self.phonon_DoS[0])
            self.omega_inverse_bar = np.trapz(self.phonon_DoS[1]/self.phonon_DoS[0],
                                    x=self.phonon_DoS[0])

    return

# Function to load Fn(omega) data corresponding to density of states file

def load_Fn(self,datadir,filename):


    if self.n_atoms == 2:
        partial_dos_path_1 = datadir + self.target+'/'+ filename[0]
        partial_fn_path_1 = datadir + self.target+'/'+ filename[0].replace('_pDoS','_Fn')
        partial_dos_path_2 = datadir + self.target+'/'+ filename[1]
        partial_fn_path_2 = datadir + self.target+'/'+ filename[1].replace('_pDoS','_Fn')

        if (not (os.path.exists(partial_dos_path_1) and os.path.exists(partial_dos_path_2))) and self.phonon_DoS_loaded:
            print("Warning! Fn(omega) functions not loaded. Need to calculate Fn(omega) from density of states")
            print("Use the function 'create_Fn_omega' to produce these files")
            self.phonon_Fn_loaded=False
        elif (not (os.path.exists(partial_dos_path_1) and os.path.exists(partial_dos_path_2))) and (not self.phonon_DoS_loaded):
            print("Warning! Fn(omega) functions not loaded. Need to load density of states")
            print("then use the function 'create_Fn_omega' to produce these Fn files")
            self.phonon_Fn_loaded=False
        else:
            self.phonon_Fn_loaded=True
            phonondat = [np.loadtxt(partial_fn_path_1).T, np.loadtxt(partial_fn_path_2).T]
            print("Loaded Fn(omega) functions corresponding to density of states in: ", filename[0], " ", filename[1])
            self.phonon_Fn = phonondat

        if self.phonon_Fn_loaded:
            # makes interpolations
            self.Fn_interpolations = {}
            for n in range(1, len(self.phonon_Fn)):
                self.Fn_interpolations[n] = [interp1d(self.phonon_Fn[i][0], self.phonon_Fn[i][n], fill_value=0) for i in self.phonon_Fn]


    else:
        dos_path = datadir + self.target+'/'+ filename
        fn_path = datadir + self.target+'/'+ filename.replace('_DoS','_Fn')

        if (not os.path.exists(fn_path)) and self.phonon_DoS_loaded:
            print("Warning! Fn(omega) functions not loaded. Need to calculate Fn(omega) from density of states")
            print("Use the function 'create_Fn_omega' to produce these files")
            self.phonon_Fn_loaded=False
        elif (not os.path.exists(fn_path)) and (not self.phonon_DoS_loaded):
            print("Warning! Fn(omega) functions not loaded. Need to load density of states")
            print("then use the function 'create_Fn_omega' to produce these Fn files")
            self.phonon_Fn_loaded=False
        else:
            self.phonon_Fn_loaded=True
            phonondat = np.loadtxt(fn_path).T
            print("Loaded Fn(omega) functions corresponding to density of states in: ", filename)
            self.phonon_Fn = phonondat

        if self.phonon_Fn_loaded:
            # makes interpolations
            self.Fn_interpolations = {}
            for n in range(1, len(self.phonon_Fn)):
                self.Fn_interpolations[n] = interp1d(self.phonon_Fn[0], self.phonon_Fn[n], fill_value=0)

    return
