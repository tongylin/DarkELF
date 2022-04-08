import numpy as np
from numpy import linspace, sqrt, array, pi, cos, sin, dot, exp, sinh, log, log10, cosh, sinh
from scipy.special import erf, erfc, gamma, gammaincc, exp1
from scipy.interpolate import interp1d, interp2d
from scipy import integrate
import sys, os, glob


# Note!! These all assume zincblende crystal structure, which should be fine
# for materials that we currently perform multiphonon calculations for

############################################################################################
#  Cross section reach for kg-yr exposure for for multi-phonon excitations,
#  obtained by integrating the structure factor with approximations as in (paper)
#  density of states is obtained from phonon_filename,
#  fn must be calculated from density of states
#  important parameters from yaml files


# There are five important functions:

# Two internal functions for plotting differential rates without coherent single phonon and coherent single phonon only:
#        _dR_domega_multiphonons_no_single, _dR_domega_coherent_single

# Two complete rates without coherent single phonon and coherent single phonon only:
#       R_multiphonons_no_single, R_single_phonon

# and the final reaches corresponding to 3 events/kg/yr:
#       sigma_multiphonons



### Useful functions for multiphonon calculations
def _debye_waller_scalar(self, q):
    # Debye-Waller factor set to 1 when q^2 small relative to characteristic q, for numerical convenience

    if self.n_atoms == 1:

        one_over_q2_char = self.omega_inverse_bar/(2*self.A*self.mp)

        if (one_over_q2_char*q**2 < 0.03):
            return 1
        else:
            return exp(-one_over_q2_char*q**2)

    else:
        one_over_q2_char = self.omega_inverse_bar/(2*self.Avec*self.mp)


        return np.where(np.less(one_over_q2_char*q**2, 0.03), 1, exp(-one_over_q2_char*q**2))


def debye_waller(self, q):
    '''Debye Waller factor
    Inputs
    ------
    q: float or array in units of eV'''# comment multi-atom output.
    if (isinstance(q,(np.ndarray,list)) ):
        return np.array([self._debye_waller_scalar(qi) for qi in q])
    elif(isinstance(q,float)):
        return self._debye_waller_scalar(q)
    else:
        print("Warning! debye_waller function given invalid quantity ")
        return 0.0

def _R_multiphonons_prefactor(self, sigman):
    # Input sigman in cm^2 output rate pre-factor in cm^2
    if self.n_atoms == 2:
        return sigman*((1/(self.Avec[0]*self.mp + self.Avec[1]*self.mp))*
                (self.rhoX*self.eVcm**3)/(2*self.mX*(self.muxnucleon)**2))*(((1/self.eVcm)**2)*
                (self.eVtoInvYr/self.eVtokg))
    else:
        return sigman*((1/(self.A*self.mp + self.A*self.mp))*
                (self.rhoX*self.eVcm**3)/(2*self.mX*(self.muxnucleon)**2))*(((1/self.eVcm)**2)*
                (self.eVtoInvYr/self.eVtokg))



def sigma_multiphonons(self, threshold, dark_photon=False):
    '''
    returns DM-proton cross-section [cm^2] corresponding to 3 events/kg/yr
    Inputs
    ------
    threshold: float
      experimental threshold, in eV
    dark_photon: Bool
      If set to True, a dark photon mediator is assumed, by setting f_d(q) = Z_d(q), with Z_d(q) the momentum dependent effective charges. If set to False, darkELF sets f_d=A_d, which corresponds to a scalar mediator with coupling to nuclei.
    '''
    if dark_photon:
      assert self.fd_loaded, "Error: effective charge not loaded. Cannot perform calculation for dark photon mediator."

    rate = self.R_multiphonons_no_single(threshold, dark_photon=dark_photon) + self.R_single_phonon(threshold, dark_photon=dark_photon)
    if rate != 0:
        return (3.0*1e-38)/rate
    else:
        return float('inf')


def R_multiphonons_no_single(self, threshold, sigman=1e-38, dark_photon=False):
    """
    Returns rate for DM scattering with a harmonic lattice, including multiphonon contributions but excluding the coherent single phonon contribution

    Inputs
    ------
    threshold: float in [eV]
    sigma_n: float
        DM-nucleon cross section in [cm^2], defined with respect to the reference momentum of q0. (q0 is specified by the 'update_params' function)
        DM-nucleus cross section assumed to be coherently enhanced by A^2 by default (if dark photon flag not set)
    dark_photon: boole
        If set to True, a dark photon mediator is assumed, by setting f_d(q) = Z_d(q), with Z_d(q) the momentum dependent effective charges. If set to False, darkELF sets f_d=A_d, which corresponds to a scalar mediator with coupling to nuclei.

    Outputs
    -------
    rate as function of threshold, in [1/kg/yr]
    """
    if dark_photon:
      assert self.fd_loaded, "Error: effective charge not loaded. Cannot perform calculation for dark photon mediator."

    if threshold > self.omegaDMmax:
        return 0
    else:
        npoints = 1000
        # For better precision, we use linear sampling for omega < max phonon energy and log sampling for omega > max phonon energy.
        if(threshold<self.dos_omega_range[-1]):
            omegarange_linear=np.linspace(threshold,np.min([self.dos_omega_range[-1],self.omegaDMmax]), npoints)
            dR_linear=[self._dR_domega_multiphonons_no_single(omega, sigman=sigman, dark_photon=dark_photon) for omega in omegarange_linear]
            R_linear=np.trapz(dR_linear, omegarange_linear)
        else:
            R_linear=0.0
        if(self.omegaDMmax>self.dos_omega_range[-1]):
            omegarange_log=np.logspace(np.max([np.log10(self.dos_omega_range[-1]),np.log10(threshold)]),\
                                     np.log10(self.omegaDMmax), npoints)
            dR_log=[self._dR_domega_multiphonons_no_single(omega, sigman=sigman, dark_photon=dark_photon) for omega in omegarange_log]
            R_log=np.trapz(dR_log, omegarange_log)
        else:
            R_log=0

        return R_linear+R_log



# Multiphonon_expansion term

def _dR_domega_multiphonons_no_single(self, omega, sigman=1e-38, dark_photon=False):

    '''dR_domega single-phonon coherent removed'''
    if(dark_photon): # check if effective charges are loaded
        assert self.fd_loaded, "Error: effective charge not loaded. Cannot perform calculation for dark photon mediator."

    if omega > self.omegaDMmax:
        return 0

    if (omega > self.dos_omega_range[-1]):
        qmin=self.qmin(omega)
    else: ## For q<qBZ and omega<max phonon energy, we use the single phonon rate.
        qmin = max(self.qmin(omega), self.qBZ)

    # for q>q_IA_cut, the impulse approximation is used
    q_IA_cut = max([2*sqrt(2*self.Avec[i]*self.mp*self.omega_bar[i]) for i in np.arange(self.n_atoms)])
    qmax = min(self.qmax(omega), q_IA_cut)

    if qmin >= qmax:
        return 0

    npoints = 100
    qrange = np.linspace(qmin, qmax, npoints)

    if self.n_atoms == 2:
        if dark_photon:
            fd = np.array([self.fd_darkphoton[i](qrange) for i in range(2)])\
            *sqrt(self.debye_waller(qrange)).T
        else:
            fd = np.tile(np.array([self.Avec]),(npoints, 1)).T*sqrt(self.debye_waller(qrange)).T

    else:
        if dark_photon:
            fd = np.array([self.fd_darkphoton(qrange),self.fd_darkphoton(qrange)])*sqrt(self.debye_waller(qrange)).T
                # need fix this part.. ## SK??
        else:
            fd =  np.tile(np.array([self.A, self.A]),(npoints, 1)).T*sqrt(self.debye_waller(qrange)).T

    formfactorsquared = self.Fmed_nucleus(qrange)**2

    if self.n_atoms == 2:
        otherpart = np.array([np.zeros(npoints), np.zeros(npoints)])
        for i in range(2):
            for n in range(1, len(self.phonon_Fn[i])):
                # Debye-Waller included in f_d
                qpart = qrange**(2*n + 1)
                otherpart[i] += (1/(2*self.Avec[i]*self.mp))**n*qpart*self.Fn_interpolations[i][n](omega)

    else:
        otherpart = np.zeros(npoints)
        for n in range(1, len(self.phonon_Fn)):
            qpart = qrange**(2*n + 1)
            otherpart += (1.0/(2*self.A*self.mp))**n*qpart*self.Fn_interpolations[n](omega)

    # add contributions from different elements
    dR_domega_dq = np.sum(fd**2*otherpart, axis = 0)*formfactorsquared*self.etav((qrange/(2*self.mX)) + omega/qrange)
    multiph_expansion_part = np.trapz(dR_domega_dq, qrange)


    #### Impulse approx part

    if self.n_atoms == 2:
        min_q = max([2*sqrt(2*self.Avec[i]*self.mp*self.omega_bar[i]) for i in [0, 1]])
        qmin = max(self.mX*(self.vmax - sqrt(self.vmax**2 - (2*omega/self.mX))), min_q)
    else:
        qmin = max(self.mX*(self.vmax - sqrt(self.vmax**2 - (2*omega/self.mX))), 2*sqrt(2*self.A*self.mp*self.omega_bar))

    qmax = self.mX*(self.vmax + sqrt(self.vmax**2 - (2*omega/self.mX)))

    if qmin >= qmax:
        impulse_approx_part = 0

    else:
        npoints = 100
        qrange = np.linspace(qmin, qmax, npoints)

        if self.n_atoms == 2:
            impulse_approx_part = 0

            if dark_photon:
                fd = np.array([self.fd_darkphoton[i](qrange) for i in range(2)])
            else:
                fd = np.tile(np.array(self.Avec), (npoints, 1)).T

            formfactorsquared = self.Fmed_nucleus(qrange)**2

            deltaq = sqrt(np.tile(self.omega_bar/(2*self.Avec*self.mp), (npoints, 1)).T*np.tile(qrange, (2, 1))**2)

            structurefactor = (1/(deltaq*sqrt(2*pi)))*exp(-(omega - np.tile(1/(2*self.Avec*self.mp),
                                            (npoints, 1)).T*(np.tile(qrange, (2, 1))**2))**2/(2*deltaq**2))*qrange
            dR_domega_dq = np.sum(structurefactor*(fd**2)*formfactorsquared*self.etav((qrange/(2*self.mX)) + omega/qrange), axis=0)

            impulse_approx_part = np.trapz(dR_domega_dq, qrange)

        else:

            if dark_photon:
                fd = self.fd_darkphoton(qrange)
            else:
                fd = self.A

            formfactorsquared = self.Fmed_nucleus(qrange)**2

            deltaq = sqrt(qrange**2*self.omega_bar/(2*self.A*self.mp))

            structurefactor = qrange*(1/(deltaq*sqrt(2*pi)))*exp(-(omega - qrange**2/(2*self.A*self.mp))**2/(2*deltaq**2))
            dR_domega_dq = (fd**2 + fd**2)*formfactorsquared*structurefactor*self.etav((qrange/(2*self.mX)) + omega/qrange)

            impulse_approx_part = np.trapz(dR_domega_dq, qrange)

    return self._R_multiphonons_prefactor(sigman)*(multiph_expansion_part + impulse_approx_part)


############################################################################################
#
# Single phonon coherent term

# !TL: functions wanted:
#      _dR_domega_coherent_single -- internal function used for plotting
#      R_coherent_single_phonon -- total rate, obtained analytically

def _dR_domega_coherent_single(self, omega, sigman=1e-38, dark_photon=False):

#   internal function used for plotting, don't integrate over as there are artificial sharp peaks
    if(dark_photon): # check if effective charges are loaded
        assert self.fd_loaded, "Error: effective charge not loaded. Cannot perform calculation for dark photon mediator."

    if self.vmax**2 < 2*omega/self.mX:
        return 0

    qmin = self.mX*(self.vmax - sqrt(self.vmax**2 - (2*omega/self.mX)))
    qmax = min(self.mX*(self.vmax + sqrt(self.vmax**2 - (2*omega/self.mX))), self.qBZ)

    if qmin > qmax:
        return 0

    # following stuff is doing the acoustic part analytically
    if self.n_atoms == 2:
        if dark_photon:
            fd = np.array([self.fd_darkphoton[i](omega/self.cLA) for i in range(2)])*sqrt(self.debye_waller(omega/self.cLA))
        else:
            fd = self.Avec
    else:
        if dark_photon:
            fd = np.array([self.fd_darkphoton(omega/self.cLA), self.fd_darkphoton(omega/self.cLA)])*sqrt(self.debye_waller(omega/self.cLA))
        else:
            fd = np.array([self.A, self.A])*sqrt(self.debye_waller(omega/self.cLA))

    formfactorsquared = self.Fmed_nucleus(omega/self.cLA)**2

    # this bit is only scalar mediators, need protection from dark_photon flag
    if (omega < 2*self.mX*self.cLA*(self.vmax - self.cLA)) and (omega < self.cLA*self.qBZ) and (omega < self.LOvec[0]):
        if self.n_atoms == 2:
            acoustic_part = ((np.sum(fd)**2/np.sum(self.Avec))*(1/(2*self.mp))*((omega/self.cLA)**2/self.cLA**2)*
                        formfactorsquared*
                        self.etav((omega/self.cLA)/(2*self.mX) + omega/(omega/self.cLA)))
        else: # SK add fd here? revisit this.
            acoustic_part = (((self.A+self.A)/(2*self.mp))*((omega/self.cLA)**2/self.cLA**2)*
                        formfactorsquared*self.debye_waller(omega/self.cLA)*self.etav((omega/self.cLA)/(2*self.mX) + omega/(omega/self.cLA)))
    else:
        acoustic_part = 0

    #### Optical part:

    def dR_domega_dq_optical(q):

        if q > self.qBZ:
            return 0
        else:
            pass

        if self.n_atoms == 2:
            if dark_photon:
                fd = np.array([self.fd_darkphoton[i](q) for i in range(2)])*sqrt(self.debye_waller(q))
            else:
                fd = self.Avec*sqrt(self.debye_waller(q))

        else:
            if dark_photon:
                fd = np.array([self.fd_darkphoton(q), self.fd_darkphoton(q)])*sqrt(self.debye_waller(q))
                    # !EV: rearrange factors of 2, currently have debye_waller function defined as e^(-2W(q))
            else:
                fd = np.array([self.A, self.A])*self.debye_waller(q)
                # !EV I've written this in a way I don't like, need to rearrange stuff

        formfactorsquared = self.Fmed_nucleus(q)**2

        width = 0.5e-3 # giving the delta functions finite width of 0.5 meV

        if self.n_atoms == 2:
            optical_factor1 = ((self.lattice_spacing/self.eVtoA0)**2/(2*self.LOvec[0]*self.mp))
            optical_factor2 = (fd[0]**2*self.Avec[1]/self.Avec[0] + fd[1]**2*self.Avec[0]/self.Avec[1] - 2*fd[0]*fd[1] +
                                fd[0]*fd[1]*q**2/16)/(np.sum(self.Avec))
            optical_part = q**3*optical_factor1*optical_factor2*(1/(width*sqrt(2*pi)))*exp(-(1/2)*(omega - self.LOvec[0])**2/(width)**2)
            # fix debye_waller part
            return formfactorsquared*self.etav(q/(2*self.mX) + omega/q)*(optical_part)

        else:
            optical_factor1 = ((self.lattice_spacing/self.eVtoA0)**2/(32*self.LOvec[0]*self.mp))
            optical_factor2 = (self.A*self.A)/(self.A + self.A)
            optical_part = q**5*optical_factor1*optical_factor2*(1/(width*sqrt(2*pi)))*exp(-(1/2)*(omega - self.LOvec[0])**2/(width)**2)
            return formfactorsquared*self.etav(q/(2*self.mX) + omega/q)*(optical_part)

    return self._R_multiphonons_prefactor(sigman)*(integrate.quad(lambda q: dR_domega_dq_optical(q),
                    qmin, qmax)[0] + acoustic_part)


def R_single_phonon(self, threshold, sigman=1e-38, dark_photon=False):

    ###############################
    # Optical part
    if(dark_photon): # check if effective charges are loaded
        assert self.fd_loaded, "Error: effective charge not loaded. Cannot perform calculation for dark photon mediator."

    if (self.LOvec[0] < threshold) or (self.mX*self.vmax**2/2 < self.LOvec[0]):
        optical_rate = 0
    else:

        qmin = self.mX*(self.vmax - sqrt(self.vmax**2 - 2*self.LOvec[0]/self.mX))
        qmax = min(self.qBZ, self.mX*(self.vmax + sqrt(self.vmax**2 - 2*self.LOvec[0]/self.mX)))

        if qmin > qmax:
            optical_rate = 0

        else:

            npoints = 100
            qrange = np.linspace(qmin, qmax, npoints)

            dR_dq_optical = np.zeros(npoints)

            formfactorsquared = self.Fmed_nucleus(qrange)**2

            if self.n_atoms == 2:

                if dark_photon:
                    fd = np.array([self.fd_darkphoton[i](qrange) for i in range(2)])*sqrt(self.debye_waller(qrange)).T
                else:
                    fd = np.tile(np.array([self.Avec]),(npoints, 1)).T*sqrt(self.debye_waller(qrange)).T

                optical_factor1 = ((self.lattice_spacing/self.eVtoA0)**2/(2*self.LOvec[0]*self.mp))
                optical_factor2 = (fd[0]**2*self.Avec[1]/self.Avec[0] + fd[1]**2*self.Avec[0]/self.Avec[1] - 2*fd[0]*fd[1] +
                                    fd[0]*fd[1]*qrange**2/16)/(np.sum(self.Avec))

                optical_part = qrange**3*optical_factor1*optical_factor2

                velocity_part = self.etav(qrange/(2*self.mX) + self.LOvec[0]/qrange)

                dR_dq_optical = optical_part*velocity_part*formfactorsquared

                optical_rate = np.trapz(dR_dq_optical, qrange)

            else:

                optical_factor1 = ((self.lattice_spacing/self.eVtoA0)**2/(32*self.LOvec[0]*self.mp))
                optical_factor2 = 1/(self.A + self.A)

                if dark_photon:
                    fd = self.fd_darkphoton(qrange)
                else:
                    fd = self.A

                optical_part = qrange**5*optical_factor1*optical_factor2

                velocity_part = self.etav(qrange/(2*self.mX) + self.LOvec[0]/qrange)

                dR_dq_optical = (fd**2)*optical_part*velocity_part*self.debye_waller(qrange)*formfactorsquared

                optical_rate = np.trapz(dR_dq_optical, qrange)

    ###############################
    # Acoustic part

    omegamin = threshold
    omegamax = min(2*self.mX*self.cLA*(self.vmax - self.cLA), self.cLA*self.qBZ, self.mX*self.vmax**2/2, self.LOvec[0])

    if omegamax < omegamin:
        acoustic_rate = 0
    else:
        npoints = 100
        omegarange = np.linspace(omegamin, omegamax, npoints)
        dR_domega_acoustic = np.zeros(npoints)

        formfactorsquared = self.Fmed_nucleus(omegarange/self.cLA)**2

        if self.n_atoms == 2:

            if dark_photon:
                fd = np.array([self.fd_darkphoton[i](omegarange/self.cLA) for i in range(2)])*sqrt(self.debye_waller(omegarange/self.cLA)).T
            else:
                fd = np.tile(np.array([self.Avec]),(npoints, 1)).T*sqrt(self.debye_waller(omegarange/self.cLA)).T

            dR_domega_acoustic = (fd[0] + fd[1])**2*((1/(self.mp*np.sum(self.Avec)))*((omegarange/self.cLA)**2/self.cLA**2)*
                formfactorsquared*self.etav((omegarange/self.cLA)/(2*self.mX)
                                            + omegarange/(omegarange/self.cLA)))

            acoustic_rate = np.trapz(dR_domega_acoustic, omegarange)

        else:
            if dark_photon:
                fd = self.fd_darkphoton(omegarange/self.cLA)
            else:
                fd = self.A

            dR_domega_acoustic = (fd + fd)**2*((1/(2*self.mp*self.A))*((omegarange/self.cLA)**2/self.cLA**2)*
                formfactorsquared*self.debye_waller(omegarange/self.cLA)*self.etav((omegarange/self.cLA)/(2*self.mX)
                                                                        + omegarange/(omegarange/self.cLA)))

            acoustic_rate = np.trapz(dR_domega_acoustic, omegarange)


    return self._R_multiphonons_prefactor(sigman)*(optical_rate + acoustic_rate)

###############################################################################################
# Loading in dark photon fd
#

def load_fd_darkphoton(self,datadir,filename):

    # Loads momentum dependent effective charges

    if self.n_atoms == 2:
        # If there are two distinct atoms, fd_darkphoton needs to be
        # set to two element list of filenames
        if len(filename) != 2:
            print("Warning! Dark photon fd not loaded. fd_darkphoton must be list of two filenames")
            self.fd_loaded=False
        else:
            fd_1_path = datadir + self.target+'/'+ filename[0]
            fd_2_path = datadir + self.target+'/'+ filename[1]

            if (not os.path.exists(fd_1_path)) or (not os.path.exists(fd_2_path)):
                print("Warning! Dark photon fd not loaded. Need to set fd_filename for both atoms")
                self.fd_loaded=False

            else:
                self.fd_loaded=True
                fd_1 = np.loadtxt(fd_1_path).T
                fd_2 = np.loadtxt(fd_2_path).T
                print("Loaded " + filename[0] + " and " + filename[1] + " for effective charges")
                # self.fd_data = np.array([fd_1, fd_2])
                # self.fd_darkphoton = np.array([interp1d(i[0],i[1],kind='linear', fill_value = (i[1][0], i[1][-1]),bounds_error=False)
                #                                for i in self.fd_data])
                self.fd_darkphoton = np.array([interp1d(fd_1[0],fd_1[1],kind='linear', fill_value = (fd_1[1][0], fd_1[1][-1]),bounds_error=False),
                                                interp1d(fd_2[0],fd_2[1],kind='linear', fill_value = (fd_2[1][0], fd_2[1][-1]),bounds_error=False)])
                            # Fills ends by making constant at end of ranges

    else:

        fd_path = datadir + self.target+'/'+ filename

        if( not os.path.exists(fd_path)):
            print("Warning! Dark photon fd not loaded. Need to set fd_filename if needed. Otherwise defaults to massive mediator ")
            self.fd_loaded=False
        else:
            self.fd_loaded=True
            self.fd_data = np.loadtxt(fd_path).T
            print("Loaded " + filename + " for dark photon couplings")
            self.fd_darkphoton = interp1d(self.fd_data[0],self.fd_data[1],kind='linear',
                                fill_value=(self.fd_data[1][0], self.fd_data[1][-1]),bounds_error=False)

    return
