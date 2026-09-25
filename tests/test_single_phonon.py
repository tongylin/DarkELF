"""Regression test: the integrated single acoustic phonon rate must match the integral of the
differential rate _dR_domega_coherent_single, i.e. the long-wavelength structure factor
S_LA = (2 pi/Omega_c) (sum_d A_d) q^2/(2 m_p omega) delta(omega - c_LA q) of 2205.02250.

Run with `python tests/test_single_phonon.py` or pytest.
"""
import os, sys
import numpy as np
from scipy import integrate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from darkelf import darkelf


def _acoustic_ratio(target, mX, mMed, threshold=1e-3):
    t = darkelf(target=target)
    t.update_params(mX=mX, mMed=mMed)
    # upper limit as in _R_single_acoustic; masses are chosen so this lies well below the LO phonon
    omegamax = min(2*t.mX*t.cLA*(t.vmax - t.cLA), t.cLA*t.qBZ, t.mX*t.vmax**2/2, t.omega1ph_max)
    omegas = np.linspace(threshold, omegamax, 2000)
    dR = np.array([t._dR_domega_coherent_single(w) for w in omegas])
    return t._R_single_acoustic(threshold) / integrate.trapezoid(dR, omegas)


def test_acoustic_single_phonon_normalization():
    for mX in (1e5, 2e5):           # eV; acoustic omega_max ~ 14, 28 meV < omega_LO = 60 meV
        for mMed in (1e9, 0):        # massive, massless mediator
            r = _acoustic_ratio('Si', mX, mMed)
            # residual differences: Debye-Waller factor (only in _R_single_acoustic) and quadrature
            assert abs(r - 1) < 0.03, f"Si mX={mX} mMed={mMed}: ratio {r:.3f}"


if __name__ == '__main__':
    for mX in (1e5, 2e5):
        for mMed in (1e9, 0):
            print(f"Si mX={mX:.0e} eV mMed={mMed:.0e}: R_single_acoustic / int dR_coherent_single = "
                  f"{_acoustic_ratio('Si', mX, mMed):.4f}")
    test_acoustic_single_phonon_normalization()
    print('passed')
