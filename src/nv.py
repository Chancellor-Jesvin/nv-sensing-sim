"""NV-center spin Hamiltonians and ODMR spectra. Frequencies in MHz, fields in gauss."""
import numpy as np
from qutip import jmat, tensor, qeye

D = 2870.0        # zero-field splitting, MHz
gam = 2.8025      # electron gyromagnetic ratio, MHz/G
A_par = -2.14     # 14N axial hyperfine, MHz
P = -4.95         # 14N quadrupole, MHz

Sx, Sy, Sz = jmat(1, 'x'), jmat(1, 'y'), jmat(1, 'z')

def H_nv(Bx=0.0, By=0.0, Bz=0.0):
    """Electron spin-1 Hamiltonian. z is the NV axis."""
    return D * Sz**2 + gam * (Bx*Sx + By*Sy + Bz*Sz)

def H_nv_hyperfine(Bz):
    """Electron plus 14N nuclear spin, field along the NV axis (diagonal Hamiltonian)."""
    Sz_e = tensor(Sz, qeye(3))
    Iz_n = tensor(qeye(3), Sz)
    return D*Sz_e**2 + gam*Bz*Sz_e + P*Iz_n**2 + A_par*Sz_e*Iz_n

def odmr_lines(B):
    """All ms=0 to ms=+-1 transition frequencies (MHz), on-axis field B (G)."""
    Ed = np.real(H_nv_hyperfine(B).full().diagonal())
    lines = []
    for j in range(3):
        E0 = Ed[3*1 + j]                 # ms = 0
        lines.append(Ed[3*0 + j] - E0)   # ms = +1
        lines.append(Ed[3*2 + j] - E0)   # ms = -1
    return np.array(lines)

def lorentz_dip(f, f0, fwhm, contrast):
    hw = fwhm / 2
    return contrast * hw**2 / ((f - f0)**2 + hw**2)

def odmr_spectrum(f, B, fwhm=1.0, contrast=0.03):
    signal = np.ones_like(f)
    for f0 in odmr_lines(B):
        signal -= lorentz_dip(f, f0, fwhm, contrast)
    return signal