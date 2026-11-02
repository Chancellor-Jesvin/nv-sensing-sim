"""Shot-noise-limited magnetometer sensitivity from a coherence value."""
import numpy as np
from .nv import gam

def readout_noise_factor(alpha0, contrast):
    alpha1 = alpha0 * (1 - contrast)
    return np.sqrt(1 + 2*(alpha0 + alpha1) / (alpha0 - alpha1)**2)

def eta_nT(T, W, N, sigma_R, t_ov=3.0):
    """Sensitivity in nT/sqrt(Hz). T and t_ov in us, W is coherence.
    N=0: DC Ramsey, phase 2*pi*gam*B*T.  N>=1: AC sensing, phase 4*gam*B*T."""
    k = 2*np.pi if N == 0 else 4.0
    eta_G = sigma_R * np.sqrt((T + t_ov) * 1e-6) / (W * k * gam * T)
    return eta_G * 1e5