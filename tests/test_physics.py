import numpy as np
from src.nv import H_nv, D, gam
from src.noise import make_noise, coherence_g

def test_resonances_on_axis():
    B = 50.0
    E = H_nv(Bz=B).eigenenergies()
    assert np.isclose(E[1] - E[0], D - gam*B)
    assert np.isclose(E[2] - E[0], D + gam*B)

def test_free_decay_matches_ou_formula():
    sigma, tau_c, dt = 0.2, 2.0, 0.05
    nz = make_noise(sigma, tau_c, dt, 20.0, 4000)
    for T in [1.0, 2.0, 4.0]:
        exact = np.exp(-(2*np.pi*sigma*tau_c)**2 * (T/tau_c - 1 + np.exp(-T/tau_c)))
        assert abs(coherence_g(nz, dt, T, 0) - exact) < 0.05