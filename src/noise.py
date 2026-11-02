"""Ornstein-Uhlenbeck detuning noise and dynamical-decoupling coherence. MHz and microseconds."""
import numpy as np

def make_noise(sigma, tau_c, dt, Tmax, n_real, seed=1):
    """Exact OU process, shape (n_real, n_steps). sigma in MHz, tau_c and dt in us."""
    n_steps = int(round(Tmax / dt))
    rng = np.random.default_rng(seed)
    a = np.exp(-dt / tau_c)
    x = np.empty((n_real, n_steps))
    x[:, 0] = rng.normal(0, sigma, n_real)
    for k in range(1, n_steps):
        x[:, k] = a * x[:, k-1] + sigma * np.sqrt(1 - a**2) * rng.normal(size=n_real)
    return x

def coherence_g(nz, dt, T, N):
    """Coherence after total time T with N ideal pi pulses (N=0 is free decay)."""
    n = int(round(T / dt))
    t = (np.arange(n) + 0.5) * dt
    s = np.ones(n) if N == 0 else (-1.0) ** np.floor(N * t / T + 0.5)
    phi = 2 * np.pi * dt * (nz[:, :n] @ s)
    return np.abs(np.mean(np.exp(1j * phi)))

def T2_from_curve(Ts, C):
    """Time where coherence crosses 1/e (linear interpolation). NaN if it never does."""
    idx = np.where(C < np.exp(-1))[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    if i == 0:
        return Ts[0]
    return np.interp(np.exp(-1), [C[i], C[i-1]], [Ts[i], Ts[i-1]])

def aligned_times(N, dt, T_avail, n_points=120):
    """Sequence times for which every pulse lands exactly on a time step."""
    step = 2 * N * dt
    m_max = int(np.floor(T_avail / step + 1e-9))
    m = np.unique(np.round(np.geomspace(1, m_max, n_points)).astype(int))
    return m * step