"""Two-level dynamics with flip-angle errors, for CPMG and XY8 comparisons."""
import numpy as np

sx_m = np.array([[0, 1], [1, 0]], dtype=complex)
sy_m = np.array([[0, -1j], [1j, 0]], dtype=complex)
I2 = np.eye(2, dtype=complex)

def rot(axis, theta):
    s = sx_m if axis == 'x' else sy_m
    return np.cos(theta/2) * I2 - 1j * np.sin(theta/2) * s

def cumulative_noise(noise):
    return np.concatenate([np.zeros((noise.shape[0], 1)), np.cumsum(noise, axis=1)], axis=1)

def run_sequence(cs, dt, pulses, T, eps, prep):
    """Mean of (1 - 2*P0) over noise realizations. cs = cumulative_noise(noise).
    Ideal pulses and no noise give 1. prep is 'x' or 'y' (axis of the pi/2 pulses)."""
    N = len(pulses)
    n_tot = int(round(T / dt))
    edges = [0] + [int(round(n_tot * (k - 0.5) / N)) for k in range(1, N + 1)] + [n_tot]
    psi = np.tile(np.array([1, 0], dtype=complex), (cs.shape[0], 1))
    psi = psi @ rot(prep, np.pi/2).T
    for j in range(N + 1):
        phi = 2 * np.pi * dt * (cs[:, edges[j+1]] - cs[:, edges[j]])
        psi = np.stack([psi[:, 0] * np.exp(-1j*phi/2), psi[:, 1] * np.exp(1j*phi/2)], axis=1)
        if j < N:
            psi = psi @ rot(pulses[j], np.pi * (1 + eps)).T
    psi = psi @ rot(prep, np.pi/2).T
    return np.mean(1 - 2 * np.abs(psi[:, 0])**2)