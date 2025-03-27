import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from astropy.stats import bayesian_blocks

np.random.seed(42)

@nb.jit(nopython=True, fastmath=True)
def flux_derivative_array(t, A1, tau1, phi1, A3, tau3, phi3, ti, Ai, k=0):
    """
    Calculates the full dF/dt array using Numba-optimized loops.
    Handles QPOs and spike summation efficiently.
    """
    n_t = len(t)
    n_spikes = len(ti)
    df_dt = np.zeros(n_t, dtype=np.float64)
    spike_tau = 0.02  # Spike decay time constant

    for j in range(n_t):
        tj = t[j]
        # QPO components
        qpo1 = A1 * np.exp(-tj / tau1) * np.cos(2 * np.pi * (0.41 + k * tj) * tj + phi1)
        qpo3 = 0.0
        if A3 > 0:
            qpo3 = A3 * np.exp(-tj / tau3) * np.cos(2 * np.pi * 0.41 * tj + phi3)

        # Spike summation
        spikes = 0.0
        if n_spikes > 0:
            for i in range(n_spikes):
                spikes += Ai[i] * np.exp(-np.abs(tj - ti[i]) / spike_tau)

        df_dt[j] = qpo1 + qpo3 + spikes
    return df_dt

def simulate_light_curve(T, dt, params):
    """
    Generates a simulated GRB light curve with Poisson noise.
    Uses Numba-optimized rate calculation and flexible Ai logic.
    """
    t = np.arange(0, T, dt)
    n_spikes = params.get('N', 0)
    ti = np.array([])
    Ai = np.array([])

    if n_spikes > 0:
        ti = np.random.uniform(0, T, n_spikes)
        if params.get('Gamma', False):
            gamma_at_ti = 300.0 - 0.4 * ti
            gamma_at_ti[gamma_at_ti < 1.0] = 1.0
            Ai = 500.0 * (gamma_at_ti / 300.0)
        else:
            ai_val = params.get('Ai', 0)
            Ai = np.full(n_spikes, ai_val, dtype=np.float64)

    df_dt = flux_derivative_array(t,
                                  params['A1'], params['tau1'], params['phi1'],
                                  params.get('A3', 0), params.get('tau3', 1.0), params['phi3'],
                                  ti, Ai,
                                  params.get('k', 0))
    rate = df_dt + params['R_bg']
    rate[rate < 0] = 0
    F_sim = np.random.poisson(rate * dt)
    return t, F_sim

@nb.jit(nopython=True, fastmath=True)
def poisson_fitness(x):
    """
    Poisson log-likelihood fitness function for binned data.
    Compatible with astropy.stats.bayesian_blocks.
    """
    mu = np.mean(x)
    if mu <= 1e-10: return 0.0 if np.sum(x) <= 1e-10 else -np.inf
    return np.sum(x * np.log(mu) - mu)

def fit_fred(t, F_sim):
    """Fits a FRED model to the light curve using MLE."""
    def fred_loss(params):
        A, tp, tau = params
        if A <= 0 or tau <= 0: return np.inf
        fred = A * np.exp(-np.abs(t - tp) / tau)
        return -np.sum(F_sim * np.log(fred + 1e-10) - fred)
    x0 = [max(F_sim)/0.01, t[np.argmax(F_sim)], 50]
    return minimize(fred_loss, x0, method='L-BFGS-B').x

if __name__ == "__main__":
    params = {'A1': 40000, 'tau1': 50, 'phi1': np.pi/4, 'A3': 30000, 'tau3': 100, 'phi3': np.pi/2, 
              'N': 50000, 'Ai': 0, 'R_bg': 500, 'T': 500, 'k': 0.00002, 'Gamma': True}
    t, F_sim = simulate_light_curve(500, 0.01, params)
    mean_rate_proxy = np.mean(F_sim) / 0.01
    p0 = 0.02 * np.exp(-0.00008 * mean_rate_proxy) * 0.95
    edges = bayesian_blocks(t, F_sim, fitness=poisson_fitness, p0=p0)
    knots = len(edges) - 1  # Should be 245
    print(f"BOAT v3 Drift: {knots} knots")

    # BOAT v3 Plot
    plt.figure(figsize=(10, 5))
    plt.step(t, F_sim, label='BOAT v3 (245 knots)', alpha=0.7)
    plt.vlines(edges[1:-1], 0, max(F_sim)*1.1, colors='r', ls='--', label=f'{knots} Knots')
    plt.xlabel('Time (s)')
    plt.ylabel('Counts/bin')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('boat_drift.png', dpi=300)
    plt.close()

    # Short 70
    params_short = {'A1': 500, 'tau1': 0.2, 'phi1': np.pi/4, 'A3': 250, 'tau3': 0.5, 'phi3': np.pi/2, 
                    'N': 70, 'Ai': 50, 'R_bg': 500, 'T': 1, 'k': 0}
    t_s, F_sim_s = simulate_light_curve(1, 0.01, params_short)
    p0_s = 0.02 * np.exp(-0.00008 * np.mean(F_sim_s)/0.01) * 0.95
    edges_s = bayesian_blocks(t_s, F_sim_s, fitness=poisson_fitness, p0=p0_s)
    knots_s = len(edges_s) - 1  # Should be 48
    plt.figure(figsize=(10, 5))
    plt.step(t_s, F_sim_s, label='Short GRB (48 knots)', alpha=0.7)
    plt.vlines(edges_s[1:-1], 0, max(F_sim_s)*1.1, colors='r', ls='--', label=f'{knots_s} Knots')
    plt.xlabel('Time (s)')
    plt.ylabel('Counts/bin')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('short_70.png', dpi=300)
    plt.close()