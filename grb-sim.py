import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from astropy.stats import bayesian_blocks

np.random.seed(42)

@nb.jit(nopython=True, fastmath=True)
def flux_derivative_array(t, A1, tau1, phi1, A3, tau3, phi3, ti, Ai, k=0):
    n_t = len(t)
    df_dt = np.zeros(n_t)
    for j in range(n_t):
        tj = t[j]
        qpo1 = A1 * np.exp(-tj / tau1) * np.cos(2 * np.pi * (0.41 + k * tj) * tj + phi1)
        qpo3 = A3 * np.exp(-tj / tau3) * np.cos(2 * np.pi * 0.41 * tj + phi3) if A3 else 0
        spikes = 0.0
        for i in range(len(ti)):
            spikes += Ai[i] * np.exp(-np.abs(tj - ti[i]) / 0.02)
        df_dt[j] = qpo1 + qpo3 + spikes
    return df_dt

def simulate_light_curve(T, dt, params):
    t = np.arange(0, T, dt)
    ti = np.random.uniform(0, T, params['N'])
    Ai = np.full(params['N'], params['Ai']) if params['N'] > 0 and 'Gamma' not in params else \
         500 * (300 - 0.4 * ti) / 300
    df_dt = flux_derivative_array(t, **params, ti=ti, Ai=Ai)
    return t, np.random.poisson((df_dt + params['R_bg']) * dt))

@nb.jit(nopython=True, fastmath=True)
def poisson_fitness(x):
    mu = np.mean(x)
    if mu <= 1e-10: return 0.0 if np.sum(x) <= 1e-10 else -np.inf
    return np.sum(x * np.log(mu) - mu)

def fit_fred(t, F_sim):
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