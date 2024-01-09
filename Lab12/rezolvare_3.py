import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def metropolis(func, n, p, draws=10000):
    trace = np.zeros(draws)
    old_x = np.random.randint(0, n+1)
    old_prob = func.pmf(old_x)
    
    for i in range(draws):
        new_x = old_x + np.random.choice([-1, 1])
        new_prob = func.pmf(new_x)
        acceptance = new_prob / old_prob
        
        if acceptance >= np.random.random():
            trace[i] = new_x
            old_x = new_x
            old_prob = new_prob
        else:
            trace[i] = old_x
    
    return trace

n_params = [1, 2, 4]
p_params = [0.25, 0.5, 0.75]
x = np.arange(0, max(n_params) + 1)
f, ax = plt.subplots(len(n_params), len(p_params), sharex=True, sharey=True, figsize=(8, 7), constrained_layout=True)

for i in range(len(n_params)):
    for j in range(len(p_params)):
        n = n_params[i]
        p = p_params[j]

        binomial_distribution = stats.binom(n=n, p=p)

        trace = metropolis(binomial_distribution, n=n, p=p)

        ax[i, j].hist(trace, bins=np.arange(0, n + 2) - 0.5, density=True, alpha=0.5, color='C0')
        ax[i, j].set_title(f"N = {n}, θ = {p}")

plt.show()
