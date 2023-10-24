import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

rate = 20

#urmatoarea linie genereaza clienti
x = np.arange(0,160)

pmf = stats.poisson.pmf(x, rate)

order_time = stats.norm.rvs(2,0.5, size=160)

alpha=0
cook_time=stats.expon.rvs(scale=alpha, size=160)

z=cook_time+order_time#timpul de servire se va afla prin insumarea acestor doua distributii

az.plot_posterior({'Timpul aproximativ de servire':z,'Poisson':pmf})

plt.show()
