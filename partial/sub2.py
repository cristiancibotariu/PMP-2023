import matplotlib.pyplot as plt

import pymc as pm
import arviz as az
from scipy import stats


if __name__ == '__main__':
    #miu = 5 valori random pt punctul 1
    #sigma = 2

    model = pm.Model()
    with model:
        miu = pm.Poisson("miu", mu=10) #urm 2 linii folosim distributii pentru a genera miu si sigma
        sigma = pm.Poisson("sigma", mu=10)
        avg_wait = pm.Normal("average wait time", miu, sigma)

        trace = pm.sample(200) #generam 200 de timpi

    az.plot_posterior(trace)
    plt.show()
