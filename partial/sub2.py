import matplotlib.pyplot as plt

import pymc as pm
import arviz as az
from scipy import stats


def main():
    miu = 5
    sigma = 2

    model = pm.Model()
    with model:
        avg_wait = pm.Normal("average wait time", miu, sigma)

        trace = pm.sample(200)

az.plot_posterior(trace)
plt.show()
