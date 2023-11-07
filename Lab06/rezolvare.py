import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

Y_values = [0, 5, 10]
θ_values = [0.2, 0.5]

if __name__ == '__main__':

    model = pm.Model()
    count = 0

    for y in Y_values:
        for θ in θ_values:     
            with model:
                name_pois = f'Poisson_y{y}_theta{θ}'

                n = pm.Poisson(name_pois, mu=10)

                count += 1
                name_like = f'Binomial_y{y}_number{count}'
                Y_observed = pm.Binomial(name_like, n=n, p=θ, observed=y)

            with model:
                trace = pm.sample(100)

    az.plot_posterior(trace)
    plt.show()
