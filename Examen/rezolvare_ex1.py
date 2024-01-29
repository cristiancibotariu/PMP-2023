import pymc as pm
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az
import pytensor as pt

#Punctul a)
BostonHousing = pd.read_csv('BostonHousing.csv')
y = BostonHousing['medv'].values
x_1 = BostonHousing['rm'].values
x_2 = BostonHousing['crim'].values
x_3 = BostonHousing['indus'].values

#Punctul b)
model = pm.Model()

if __name__ == '__main__':
    with model:
        α = pm.Normal('α', mu=0, sigma=10)
        β1 = pm.Normal('β1', mu=0, sigma=10)
        β2 = pm.Normal('β2', mu=0, sigma=10)
        β3 = pm.Normal('β3', mu=0, sigma=10)
        ϵ = pm.HalfCauchy('ϵ', 5000)
        ν = pm.Exponential('ν', 1/30)
        μ = pm.Deterministic('μ',α + β1*x_1 + β2*x_2 + β1*x_3)#am facut o distributie determinista pentru a tine cont de 
                                                            #toti factorii

        y_pred = pm.StudentT('y_pred', mu=μ, sigma=ϵ, nu=ν, observed=y)

        idata = pm.sample(100, tune=0, chains=2, return_inferencedata=True)

    #Punctul c)
    az.plot_forest(idata,hdi_prob=0.95,var_names=['β1','β2','β3'])
    az.summary(idata,hdi_prob=0.95,var_names=['β1','β2','β3'])

    #Punctul d)
    pm.set_data(model=model)
    ppc = pm.sample_posterior_predictive(idata, model=model)
    y_ppc = ppc.posterior_predictive['y_pred'].stack(sample=("chain", "draw")).values
    az.plot_posterior(y_ppc,hdi_prob=0.5)
    plt.show()