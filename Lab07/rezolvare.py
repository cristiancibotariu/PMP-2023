import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt
import numpy as np
import arviz as az

df = pd.read_csv('auto-mpg.csv')

needed_columns = df[['mpg', 'horsepower']]

# needed_columns = needed_columns[needed_columns['horsepower'] > '0']

needed_columns['mpg'] = pd.to_numeric(needed_columns['mpg'], errors='coerce')
needed_columns = needed_columns.dropna(subset=['mpg'])

needed_columns['horsepower'] = pd.to_numeric(needed_columns['horsepower'], errors='coerce')
needed_columns = needed_columns.dropna(subset=['horsepower'])

mpg_data = needed_columns['mpg'].values
horsepower_data = needed_columns['horsepower'].values

if __name__ == '__main__':
    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0)
        beta = pm.Normal('beta', mu=0)

        mu = alpha + beta * horsepower_data
        likelihood = pm.Normal('mpg', mu=mu, observed=mpg_data)

        trace = pm.sample(100)

    az.plot_posterior(trace)
    plt.show()
