import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

data = pd.read_csv("trafic.csv")

model = pm.Model()

with model:
    flag16 = 0
    flag7 = 0

    lambda_prior = pm.Exponential("lambda", 1)

    morning_change = pm.Normal("morning_change", 0, 1)
    evening_change = pm.Normal("evening_change", 0, 1)

    if (data["hour"] == 16):
        flag16 = 1

    if (data["hour"] == 7):
         flag7 = 1

    lambda_hourly = lambda_prior * (1 + morning_change * flag7 + evening_change * flag16)

    observed = pm.Poisson("observed", lambda_hourly, observed=data["traffic"])

with model:
    trace = pm.sample(1000)

az.plot_posterior(trace)
plt.show()
