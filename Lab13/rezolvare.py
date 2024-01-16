from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import arviz as az

print("Urmatoarele date sunt ale modelului centrat")
data_centered = az.load_arviz_data("centered_eight")

num_chains1 = data_centered.posterior.chain.size
print(f'Numarul de lanturi: {num_chains1}')

total_samples1 = data_centered.posterior.draw.size
print(f'Marimea totala a esantionului generat: {total_samples1}')

posterior_distribution1 = data_centered.posterior
print(f'Distributia a posteriori:\n{posterior_distribution1}')

# summary_centered = az.summary(data_centered)

print("Urmatoarele date sunt ale modelului necentrat")
data_non_centered = az.load_arviz_data("non_centered_eight")

num_chains2 = data_non_centered.posterior.chain.size
print(f'Numarul de lanturi: {num_chains2}')

total_samples2 = data_non_centered.posterior.draw.size
print(f'Marimea totala a esantionului generat: {total_samples2}')

posterior_distribution2 = data_non_centered.posterior
print(f'Distributia a posteriori:\n{posterior_distribution2}')

# summary_non_centered = az.summary(data_non_centered)

# print(data_centered.posterior.to_dataframe().info())
# print(data_non_centered.posterior.to_dataframe().info())

# data_non_centered.posterior.to_dataframe().drop('theta_t', axis=1)

# print(data_centered.posterior.to_dataframe().info())
# print(data_non_centered.posterior.to_dataframe().info())

# summaries = pd.concat([summary_centered, summary_non_centered], axis=1)
# summaries.index = ['centered', 'non_centered']
# print(summaries)

print("aceasta este divergenta")
print(data_centered.sample_stats.diverging.sum())
print(data_non_centered.sample_stats.diverging.sum())

az.plot_parallel(data_centered)
az.plot_parallel(data_non_centered)
plt.show()