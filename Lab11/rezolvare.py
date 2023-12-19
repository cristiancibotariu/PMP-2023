import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from sklearn.mixture import GaussianMixture

clusters = 3
n_cluster = [200, 120, 180]
n_total = sum(n_cluster)
means = [5, 0, 20]
std_devs = [2, 0, 10]
mix = np.random.normal(np.repeat(means, n_cluster), np.repeat(std_devs, n_cluster))

plt.hist(mix, bins=30, density=True, alpha=0.5, color='blue')

for n_components in [2, 3, 4]:
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(mix.reshape(-1, 1))

    x = np.linspace(mix.min(), mix.max(), 1000)
    pdf = np.exp(gmm.score_samples(x.reshape(-1, 1)))
    plt.plot(x, pdf, label=f'{n_components} componente')

plt.title('Model de Mixtură de Distribuții Gaussiene')
plt.legend()
plt.show()
