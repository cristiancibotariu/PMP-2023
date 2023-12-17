import pymc as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az
az.style.use('arviz-darkgrid')


def read_data(file_path):
    data = np.loadtxt(file_path)
    return data


def generate_synthetic_data(original_data):
    num_points = 500
    noise_factor = 0.1

    synthetic_data = np.zeros((num_points, original_data.shape[1]))
    synthetic_data[:, 0] = np.linspace(original_data[:, 0].min(), original_data[:, 0].max(), num_points)
    synthetic_data[:, 1] = np.interp(synthetic_data[:, 0], original_data[:, 0], original_data[:, 1]) + np.random.normal(0, noise_factor, num_points)

    return synthetic_data

dummy_data = read_data('./dummy.csv')

synthetic_data = generate_synthetic_data(dummy_data) #comenteaza linia asta daca vrei sa lucrezi pe datele din dummy
x_1 = dummy_data[:, 0]
y_1 = dummy_data[:, 1]

order = 5
x_1p = np.vstack([x_1**i for i in range(1, order+1)])
x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
y_1s = (y_1 - y_1.mean()) / y_1.std()
plt.scatter(x_1s[0], y_1s)
plt.xlabel('x')
plt.ylabel('y')
sd=np.array([10, 0.1, 0.1, 0.1, 0.1])
if __name__== '__main__':
    with pm.Model() as model_p:
        α = pm.Normal('α', mu=0, sigma=1)
        # β_stack = [pm.Normal(f'β_{i}', mu=0, sigma=sd[i]) for i in range(order)]
        # β = pm.Deterministic('β', pm.math.stack(β_stack, axis=0)) #asta este folosita pt array-ul sd
        β = pm.Normal('β', mu=0, sigma=10, shape=order)#sigma=100
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_1s)
        idata_p = pm.sample(2000, chains=1, tune=0, return_inferencedata=True)

    α_p_post = idata_p.posterior['α'].mean(("chain", "draw")).values
    β_p_post = idata_p.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = α_p_post + np.dot(β_p_post, x_1s)
    plt.plot(x_1s[0][idx], y_p_post[idx], 'C2', label=f'model order {order}')
    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()
    plt.show()