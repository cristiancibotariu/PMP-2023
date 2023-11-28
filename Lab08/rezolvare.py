import pymc as pm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import arviz as az

def read_data():    # a
    file_path = 'Prices.csv'
    df = pd.read_csv(file_path)
    df = df[df['Price'] != '?']

    price = df['Price'].values.astype(int)
    speed = df['Speed'].values.astype(int)
    hard_drive = df['HardDrive'].values.astype(int)

    return np.array(price), np.array(speed), np.array(hard_drive)

# def plot_data(price, speed, harddrive):     # a
#     plt.scatter(price, speed, harddrive, marker='o')
#     plt.xlabel('price')
#     plt.ylabel('speed')
#     plt.title('my_data')
#     plt.show()

def main():
    price_array, speed_array, harddrive_array = read_data()
    model = pm.Model()

    with model:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta1 = pm.Normal('beta1', mu=0, sigma=1)
        beta2 = pm.Normal('beta2', mu=0, sigma=5)
        eps = pm.HalfCauchy('eps', 5)


        niu = pm.Deterministic('niu', speed_array * beta1 + (np.log(harddrive_array))*beta2 + alpha)
        price_pred = pm.Normal('price_pred', mu=niu, sigma=eps, observed=price_array)
        idata = pm.sample(200, tune=200, return_inferencedata=True)

    az.plot_trace(idata, var_names=['alpha', 'beta', 'eps'])
    plt.show()


if __name__ == "__main__":
    np.random.seed(1)
    main()