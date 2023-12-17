import pymc as pm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import arviz as az

def read_data():  
    file_path = 'Prices.csv'
    df = pd.read_csv(file_path)
    df = df[df['Price'] != '?']

    price = df['Price'].values.astype(int)
    speed = df['Speed'].values.astype(int)
    hard_drive = df['HardDrive'].values.astype(int)

    return np.array(price), np.array(speed), np.array(hard_drive)

def read_data_b(): # 
    file_path = 'Prices.csv'
    df = pd.read_csv(file_path)

    df_filtered = df[(df['Speed'] == 33) & (df['HardDrive'] == 540)]

    df_filtered = df_filtered.dropna(subset=['Price'])

    price = df_filtered['Price'].values.astype(float)
    speed = df_filtered['Speed'].values.astype(float)
    hard_drive = df_filtered['HardDrive'].values.astype(float)

    return np.array(price), np.array(speed), np.array(hard_drive)

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
        idata = pm.sample(20, tune=20, chains=1)

    az.plot_trace(idata, var_names=['alpha', 'beta1', 'beta2', 'eps'])
    plt.show()


if __name__ == "__main__":
    np.random.seed(1)
    main()
