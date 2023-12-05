import pymc as pm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import arviz as az

def read_data():  
    file_path = 'Admission.csv'
    df = pd.read_csv(file_path)
    df = df[df['GPA'] != '?']
    df = df[df['GRE'] != '?']

    gpa = df['GPA'].values.astype(float)
    gre = df['GRE'].values.astype(int)
    admitted = df['Admission'].values.astype(int)

    return np.array(gpa), np.array(gre), np.array(admitted)

def main():
    gpa_array, gre_array, admitted_array = read_data()
    model = pm.Model()

    with model:
        beta0 = pm.Normal('beta0', mu=0, sigma=10)
        beta1 = pm.Normal('beta1', mu=0, sigma=1)
        beta2 = pm.Normal('beta2', mu=0, sigma=5)

        niu = pm.Logistic('niu', beta0 + gre_array * beta1 + gpa_array*beta2)
        admission_pred = pm.Normal('admission_pred', mu=niu, observed=admitted_array)
        idata = pm.sample(100, tune=0, chains=1)

    az.plot_trace(idata, var_names=['beta0', 'beta1', 'beta2',])
    plt.show()


if __name__ == "__main__":
    np.random.seed(1)
    main()
