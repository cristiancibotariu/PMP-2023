import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)

i = 0
while i < 10000:
    if np.random < 0.4:
        m1=stats.expon(0,1/4)
    else:
        m2=stats.expon(0,1/6)
    

az.plot_posterior({'m1':m1, 'm2':m2})
plt.show() 