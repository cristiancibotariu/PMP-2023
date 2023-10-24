<<<<<<< HEAD
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

n_clients_mechanic2 = 1.5

p_mechanic1 = 0.4

selected_mechanic = np.random.choice([1, 2], size=10000, p=[p_mechanic1, 1 - p_mechanic1])

m1=stats.expon.rvs(scale=1/4, size=10000)
m2=stats.expon.rvs(scale=1/6, size=10000)

m2[selected_mechanic == 2] /= n_clients_mechanic2

total_service_time = np.where(selected_mechanic == 1, m1, m2)

mean_service_time = np.mean(total_service_time)
std_deviation = np.std(total_service_time)

# Afișați rezultatele
print("Media timpului de servire:", mean_service_time)
print("Deviatia standard a timpului de servire:", std_deviation)

az.plot_posterior({'x':total_service_time}) # Afisarea aproximarii densitatii probabilitatilor, mediei, intervalului etc. variabilelor x,y,z
=======
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
>>>>>>> 3631879036d542fa0689bdb11f257cf7f57a4e9a
plt.show() 