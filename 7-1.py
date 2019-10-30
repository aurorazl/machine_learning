from scipy.misc import comb
import math

def ensemble_error(n_classifier,error):
    k_start = math.ceil(n_classifier/2.0)
    probs = [comb(n_classifier,k)*error**k *(1-error)**(n_classifier-k)
             for k in range(k_start,n_classifier+1)]
    return sum(probs)

import numpy as np
error_range = np.arange(0.0,1.01,0.01)
ens_errors = [ensemble_error(11,error) for error in error_range]
import matplotlib.pyplot as plt
plt.plot(error_range,ens_errors,label="Ensemble error",lw=2)
plt.plot(error_range,error_range,label="Base error",lw=2,linestyle='--')
plt.xlabel("Base error")
plt.ylabel("Base/Ensemble error")
plt.legend(loc='upper left')
plt.grid()
plt.show()
