import os
import struct
import numpy as np
from scipy.special import expit
import sys
import matplotlib.pyplot as plt
def logistic(z):
    return 1.0/(1.0+np.exp(-z))
def tanh(z):
    e_p = np.exp(z)
    e_m = np.exp(-z)
    return (e_p-e_m)/(e_p+e_m)
z = np.arange(-5,5,0.005)
# log_act = logistic(z)
from scipy.special import expit
log_act = expit(z)
# tanh_act = tanh(z)
tanh_act = np.tanh(z)

plt.ylim([-1.5,1.5])
plt.xlabel("net input $z$")
plt.ylabel("activation $\phi(z)$")
plt.axhline(1,color='black',linestyle='--')
plt.axhline(0.5,color='black',linestyle='--')
plt.axhline(0,color='black',linestyle='--')
plt.axhline(-1,color='black',linestyle='--')
plt.plot(z,tanh_act,color='black',label='tanh')
plt.plot(z,log_act,color='lightgreen',label='logistic')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
