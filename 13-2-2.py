import os
import struct
import numpy as np
from scipy.special import expit
import sys

X = np.array([[1,1.4,1.5]])
w = np.array([0.0,0.2,0.4])
def net_input(X,w):
    z = X.dot(w)
    return z
def logistic(z):
    return 1.0/(1.0+np.exp(-z))
def logistic_activation(X,w):
    z = net_input(X,w)
    return logistic(z)
print(logistic_activation(X,w)[0])

W = np.array([[1.1,1.3,1.3,0.5],
              [0.1,0.2,0.4,0.1],
              [0.2,0.5,2.1,1.9]])
A = np.array([[1.0],
              [0.1],
              [0.3],
              [0.7]])
Z = W.dot(A)
y_probas = logistic(Z)
print(y_probas)
y_class = np.argmax(Z,axis=0)
print(y_class)

def softmax(z):
    return np.exp(z)/np.sum(np.exp(z))
def softmax_activation(X,w):
    z = net_input(X,w)
    return softmax(z)

print(softmax_activation(W,A))
