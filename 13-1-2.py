import os
import struct
import numpy as np
from scipy.special import expit
import sys
import theano
from theano import tensor as T

x1 = T.scalar()
w1 = T.scalar()
w0 = T.scalar()
z1 = w1*x1+w0
net_input = theano.function(inputs=[w1,x1,w0],outputs=z1)
print(net_input(2.0,1.0,0.5))

theano.config.floatX = "float32"
print(theano.config.floatX)

print(theano.config.device)
