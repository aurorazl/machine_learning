import os
import struct
import numpy as np
from scipy.special import expit
import sys
import theano
from theano import tensor as T

# x = T.fmatrix(name='x')
# x_sum = T.sum(x,axis=0)
# calc_sum = theano.function(inputs=[x],outputs=x_sum)
# # ary = [[1,2,3],[1,2,3]]
# # print(calc_sum(ary))
# ary = np.array([[1,2,3],[1,2,3]],dtype='float32')
# print(calc_sum(ary))
# print(x)

# x = T.fmatrix('x')
# w = theano.shared(np.asarray([[0.0,0.0,0.0]],dtype='float32'))
# z = x.dot(w.T)
# update = [[w,w+1.0]]
# net_input = theano.function(inputs=[x],updates=update,outputs=z)
# data = np.array([[1,2,3]],dtype='float32')
# for i in range(5):
#     print(net_input(data))

data = np.array([[1,2,3]],dtype='float32')
x = T.fmatrix('x')
w = theano.shared(np.asarray([[0.0,0.0,0.0]],dtype='float32'))
z = x.dot(w.T)
update = [[w,w+1.0]]
net_input = theano.function(inputs=[],updates=update,givens={x:data},outputs=z)
for i in range(5):
    print(net_input())