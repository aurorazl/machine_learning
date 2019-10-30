import os
import struct
import numpy as np
from scipy.special import expit
import sys
import theano
from theano import tensor as T

X_train = np.asarray([[0.0],[1.0],[2.0],[3.0],[4.0],[5.0],[6.0],[7.0],[8.0],[9.0]],dtype='float32')
y_train = np.asarray([1.0,1.3,3.1,2.0,5.0,6.3,6.6,7.4,8.0,9.0],dtype='float32')

def train_linreg(X_train,y_train,eta,epochs):
    costs=[]
    eta0 = T.fscalar('eta0')
    y = T.fvector('y')
    X = T.fmatrix('X')
    w = theano.shared(np.zeros(shape=(X_train.shape[1]+1),dtype="float32"),name='w')
    net_input = T.dot(X,w[1:])+w[0]
    errors = y-net_input
    cost = T.sum(T.pow(errors,2))
    gradient = T.grad(cost,wrt=w)
    update = [(w,w-eta0*gradient)]
    train = theano.function(inputs=[eta0],outputs=cost,updates=update,givens={X:X_train,y:y_train,},allow_input_downcast=True)
    for _ in range(epochs):
        costs.append(train(eta))
    return costs,w
import matplotlib.pyplot as plt
costs,w = train_linreg(X_train,y_train,eta=0.001,epochs=10)
# plt.plot(range(1,len(costs)+1),costs)
# plt.tight_layout()
# plt.xlabel("Epoch")
# plt.ylabel("Cost")
# plt.show()
# print(w.get_value())

def predict_linreg(X,w):
    Xt = T.matrix(name='X')
    net_input = T.dot(Xt,w[1:])+w[0]
    predict = theano.function(inputs=[Xt],givens={w:w},outputs=net_input)
    return predict(X)
plt.scatter(X_train,y_train,marker='s',s=50)
plt.plot(range(X_train.shape[0]),predict_linreg(X_train,w),color='gray',marker='o',markersize=4,linewidth=3)
plt.xlabel('x')
plt.ylabel('y')
plt.show()