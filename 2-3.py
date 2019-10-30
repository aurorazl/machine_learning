import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class AdalineGD(object):
    def __init__(self,eta=0.01,n_iter=50):
        self.eta = eta
        self.n_iter = n_iter
    def fit(self,X,y):
        self.w_ = np.zeros(1+X.shape[1])# 将权重初始化为零
        self.cost_ = []
        for _ in range(self.n_iter):
            output = self.net_input(X)# 与感知器中针对每一个样本做一次权重更新不同
            errors = (y-output)
            self.w_[1:]+= self.eta*X.T.dot(errors)# 计算1到m位置的权重
            self.w_[0] += self.eta*errors.sum()
            cost = (errors**2).sum()/2.0
            self.cost_.append(cost)
        return self
    def net_input(self,X):
        re = np.dot(X,self.w_[1:])+self.w_[0]
        return re
    def activation(self,X):
        return self.net_input(X)
    def predict(self,X):
        return np.where(self.activation(X)>=0.0,1,-1)

def plot_decision_regions(X,y,classifier,resolution=0.02):
    markers=("s",'x','o',"^",'v')
    colors=("red","blue","lightgreen","gray","cyan")
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min,x1_max = X[:,0].min()-1, X[:,0].max()+1
    x2_min,x2_max = X[:,1].min()-1, X[:,1].max()+1
    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,Z,alpha=0.4,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0],y=X[y==cl,1],alpha=0.8,c=cmap(idx),marker=markers[idx],label=cl)

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",header=None)
y = df.iloc[0:100,4].values
y = np.where(y=="Iris-versicolor",-1,1)
X = df.iloc[0:100,[0,2]].values

# fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(8,4))
# ada1=AdalineGD(0.01,10).fit(X,y)
# ax[0].plot(range(1,len(ada1.cost_)+1),np.log10(ada1.cost_),marker='o')
# ax[0].set_xlabel("Epochs")
# ax[0].set_ylabel("log(Sum-squared-error)")
# ax[0].set_title("Adaline-learning rate 0.01")
#
# ada2=AdalineGD(0.0001,10).fit(X,y)
# ax[1].plot(range(1,len(ada2.cost_)+1),ada2.cost_,marker='o')
# ax[1].set_xlabel("Epochs")
# ax[1].set_ylabel("Sum-squared-error")
# ax[1].set_title("Adaline-learning rate 0.0001")
# plt.show()
X_std = np.copy(X)
X_std[:,0] = (X[:,0]-X[:,0].mean())/X[:,0].std()
X_std[:,1] = (X[:,1]-X[:,1].mean())/X[:,1].std()
ada=AdalineGD(0.01,15)
print(X_std,type(X_std))
ada.fit(X_std,y)
plot_decision_regions(X_std,y,classifier=ada)
plt.xlabel("sepal length [standardized]")
plt.ylabel("petal length [standardized]")
plt.legend(loc="upper left")
plt.show()
plt.plot(range(1,len(ada.cost_)+1),ada.cost_,marker='o')
plt.xlabel("Epochs")
plt.ylabel("Sum-squared-error")
plt.show()