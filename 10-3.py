import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
                 header=None,sep='\s+')
df.columns = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]
# print(df.head())

# sns.set(style="whitegrid",context="notebook")
cols = ['LSTAT',"INDUS","NOX","RM","MEDV"]

class LinearRegressionGD(object):
    def __init__(self,eta=0.001,n_iter=20):
        self.eta = eta
        self.n_iter = n_iter
    def fit(self,X,y):
        self.w_ = np.zeros(1+X.shape[1])# 将权重初始化为零
        self.cost_ = []
        for _ in range(self.n_iter):
            output = self.net_input(X)# 与感知器中针对每一个样本做一次权重更新不同
            errors = (y-output)
            self.w_[1:]+= self.eta * (X.T.dot(errors))# 计算1到m位置的权重
            self.w_[0] += self.eta*errors.sum()
            cost = (errors**2).sum()/2.0
            self.cost_.append(cost)
        return self
    def net_input(self,X):
        re = np.dot(X,self.w_[1:])+self.w_[0]
        return re
    def predict(self,X):
        return self.net_input(X)

X = df[['RM']].values
y = df["MEDV"].values

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y.reshape(-1,1)).reshape(-1,)
lr = LinearRegressionGD()
lr.fit(X_std,y_std)

# plt.plot(range(1,lr.n_iter+1),lr.cost_)
# plt.ylabel("SSE")
# plt.ylabel("Epoch")
# plt.show()

def lin_regplot(X,y,model):
    plt.scatter(X,y,c="blue")
    plt.plot(X,model.predict(X),color='red')

# lin_regplot(X_std,y_std,lr)
# plt.xlabel("Average number of roots [RM] (standardized)")
# plt.ylabel("Price in $1000\"s [MEDV] (standardized)")
# plt.show()

# num_rooms_std = sc_x.transform(np.array([5.0]).reshape(-1,1))
# price_std = lr.predict(num_rooms_std)
# print(sc_y.inverse_transform(price_std))

print(lr.w_[1])
print(lr.w_[0])