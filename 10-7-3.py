import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.tree import DecisionTreeRegressor
def lin_regplot(X,y,model):
    plt.scatter(X,y,c="blue")
    plt.plot(X,model.predict(X),color='red')
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
                 header=None,sep='\s+')
df.columns = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]
X = df[["LSTAT"]].values
y = df["MEDV"].values
X_fit = np.arange(X.min()-1,X.max()+1,1)[:,np.newaxis]

tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X,y)

y_lin_fit = tree.predict(X_fit)
r2 = r2_score(y,tree.predict(X))

sort_idx = X.flatten().argsort()
lin_regplot(X[sort_idx],y[sort_idx],tree)
# plt.scatter(X,y,label="training points",color="lightgray")
# plt.plot(X_fit,y_lin_fit,label = 'linear(d=1),$R^2=%.2f$'%r2,color='blue',lw=2,linestyle=':')
plt.xlabel("% lower status of the population[LSTAT]")
plt.ylabel("Price \;in\; \$1000\'s [MEDV]")
plt.legend(loc="upper right")
plt.show()