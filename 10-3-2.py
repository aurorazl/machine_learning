import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
                 header=None,sep='\s+')
df.columns = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]

def lin_regplot(X,y,model):
    plt.scatter(X,y,c="blue")
    plt.plot(X,model.predict(X),color='red')

slr = LinearRegression()
X = df[['RM']].values
y = df["MEDV"].values
slr.fit(X,y)
# print(slr.coef_[0])
# print(slr.intercept_)

lin_regplot(X,y,slr)
plt.xlabel("Average number of roots [RM] (standardized)")
plt.ylabel("Price in $1000\"s [MEDV] (standardized)")
plt.show()
