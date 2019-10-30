import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score


from sklearn.preprocessing import PolynomialFeatures
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
                 header=None,sep='\s+')
df.columns = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]
X = df[["LSTAT"]].values
y = df["MEDV"].values
X_log = np.log(X)
y_sqrt = np.sqrt(y)
X_fit = np.arange(X_log.min()-1,X_log.max()+1,1)[:,np.newaxis]

regr = LinearRegression()
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)

regr.fit(X_log,y_sqrt)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y,regr.predict(X_log))

plt.scatter(X_log,y_sqrt,label="training points",color="lightgray")
plt.plot(X_fit,y_lin_fit,label = 'linear(d=1),$R^2=%.2f$'%linear_r2,color='blue',lw=2,linestyle=':')
plt.xlabel("log(% lower status of the population[LSTAT])")
plt.ylabel("$\sqrt{Price \;in\; \$1000\'s [MEDV]}$")
plt.legend(loc="upper right")
plt.show()