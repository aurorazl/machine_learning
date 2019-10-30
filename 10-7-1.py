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
regr = LinearRegression()
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)

X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)


X_fit = np.arange(X.min(),X.max(),1)[:,np.newaxis]
regr.fit(X,y)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y,regr.predict(X))

regr.fit(X_quad,y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(y,regr.predict(X_quad))

regr.fit(X_cubic,y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y,regr.predict(X_cubic))

plt.scatter(X,y,label="training points",color="lightgray")
plt.plot(X_fit,y_lin_fit,label = 'linear(d=1),$R^2=%.2f$'%linear_r2,color='blue',lw=2,linestyle=':')
plt.plot(X_fit,y_quad_fit,label = 'quadratic(d=2),$R^2=%.2f$'%quadratic_r2,color='blue',lw=2,linestyle='-')
plt.plot(X_fit,y_cubic_fit,label = 'cubic(d=3),$R^2=%.2f$'%cubic_r2,color='green',lw=2,linestyle='--')
plt.xlabel("% lower status of the population[LSTAT]")
plt.ylabel("Price in $1000\'s [MEDV]")
plt.legend(loc="upper right")
plt.show()