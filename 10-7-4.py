import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


def lin_regplot(X,y,model):
    plt.scatter(X,y,c="blue")
    plt.plot(X,model.predict(X),color='red')
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
                 header=None,sep='\s+')
df.columns = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]
X = df.iloc[:,:-1].values
y = df["MEDV"].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=1)

forest = RandomForestRegressor(n_estimators=1000,criterion='mse',random_state=1,n_jobs=-1)
forest.fit(X_train,y_train)

y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)
# print(mean_squared_error(y_train,y_train_pred))
# print(mean_squared_error(y_test,y_test_pred))
# print(r2_score(y_train,y_train_pred))
# print(r2_score(y_test,y_test_pred))
plt.scatter(y_train_pred,y_train_pred-y_train,label="training data",color="black",marker='o',s=35,alpha=0.5)
plt.scatter(y_test_pred,y_test_pred-y_test,label="Test data",color="lightgreen",marker='s',s=35,alpha=0.7)
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc="upper right")
plt.hlines(y=0,xmin=-10,xmax=50,lw=2,color='red')
plt.xlim([-10,50])
plt.show()
