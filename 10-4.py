import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
                 header=None,sep='\s+')
df.columns = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]

def lin_regplot(X,y,model):
    plt.scatter(X,y,c="blue")
    plt.plot(X,model.predict(X),color='red')

ransac = RANSACRegressor(LinearRegression(),max_trials=100,min_samples=50,
                         residual_threshold=5.0,random_state=0)
X = df[['RM']].values
y = df["MEDV"].values
ransac.fit(X,y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.arange(3,10,1)
line_y_ransac = ransac.predict(line_X[:,np.newaxis])
plt.scatter(X[inlier_mask],y[inlier_mask],c="blue",marker='o',label="Inliers")
plt.scatter(X[outlier_mask],y[outlier_mask],c="lightgreen",marker='s',label="Outliers")
plt.plot(line_X,line_y_ransac,color="red")
plt.xlabel("Average number of roots [RM] (standardized)")
plt.ylabel("Price in $1000\"s [MEDV] (standardized)")
plt.legend(loc="upper left")
plt.show()

print(ransac.estimator_.coef_)
print(ransac.estimator_.intercept_)