from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import numpy as np

iris = datasets.load_iris()
x = iris['data'][:,(2,3)]
y=iris['target']


svm = Pipeline([('scaler',StandardScaler()),('linear_svc',LinearSVC(C=1,loss='hinge')),])
svm.fit(x,y)
print(svm.predict([[5,2]]))