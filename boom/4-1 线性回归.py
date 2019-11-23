from sklearn import datasets

import numpy as np

iris = datasets.load_iris()
x = iris['data'][:,(2,3)]
y=iris['target']

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(multi_class='multinomial',solver='lbfgs',C=10)
lr.fit(x,y)
print(lr.predict_proba([[5,2]]))