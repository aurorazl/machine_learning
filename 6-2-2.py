import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",header=None)
X = df.loc[:,2:].values
y = df.loc[:,1].values
le = LabelEncoder()
y = le.fit_transform(y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
pipe_lr = Pipeline([('scl',StandardScaler()),("pca",PCA(n_components=2)),("clf",LogisticRegression(random_state=1))])

# kfold = StratifiedKFold(n_splits=10,random_state=1)
# scores = []
# for k,(train,test) in enumerate(kfold.split(X_train,y_train)):
#     pipe_lr.fit(X_train[train],y_train[train])
#     score = pipe_lr.score(X_train[test],y_train[test])
#     scores.append(score)
#     print("Fold:%s,Class dist.:%s,Acc:%.3f" %(k+1,np.bincount(y_train[train]),score))

scores = cross_val_score(estimator=pipe_lr,X=X_train,y=y_train,cv=10,n_jobs=1)
print("CV accuracy scores: %s" % scores)
print("CV accuracy: %.3f +/- %.3f" %(np.mean(scores),np.std(scores)))