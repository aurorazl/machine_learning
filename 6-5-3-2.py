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
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,make_scorer
from sklearn.metrics import roc_curve,auc
from scipy import interp
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",header=None)
X = df.loc[:,2:].values
y = df.loc[:,1].values
le = LabelEncoder()
y = le.fit_transform(y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
X_train2 =X_train[:,[4,14]]
pipe_svc = Pipeline([('scl',StandardScaler()),("clf",SVC(random_state=1))])
pipe_svc.fit(X_train2,y_train)
y_pred2 = pipe_svc.predict(X_test[:,[4,14]])
print(roc_auc_score(y_true=y_test,y_score=y_pred2))
print(accuracy_score(y_true=y_test,y_pred=y_pred2))

pre_scorer = make_scorer(score_func=precision_score,pos_label=1,greater_is_better=True,average="micro")
