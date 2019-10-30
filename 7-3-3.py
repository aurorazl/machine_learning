from sklearn.base import BaseEstimator
from sklearn.base import  ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
import operator
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from itertools import product
from sklearn.model_selection import GridSearchCV

class MajorityVoteClassifier(BaseEstimator,ClassifierMixin):
    def __init__(self,classifiers,vote='classlabel',weights=None):
        self.classifiers=classifiers
        self.name_classifiers={key:value for key,value in _name_estimators(classifiers)}
        self.vote= vote
        self.weights = weights
    def fit(self,X,y):
        self.lablenc_=LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X,self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self
    def predict(self,X):
        if self.vote == "probability":
            maj_vote = np.argmax(self.predict_proba(X),axis=1)
        else:
            predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T
            maj_vote = np.apply_along_axis(lambda x:np.argmax(np.bincount(x,weights=self.weights)),axis=1,arr=predictions)
        maj_vote = self.lablenc_.inverse_transform(maj_vote)
        return maj_vote
    def predict_proba(self,X):
        probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        avg_proba =np.average(probas,axis=0,weights=self.weights)
        return avg_proba
    def get_params(self, deep=True):
        if not deep:
            return super(MajorityVoteClassifier,self).get_params(deep=False)
        else:
            out = self.name_classifiers.copy()
            for name,step in six.iteritems(self.name_classifiers):
                for key,value in six.iteritems(step.get_params(deep=True)):
                    out["%s__%s" %(name,key)]=value
            return out

iris = datasets.load_iris()
X,y=iris.data[50:,[1,2]],iris.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5,random_state=1)
clf1 = LogisticRegression(penalty='l2',C=0.001,random_state=0)
clf2 = DecisionTreeClassifier(max_depth=1,criterion='entropy',random_state=0)
clf3 = KNeighborsClassifier(n_neighbors=1,p=2,metric='minkowski')
pipe1 = Pipeline([["sc",StandardScaler()],["clf",clf1]])
pipe2 = Pipeline([["sc",StandardScaler()],["clf",clf2]])
pipe3 = Pipeline([["sc",StandardScaler()],["clf",clf3]])
mv_clf = MajorityVoteClassifier(classifiers=[pipe1,clf2,pipe3])
params = {"decisiontreeclassifier__max_depth":[1,2],
          "pipeline-1__clf__C":[0.001,0.1,100.0]}
grid = GridSearchCV(estimator=mv_clf,param_grid=params,cv=10,scoring='roc_auc')
grid.fit(X_train,y_train)
print(grid.cv_results_)
for params,mean_score,scores in zip(grid.cv_results_["params"],grid.cv_results_["mean_test_score"],grid.cv_results_["std_test_score"]):
    print("%0.3f+/-%0.2f %r" %(mean_score,scores/2,params))
print(grid.best_params_)
print(grid.best_score_)