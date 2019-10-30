import pickle
import re
import os
from movieclassifier.vectorizer import vect

clf = pickle.load(open(os.path.join("movieclassifier","pkl_objects","classifier.pkl"),'rb'))
import numpy as np
label = {0:'negative',1:"positive"}
example = ['I love this movie']
X = vect.transform(example)
print(label[clf.predict(X)[0]])
print(np.max(clf.predict_proba(X)))