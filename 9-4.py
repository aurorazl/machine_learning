import pickle
import re
import os
from movieclassifier.vectorizer import vect
import numpy as np

clf = pickle.load(open(os.path.join("movieclassifier","pkl_objects","classifier.pkl"),'rb'))

def classify(document):
    label = {0: 'negative', 1: "positive"}
    X = vect.transform(document)
    y = clf.predict(X)[0]
    proba = np.max(clf.predict_proba(X))
    return label[y],proba

def train(document,y):
    X = vect.transform(document)
    clf.partial_fit(X,[y])

def feedback(feedback,review,prediction):
    inv_label = {0: 'negative', 1: "positive"}
    y = inv_label[prediction]
    if feedback =='Incorrect':
        y = int(not(y))
    train(review,y)