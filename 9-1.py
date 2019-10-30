import pyprind
import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords
stop = stopwords.words("english")

def tokenizer(text):
    text = re.sub('<[^>]*>','',text)
    emotions = re.findall('(?::|;|=)(?:-)?(:\)|\(|D|P)',text.lower())
    text = re.sub("[\W]+",' ',text.lower())+"".join(emotions).replace("-",'')
    return [word for word in text.split() if word not in stop]

def stream_docs(path):
    with open(path,'r',encoding='UTF-8') as csv:
        next(csv)
        for line in csv:
            text,label = line[:-3],int(line[-2])
            yield text,label

# print(next(stream_docs("./movie_data.csv")))

def get_minibatch(doc_stream,size):
    docs,y=[],[]
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None,None
    return docs,y

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
vect = HashingVectorizer(decode_error='ignore',n_features=2**21,preprocessor=None,
                         tokenizer=tokenizer)
clf = SGDClassifier(loss='log',random_state=1)
doc_stream = stream_docs(path="./movie_data.csv")

import pyprind
pbar = pyprind.ProgBar(45)
classes = np.array([0,1])
for _ in range(45):
    X_train,y_train = get_minibatch(doc_stream,size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train,y_train,classes=classes)
    pbar.update()

X_test,y_test = get_minibatch(doc_stream,size=5000)
X_test = vect.transform(X_test)
print(clf.score(X_test,y_test))

import pickle
import os
dest = os.path.join('movieclassifier','pkl_objects')
if not os.path.exists(dest):
    os.makedirs(dest)
pickle.dump(stop,open(os.path.join(dest,"stopwords.pkl"),'wb'),protocol=4)
pickle.dump(clf,open(os.path.join(dest,"classifier.pkl"),'wb'),protocol=4)