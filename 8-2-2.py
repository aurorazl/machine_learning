import pyprind
import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer


count = CountVectorizer()
docs = np.array(["The sum is shinning","The weather is sweet","The sun is shinning and the weather is sweet"])
bag = count.fit_transform(docs)
tfidf = TfidfTransformer()
np.set_printoptions(precision=2)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(strip_accents=None,lowercase=False,preprocessor=None)
print(tfidf.fit_transform(docs))