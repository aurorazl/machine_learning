import pyprind
import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("./movie_data.csv")
count = CountVectorizer(ngram_range=(2,2))
docs = np.array(["The sum is shinning","The weather is sweet","The sun is shinning and the weather is sweet"])
bag = count.fit_transform(docs)
print(count.vocabulary_)
print(bag.toarray())