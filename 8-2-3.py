import pyprind
import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.stem.porter import PorterStemmer
import nltk
porter = PorterStemmer()

def preprocessor(text):
    text = re.sub('<[^>]*>','',text)
    emotions = re.findall('(?::|;|=)(?:-)?(:\)|\(|D|P)',text)
    text = re.sub("[\W]+",' ',text.lower())+"".join(emotions).replace("-",'')
    return text

def tokenizer(text):
    return text.split()

def tokenizer_port(text):
    return [porter.stem(word) for word in text.split()]

# df = pd.read_csv("./movie_data.csv")
# print(preprocessor(df.loc[0,'review'][-50:]))

# df["review"]=df["review"].apply(preprocessor)

# print(tokenizer_port("runners like running and thus they run"))

print(nltk.download("stopwords"))

from nltk.corpus import stopwords
stop = stopwords.words("english")
print([w for w in tokenizer_port("a runner like running and runs a lot")[-10:] if w not in stop])