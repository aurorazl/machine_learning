from sklearn.feature_extraction.text import HashingVectorizer
import re
import os
import pickle

cur_dir = os.path.dirname(__file__)
stop = pickle.load(open(os.path.join(cur_dir,"pkl_objects","stopwords.pkl"),"rb"))

def tokenizer(text):
    text = re.sub('<[^>]*>','',text)
    emotions = re.findall('(?::|;|=)(?:-)?(:\)|\(|D|P)',text.lower())
    text = re.sub("[\W]+",' ',text.lower())+"".join(emotions).replace("-",'')
    return [word for word in text.split() if word not in stop]

vect = HashingVectorizer(decode_error='ignore',n_features=2**21,preprocessor=None,
                         tokenizer=tokenizer)