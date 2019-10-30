import pandas as pd
from io import StringIO
from sklearn.preprocessing import Imputer
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

df = pd.DataFrame([
    ['green','M',10.1,'class1'],
    ['read','L',13.5,'class2'],
    ['blue','XL',15.3,'class1'],
    ]
)
df.columns = ["color",'size','price','classlabel']
size_mapping = {
    'XL':3,
    'L':2,
    'M':1,
}
# inv_size_mapping={v:k for k,v in size_mapping.items()}
# df["size"] =df["size"].map(size_mapping)
# class_mapping = {label:idx for idx,label in enumerate(np.unique(df["classlabel"]))}
# df["classlabel"] =df["classlabel"].map(class_mapping)
# class_le = LabelEncoder()
# y = class_le.fit_transform(df["classlabel"].values)
# print(y)
# cl = class_le.inverse_transform(y)
X = df[["color",'size','price']].values
color_le = LabelEncoder()
# X[:,0] = color_le.fit_transform(X[:,0])
ohe = OneHotEncoder(categorical_features=[0],sparse=False)
# ohe.fit_transform(X).toarray()
print(ohe.fit_transform(X))

# print(pd.get_dummies(df[["price",'color','size']]))