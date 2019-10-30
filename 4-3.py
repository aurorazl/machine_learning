import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.linear_model import LogisticRegression
df_wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",header=None)
df_wine.columns = ['Class label',"Alcohol",
                   'Malic acid','Ash',
                   "Alcalinity of ash",'Magensium',
                   'Total phenols','Flavanoids',
                   "Nonflavanoid",'Proanthocyanins',
                   'Color Intensity','Hue',
                   "OD280/OD315 of diluted wines",
                   "Proline"
                   ]
X,y = df_wine.iloc[:,1:].values,df_wine.iloc[:,0].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
# mms = MinMaxScaler()
# X_train_norm = mms.fit_transform(X_train)
# X_test_norm = mms.fit_transform(X_test)
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.fit_transform(X_test)
lr = LogisticRegression(penalty="l1",C=0.1)
lr.fit(X_train_std,y_train)
print(lr.score(X_train_std,y_train))
print(lr.score(X_test_std,y_test))