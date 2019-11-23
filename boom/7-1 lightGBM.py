import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler

# df_wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",header=None)
# df_wine.columns = ['Class label',"Alcohol",
#                    'Malic acid','Ash',
#                    "Alcalinity of ash",'Magensium',
#                    'Total phenols','Flavanoids',
#                    "Nonflavanoid",'Proanthocyanins',
#                    'Color Intensity','Hue',
#                    "OD280/OD315 of diluted wines",
#                    "Proline"
#                    ]
# X,y = df_wine.iloc[:,1:].values,df_wine.iloc[:,0].values
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
# feat_labels = df_wine.columns[1:]

# from sklearn import datasets
# iris = datasets.load_iris()
# X = iris.data[:,[2,3]]
# y=iris.target
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
# feat_labels = iris.feature_names

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
                 header=None,sep='\s+')
df.columns = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]
# X,y = df.iloc[:,:-1].values,np.where(df.iloc[:,-1].values>20,1,0)
X,y = df.iloc[:,:-1].values,df.iloc[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
feat_labels = df.columns[:-1]
sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

# df = pd.read_csv("../movie_data.csv")
# from nltk.corpus import stopwords
# stop = stopwords.words("english")
# import re
# def preprocessor(text):
#     text = re.sub('<[^>]*>','',text)
#     emotions = re.findall('(?::|;|=)(?:-)?(:\)|\(|D|P)',text)
#     text = re.sub("[\W]+",' ',text.lower())+"".join(emotions).replace("-",'')
#     return text
# from nltk.stem.porter import PorterStemmer
# porter = PorterStemmer()
# def tokenizer_port(text):
#     return [porter.stem(word) for word in text.split()]
# df["review"]=df["review"].apply(preprocessor)
# X_train = df.loc[:25000,"review"].values
# y_train = df.loc[:25000,"sentiment"].values
# X_test = df.loc[25000:,"review"].values
# y_test = df.loc[25000:,"sentiment"].values
# from sklearn.feature_extraction.text import TfidfVectorizer
# tfidf = TfidfVectorizer(strip_accents=None,lowercase=False,preprocessor=None,tokenizer=tokenizer_port,stop_words=stop)
# from sklearn.pipeline import Pipeline
# np.set_printoptions(threshold=np.inf)
# print(tfidf.fit_transform(X_train).toarray()[0,:])

# from sklearn.svm import SVR
# lf = SVR()
# from sklearn.tree import DecisionTreeRegressor
# lf = DecisionTreeRegressor()
# from sklearn.linear_model import LinearRegression
# lf = LinearRegression()
# from sklearn.linear_model import RANSACRegressor
# lf = RANSACRegressor()
# from sklearn.preprocessing import PolynomialFeatures
# pr = PolynomialFeatures(degree=3)
# from sklearn.ensemble import RandomForestRegressor
# lf = RandomForestRegressor()
from sklearn.ensemble import BaggingRegressor
# lf = BaggingRegressor()
# from sklearn.ensemble import AdaBoostRegressor
# lf = AdaBoostRegressor()
# from sklearn.ensemble import GradientBoostingRegressor
# lf = GradientBoostingRegressor()
# from sklearn.neighbors import KNeighborsRegressor
# lf = KNeighborsRegressor()
# from sklearn.linear_model import SGDRegressor
# lf = SGDRegressor(loss="squared_epsilon_insensitive")   # 'squared_loss'普通最小二乘法,'huber'稳健回归, 'epsilon_insensitive'线性SVM, or 'squared_epsilon_insensitive'
# lf = xgb.XGBRegressor()
lf = lgb.LGBMRegressor(min_child_samples=10)
# lf = lgb.LGBMClassifier()
# lf = xgb.XGBClassifier()
# lf = Pipeline([('vect',tfidf),('clf',lf)])
# lf.fit(pr.fit_transform(X_train),y_train)
lf.fit(X_train,y_train)
# y_pred = lf.predict(X_test)
# importances = lf.feature_importances_
# indices = np.argsort(importances)[::-1]
# for f in range(X_train.shape[1]):
#     print("%2d) %-*s %f" % (f+1,30,feat_labels[f],importances[indices[f]]))
# print(precision_score(y_test,y_pred,average='weighted'))
# print(lf.score(X_test,y_test))
# print(lf.predict(X_test))
import matplotlib.pyplot as plt
# plt.plot(X_test,lf.predict(X_test))


y_train_pred=lf.predict(X_train)
# y_train_pred=lf.predict(pr.fit_transform(X_train))
y_test_pred = lf.predict(X_test)
# y_test_pred = lf.predict(pr.fit_transform(X_test))
plt.scatter(y_train_pred,y_train_pred-y_train,c="blue",marker='o',label="Training Data")
plt.scatter(y_test_pred,y_test_pred-y_test,c="lightgreen",marker='s',label="Test Data")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc="upper left")
plt.hlines(y=0,xmin=-10,xmax=50,lw=2,colors="red")
plt.xlim([-10,50])
plt.show()

from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_train,y_train_pred))
print(mean_squared_error(y_test,y_test_pred))

from sklearn.metrics import r2_score
print(r2_score(y_train,y_train_pred))
print(r2_score(y_test,y_test_pred))



