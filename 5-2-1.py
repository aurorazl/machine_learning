import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA

def plot_decision_regions(X,y,classifier,resolution=0.02):
    markers=("s",'x','o',"^",'v')
    colors=("red","blue","lightgreen","gray","cyan")
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min,x1_max = X[:,0].min()-1, X[:,0].max()+1
    x2_min,x2_max = X[:,1].min()-1, X[:,1].max()+1
    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,Z,alpha=0.4,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0],y=X[y==cl,1],alpha=0.8,c=cmap(idx),marker=markers[idx],label=cl)

df_wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",header=None)
X,y = df_wine.iloc[:,1:].values,df_wine.iloc[:,0].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.fit_transform(X_test)
np.set_printoptions(precision=4)
mean_vecs = []
for label in range(1,4):
    mean_vecs.append(
        np.mean(X_train_std[y_train==label],axis=0)
    )
    # print(label,mean_vecs[label-1])
d=13
S_W = np.zeros((d,d))
for label,mv in zip(range(1,4),mean_vecs):
    # class_scatter = np.zeros((d,d))
    # for row in X_train_std[y_train==label]:
    #     row,mv = row.reshape(d,1),mv.reshape(d,1)
    #     class_scatter += (row-mv).dot((row-mv).T)
    class_scatter = np.cov(X_train_std[y_train==label].T)
    S_W += class_scatter

mean_overall = np.mean(X_train_std,axis=0)
S_B = np.zeros((d,d))
for i,mean_vec in enumerate(mean_vecs):
    n=X[y==i+1,:].shape[0]
    mean_overall = mean_overall.reshape(d,1)
    S_B += n*(mean_vec-mean_overall).dot((mean_vec-mean_overall).T)
# print(S_B.shape[0],S_B.shape[1])

eigen_vals,eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
eigen_pairs = [(np.abs(eigen_vals[i]),eigen_vecs[:,i]) for i in range(len(eigen_vals))]
eigen_pairs = sorted(eigen_pairs,key=lambda k:k[0],reverse=True)
# print(eigen_pairs)

# tot = sum(eigen_vals)
# var_exp = [(i/tot) for i in sorted(eigen_vals.real,reverse=True)]
# cum_var_exp = np.cumsum(var_exp)
# plt.bar(range(1,14),var_exp,alpha=0.5,align='center',label='individual explained variance')
# plt.step(range(1,14),cum_var_exp,where='mid',label='cumulative explained variance')
# plt.ylabel("Explained variance ratio")
# plt.xlabel("Principal components")
# plt.ylim([-0.1,1.1])
# plt.legend(loc='best')
# plt.show()

W = np.hstack((eigen_pairs[0][1][:,np.newaxis].real,eigen_pairs[1][1][:,np.newaxis].real))
X_train_lda = X_train_std.dot(W)
colors = ['r','b','g']
markers = ['s','x','o']
for l,c,m in zip(np.unique(y_train),colors,markers):
    plt.scatter(X_train_lda[y_train==l,0],X_train_lda[y_train==l,1],c=c,label=l,marker=m)
plt.xlabel("LD 1")
plt.ylabel("LD 2")
plt.legend(loc="lower left")
plt.show()