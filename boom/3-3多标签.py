import os
import struct
import numpy as np

def load_mnist(path,kind='train'):
    labels_path = os.path.join(path,"%s-labels.idx1-ubyte"%kind)
    images_path = os.path.join(path,"%s-images.idx3-ubyte"%kind)

    with open(labels_path,'rb') as lbpath:
        magic,n = struct.unpack(">II",lbpath.read(8))
        labels = np.fromfile(lbpath,dtype=np.uint8)

    with open(images_path,'rb') as imgpath:
        magic,num,row,cols = struct.unpack(">IIII",imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels),784)

    return images,labels

X_train,y_train = load_mnist('../',kind='train')
X_test,y_test = load_mnist('../',kind='t10k')
some_digit = X_train[36000]
some_digit_image = some_digit.reshape(28,28)
import matplotlib.pyplot as plt
import matplotlib

from sklearn.neighbors import KNeighborsClassifier
y_train_large = (y_train>=7)
y_train_odd = (y_train%2==1)
y_mutli = np.c_[y_train_large,y_train_odd]
knn = KNeighborsClassifier()
# knn.fit(X_train,y_mutli)
# print(knn.predict([some_digit]))

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
y_train_pred = cross_val_predict(knn,X_train,y_train,cv=3)
# knn.fit(X_train,y_mutli)
# y_train_pred = knn.predict(X_train)
print(f1_score(y_train,y_train_pred,average='macro'))