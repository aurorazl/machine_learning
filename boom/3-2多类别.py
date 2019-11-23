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

import matplotlib.pyplot as plt
import matplotlib
some_digit = X_train[36000]
some_digit_image = some_digit.reshape(28,28)
# plt.imshow(some_digit_image,cmap=matplotlib.cm.binary,interpolation='nearest')
# plt.axis("off")
# plt.show()

from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsOneClassifier
sgd_clf = SGDClassifier(random_state=42)
ovo_clf = OneVsOneClassifier(sgd_clf)
# y_train_9 = (y_train==9)
# sgd_clf.fit(X_train,y_train)
# print(sgd_clf.predict([some_digit]))
# print(sgd_clf.decision_function([some_digit]))

# ovo_clf.fit(X_train,y_train)
# print(ovo_clf.predict([some_digit]))
# print(ovo_clf.decision_function([some_digit]))
# print(len(ovo_clf.estimators_))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
from sklearn.model_selection import cross_val_score,cross_val_predict
# print(cross_val_score(sgd_clf,X_train,y_train,cv=3,scoring='accuracy'))
# print(cross_val_score(sgd_clf,X_train_scaled,y_train,cv=3,scoring='accuracy'))

from sklearn.metrics import confusion_matrix
y_train_pred = cross_val_predict(sgd_clf,X_train_scaled,y_train,cv=3)
conf_mx = confusion_matrix(y_train,y_train_pred)
# print(conf_mx)

row_sums = conf_mx.sum(axis=1,keepdims=True)
norm_conf_mx = conf_mx/row_sums
np.fill_diagonal(norm_conf_mx,0)
plt.matshow(norm_conf_mx,cmap=plt.cm.Blues)
plt.show()