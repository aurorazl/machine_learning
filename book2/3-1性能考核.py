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
sgd_clf = SGDClassifier(random_state=42)
y_train_9 = (y_train==9)
sgd_clf.fit(X_train,y_train_9)
print(sgd_clf.predict([some_digit]))
y_scores = sgd_clf.decision_function([some_digit])
print(y_scores)
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf,X_train,y_train_9,cv=3)
print(y_train_pred)

y_scores = cross_val_predict(sgd_clf,X_train,y_train_9,cv=3,method='decision_function')
print(y_scores)
from sklearn.metrics import precision_recall_curve
precisions,recalls,thresholds = precision_recall_curve(y_train_9,y_scores)
plt.plot(thresholds,precisions[:-1],'b--',label='precision')
plt.plot(thresholds,recalls[:-1],'g-',label='precision')
plt.ylim([0,1])
plt.show()