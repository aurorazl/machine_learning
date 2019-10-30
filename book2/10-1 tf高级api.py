import tensorflow as tf
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

x,y = load_mnist('../',kind='train')
X_test,y_test = load_mnist('../',kind='t10k')
print(x.shape,y.shape)
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_mldata

sd = StandardScaler()
x_sd = sd.fit_transform(x)
x_test = sd.fit_transform(X_test)
feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(x_sd)
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[300,100],n_classes=10,feature_columns=feature_columns)
dnn_clf.fit(x=x_sd,y=y.astype(np.int32),batch_size=50,steps=40000)

from sklearn.metrics import accuracy_score
# y_pred = list(dnn_clf.predict(x_test[[1],:]))
# print(accuracy_score(y_test,y_pred))
print(dnn_clf.evaluate(x_test, y_test.astype(np.int32)))