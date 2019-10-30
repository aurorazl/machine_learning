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
knn = KNeighborsClassifier()
noise = np.random.randint(0,100,(len(X_train),784))
noise_2 = np.random.randint(0,100,(len(X_test),784))
X_train_noise = X_train+noise
X_test_noise= X_test+noise_2
knn.fit(X_train_noise,X_train)
plt.imshow(X_test_noise[1].reshape(28,28),cmap=matplotlib.cm.binary,interpolation='nearest')
plt.show()
y_pred = knn.predict([X_test_noise[1]]).reshape(28,28)
plt.imshow(y_pred,cmap=matplotlib.cm.binary,interpolation='nearest')
plt.show()