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

X_train,y_train = load_mnist('./',kind='train')
print(X_train.shape)
print(y_train)
X_test,y_test = load_mnist('./',kind='t10k')
print(X_test.shape)

import matplotlib.pyplot as plt

# fig,ax = plt.subplots(nrows=2,ncols=5,sharex=True,sharey=True)
# ax = ax.flatten()
# for i in range(10):
#     img = X_train[y_train==i][0].reshape(28,28)
#     ax[i].imshow(img,cmap='Greys',interpolation='nearest')
# ax[0].set_xticks([])
# ax[0].set_yticks([])
# plt.tight_layout()
# plt.show()

# fig,ax = plt.subplots(nrows=5,ncols=5,sharex=True,sharey=True)
# ax = ax.flatten()
# for i in range(25):
#     img = X_train[y_train==7][i].reshape(28,28)
#     ax[i].imshow(img,cmap='Greys',interpolation='nearest')
# ax[0].set_xticks([])
# ax[0].set_yticks([])
# plt.tight_layout()
# plt.show()

# np.savetxt("train_img.csv",X_train,fmt='%i',delimiter=',')
# np.savetxt("train_labels.csv",y_train,fmt='%i',delimiter=',')
# np.savetxt("test_img.csv",X_test,fmt='%i',delimiter=',')
# np.savetxt("test_labels.csv",y_test,fmt='%i',delimiter=',')

# X_train = np.genfromtxt("train_img.csv",dtype=int,delimiter=',')
# y_train = np.genfromtxt("train_labels.csv",dtype=int,delimiter=',')
# X_test = np.genfromtxt("test_img.csv",dtype=int,delimiter=',')
# y_test = np.genfromtxt("test_labels.csv",dtype=int,delimiter=',')