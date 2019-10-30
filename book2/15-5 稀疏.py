import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np
from six.moves import urllib
import matplotlib.pyplot as plt
import errno
import os
import zipfile

import numpy.random as rnd

from functools import partial
import os
import sys
import numpy as np
import struct
from sklearn.preprocessing import StandardScaler
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
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
from sklearn.preprocessing import StandardScaler,MinMaxScaler
sd = MinMaxScaler()
X_train = sd.fit_transform(X_train)
X_test = sd.fit_transform(X_test)

def plot_image(image, shape=[28, 28]):
    plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
    plt.axis("off")
def show_reconstructed_digits(X, outputs, model_path = None, n_test_digits = 2):
    with tf.Session() as sess:
        if model_path:
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
        x_test = X_test[:n_test_digits]
        outputs_val = outputs.eval(feed_dict={X: x_test})

    fig = plt.figure(figsize=(8, 3 * n_test_digits))
    for digit_index in range(n_test_digits):
        plt.subplot(n_test_digits, 2, digit_index * 2 + 1)
        plot_image(X_test[digit_index])
        plt.subplot(n_test_digits, 2, digit_index * 2 + 2)
        plot_image(outputs_val[digit_index])
    plt.show()

# p = 0.1
# q = np.linspace(0.001, 0.999, 500)
# kl_div = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
# mse = (p - q)**2
# plt.plot([p, p], [0, 0.3], "k:")
# plt.text(0.05, 0.32, "Target\nsparsity", fontsize=14)
# plt.plot(q, kl_div, "b-", label="KL divergence")
# plt.plot(q, mse, "r--", label="MSE")
# plt.legend(loc="upper left")
# plt.xlabel("Actual sparsity")
# plt.ylabel("Cost", rotation=0)
# plt.axis([0, 1, 0, 0.95])
# plt.show()

n_inputs = 28 * 28
n_hidden1 = 1000  # sparse codings
n_outputs = n_inputs

def kl_divergence(p, q):
    # Kullback Leibler divergence
    return p * tf.log(p / q) + (1 - p) * tf.log((1 - p) / (1 - q))

learning_rate = 0.01
sparsity_target = 0.1
sparsity_weight = 0.2

X = tf.placeholder(tf.float32, shape=[None, n_inputs])            # not shown in the book

# hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.softmax) # not shown
hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.sigmoid) # not shown
outputs = tf.layers.dense(hidden1, n_outputs)                     # not shown

hidden1_mean = tf.reduce_mean(hidden1, axis=0) # batch mean
sparsity_loss = tf.reduce_sum(kl_divergence(sparsity_target, hidden1_mean))
reconstruction_loss = tf.reduce_mean(tf.square(outputs - X)) # MSE
# reconstruction_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=X, logits=outputs))
loss = reconstruction_loss + sparsity_weight * sparsity_loss

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

saver = tf.train.Saver()
init = tf.global_variables_initializer()

n_epochs = 100
batch_size = 1000
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("D:\handson-ml\datasets\mnist")
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        mini = np.array_split(range(y_train.shape[0]), 60)
        for idx in mini:
            X_batch, y_batch = X_train[idx], y_train[idx]  # not shown
        # n_batches = mnist.train.num_examples // batch_size
        # np.set_printoptions(threshold=np.inf)
        # for iteration in range(n_batches):
        #     X_batch, y_batch = mnist.train.next_batch(batch_size)
            # print(X_batch==X_train[mini[iteration]])
            # print(X_batch[X_batch==X_train[mini[iteration]]])
            # print(X_train[mini[iteration]][X_batch==X_train[mini[iteration]]])
            sess.run(training_op, feed_dict={X: X_batch})
        reconstruction_loss_val, sparsity_loss_val, loss_val = sess.run([reconstruction_loss, sparsity_loss, loss], feed_dict={X: X_batch})
        print("\r{}".format(epoch), "Train MSE:", reconstruction_loss_val, "\tSparsity loss:", sparsity_loss_val, "\tTotal loss:", loss_val)
        saver.save(sess, "./my_model_sparse.ckpt")

# show_reconstructed_digits(X, outputs, "./my_model_sparse.ckpt")