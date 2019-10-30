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

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 150  # codings
n_hidden3 = n_hidden1
n_outputs = n_inputs

learning_rate = 0.01
l2_reg = 0.0001

activation = tf.nn.elu
regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, n_inputs])

weights1_init = initializer([n_inputs, n_hidden1])
weights2_init = initializer([n_hidden1, n_hidden2])
weights3_init = initializer([n_hidden2, n_hidden3])
weights4_init = initializer([n_hidden3, n_outputs])

weights1 = tf.Variable(weights1_init, dtype=tf.float32, name="weights1")
weights2 = tf.Variable(weights2_init, dtype=tf.float32, name="weights2")
weights3 = tf.Variable(weights3_init, dtype=tf.float32, name="weights3")
weights4 = tf.Variable(weights4_init, dtype=tf.float32, name="weights4")

biases1 = tf.Variable(tf.zeros(n_hidden1), name="biases1")
biases2 = tf.Variable(tf.zeros(n_hidden2), name="biases2")
biases3 = tf.Variable(tf.zeros(n_hidden3), name="biases3")
biases4 = tf.Variable(tf.zeros(n_outputs), name="biases4")

hidden1 = activation(tf.matmul(X, weights1) + biases1)
hidden2 = activation(tf.matmul(hidden1, weights2) + biases2)
hidden3 = activation(tf.matmul(hidden2, weights3) + biases3)
outputs = tf.matmul(hidden3, weights4) + biases4

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))

optimizer = tf.train.AdamOptimizer(learning_rate)

with tf.name_scope("phase1"):
    phase1_outputs = tf.matmul(hidden1, weights4) + biases4  # bypass hidden2 and hidden3
    phase1_reconstruction_loss = tf.reduce_mean(tf.square(phase1_outputs - X))
    phase1_reg_loss = regularizer(weights1) + regularizer(weights4)
    phase1_loss = phase1_reconstruction_loss + phase1_reg_loss
    phase1_training_op = optimizer.minimize(phase1_loss)

with tf.name_scope("phase2"):
    phase2_reconstruction_loss = tf.reduce_mean(tf.square(hidden3 - hidden1))
    phase2_reg_loss = regularizer(weights2) + regularizer(weights3)
    phase2_loss = phase2_reconstruction_loss + phase2_reg_loss
    train_vars = [weights2, biases2, weights3, biases3]
    phase2_training_op = optimizer.minimize(phase2_loss, var_list=train_vars) # freeze hidden1

init = tf.global_variables_initializer()
saver = tf.train.Saver()

training_ops = [phase1_training_op, phase2_training_op]
reconstruction_losses = [phase1_reconstruction_loss, phase2_reconstruction_loss]
n_epochs = [4, 4]
batch_sizes = [150, 150]

import os
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

# with tf.Session() as sess:
#     init.run()
#     for phase in range(2):
#         print("Training phase #{}".format(phase + 1))
#         for epoch in range(n_epochs[phase]):
#             mini = np.array_split(range(y_train.shape[0]), batch_sizes[phase])
#             for idx in mini:
#                 X_batch, y_batch = X_train[idx], y_train[idx]  # not shown
#                 sess.run(training_ops[phase], feed_dict={X: X_batch})
#             loss_train = reconstruction_losses[phase].eval(feed_dict={X: X_batch})
#             print("\r{}".format(epoch), "Train MSE:", loss_train)
#             saver.save(sess, "./my_model_one_at_a_time.ckpt")
#     loss_test = reconstruction_loss.eval(feed_dict={X: X_test})
#     print("Test MSE:", loss_test)
import sys
# with tf.Session() as sess:
#     init.run()
#     for phase in range(2):
#         print("Training phase #{}".format(phase + 1))
#         if phase == 1:
#             hidden1_cache = hidden1.eval(feed_dict={X: X_train})
#         for epoch in range(n_epochs[phase]):
#             mini = np.array_split(range(y_train.shape[0]), batch_sizes[phase])
#             for idx in mini:
#                 if phase == 1:
#                     indices = rnd.permutation(y_train.shape[0])
#                     hidden1_batch = hidden1_cache[indices[:batch_sizes[phase]]]
#                     feed_dict = {hidden1: hidden1_batch}
#                     sess.run(training_ops[phase], feed_dict=feed_dict)
#                 else:
#                     X_batch, y_batch = X_train[idx], y_train[idx]  # not shown
#                     feed_dict = {X: X_batch}
#                     sess.run(training_ops[phase], feed_dict=feed_dict)
#             loss_train = reconstruction_losses[phase].eval(feed_dict=feed_dict)
#             print("\r{}".format(epoch), "Train MSE:", loss_train)
#             saver.save(sess, "./my_model_cache_frozen.ckpt")
#     loss_test = reconstruction_loss.eval(feed_dict={X:X_test})
#     print("Test MSE:", loss_test)

n_test_digits = 2
x_test = X_test[:n_test_digits]

# with tf.Session() as sess:
#     saver.restore(sess, "./my_model_one_at_a_time.ckpt") # not shown in the book
#     outputs_val = outputs.eval(feed_dict={X: x_test})

def plot_image(image, shape=[28, 28]):
    plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
    plt.axis("off")
#
# for digit_index in range(n_test_digits):
#     plt.subplot(n_test_digits, 2, digit_index * 2 + 1)
#     plot_image(X_test[digit_index])
#     plt.subplot(n_test_digits, 2, digit_index * 2 + 2)
#     plot_image(outputs_val[digit_index])
# plt.show()

with tf.Session() as sess:
    saver.restore(sess, "./my_model_one_at_a_time.ckpt") # not shown in the book
    weights1_val = weights1.eval()

for i in range(5):
    plt.subplot(1, 5, i + 1)
    plot_image(weights1_val.T[i])

plt.show()                          # not shown