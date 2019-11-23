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
y = y.astype(np.int32)
X_test,y_test = load_mnist('../',kind='t10k')
y_test = y_test.astype(np.int32)
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_mldata

sd = StandardScaler()
x_sd = sd.fit_transform(x)
x_test_sd = sd.fit_transform(X_test)
n_inputs = 28*28
n_hidden1 = 300
n_hidden2 = 100
n_outputs=10
X = tf.placeholder(tf.float32,shape=(None,n_inputs),name='X')
Y = tf.placeholder(tf.int64,shape=(None),name='Y')
is_training = tf.placeholder(tf.bool,shape=(),name="is_training")

def neuron_layer(X,n_neurons,name,activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2/np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs,n_neurons),stddev=stddev)
        w = tf.Variable(init,name='weights')
        b = tf.Variable(tf.zeros([n_neurons]),name='biases')
        z = tf.matmul(X,w)+b
        if activation=='relu':
            return tf.nn.relu(z)
        else:
            return z
# with tf.name_scope("dnn"):
#     hidden1 = neuron_layer(X,n_hidden1,"hidden1",activation='relu')
#     hidden2 = neuron_layer(hidden1,n_hidden2,"hidden2",activation='relu')
#     logits = neuron_layer(hidden2,n_outputs,"outputs")

from tensorflow.contrib.layers import fully_connected
he_init = tf.contrib.layers.variance_scaling_initializer()
def leaky_relu(z,name=None):
    return tf.maximum(0.01 * z, z, name=name)

from tensorflow.contrib.layers import batch_norm
bn_params = {
    "is_training":is_training,
    "decay":0.99,
    "updates_collections":None,
    "scale":True
}
with tf.name_scope("dnn"):
    with tf.contrib.framework.arg_scope([fully_connected],normalizer_fn=batch_norm,normalizer_params=bn_params):
        hidden1 = fully_connected(X,n_hidden1,scope='hidden1',weights_initializer=he_init,activation_fn=tf.nn.elu)
        hidden2 = fully_connected(hidden1,n_hidden2,scope='hidden2',weights_initializer=he_init,activation_fn=tf.nn.elu)
        logits = fully_connected(hidden2,n_outputs,activation_fn=None,scope='outputs')

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y,logits=logits)
    loss=tf.reduce_mean(xentropy,name='loss')

learn_rate=0.01
threshold = 1.0
with tf.name_scope("train"):
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="hidden[2]|outputs")
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate)
    training_op = optimizer.minimize(loss, var_list=train_vars)
    # training_op = optimizer.minimize(loss)
    # grads_and_vars = optimizer.compute_gradients(loss)
    # capped_gvs = [(tf.clip_by_value(grad,-threshold,threshold),var) for grad,var in grads_and_vars]
    # training_op = optimizer.apply_gradients(capped_gvs)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits,Y,1)
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()
reuse_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='hidden[1]')
reuse_vars_dict = dict([(var.name,var) for var in reuse_vars])
original_saver = tf.train.Saver(reuse_vars_dict)

from tensorflow.examples.tutorials.mnist import input_data

n_epochs = 400
batch_size = 50
data = []
data_2 = []
with tf.Session() as sess:
    init.run()
    original_saver.restore(sess,"./my_model_final.ckpt")
    hidden2_outputs = sess.run(hidden2, feed_dict={X: x})

    for epoch in range(n_epochs):
        mini = np.array_split(range(y.shape[0]),batch_size)
        for idx in mini:
            X_batch,y_batch = x_sd[idx],y[idx]
            sess.run(training_op, feed_dict={X: X_batch, Y: y_batch,is_training:True,hidden2:hidden2_outputs})
        acc_train = accuracy.eval(feed_dict={X: X_batch, Y: y_batch,is_training:False,hidden2:hidden2_outputs})
        acc_test = accuracy.eval(feed_dict={X: x_test_sd,Y: y_test,is_training:False})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
        data.append(acc_train)
        data_2.append(acc_test)
    save_path = saver.save(sess, "./my_model_final.ckpt")

import matplotlib.pyplot as plt
plt.plot(range(n_epochs),data,color='blue',marker='o')
plt.plot(range(n_epochs),data_2,color='red',marker='s')
plt.show()