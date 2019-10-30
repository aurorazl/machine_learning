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
from tensorflow.contrib.layers import dropout
keep_prob = 0.5
def max_norm_regularizer(threshold,axes=1,name='max_norm',collection='max_norm'):
    def max_norm(weights):
        clipped_weights = tf.clip_by_norm(weights,clip_norm=threshold,axes=axes)
        clip_weights = tf.assign(weights,clipped_weights,name=name)
        tf.add_to_collection(collection,clip_weights)
        return None
    return max_norm
max_norm_reg = max_norm_regularizer(threshold=1.0)
with tf.name_scope("dnn"):
    hidden1 = fully_connected(X,n_hidden1,scope='hidden1',weights_initializer=he_init,activation_fn=tf.nn.elu,weights_regularizer=max_norm_reg)
    hidden2 = fully_connected(hidden1,n_hidden2,scope='hidden2',weights_initializer=he_init,activation_fn=tf.nn.elu,weights_regularizer=max_norm_reg)
    logits = fully_connected(hidden2,n_outputs,activation_fn=None,scope='outputs')

# with tf.name_scope("dnn"):
#     # with tf.contrib.framework.arg_scope([fully_connected],normalizer_fn=batch_norm,normalizer_params=bn_params,weights_regularizer=tf.contrib.layers.l1_regularizer(scale=0.01)):
#         x_drop = dropout(X, keep_prob, is_training=is_training)
#         hidden1 = fully_connected(x_drop,n_hidden1,scope='hidden1',weights_initializer=he_init,activation_fn=tf.nn.elu)
#         hidden1_drop = dropout(hidden1,keep_prob, is_training=is_training)
#         hidden2 = fully_connected(hidden1_drop,n_hidden2,scope='hidden2',weights_initializer=he_init,activation_fn=tf.nn.elu)
#         hidden2_drop = dropout(hidden2, keep_prob, is_training=is_training)
#         logits = fully_connected(hidden2_drop,n_outputs,activation_fn=None,scope='outputs')

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y,logits=logits)
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    base_loss = tf.reduce_mean(xentropy, name="avg_xentropy")
    loss = tf.add_n([base_loss] + reg_losses, name="loss")

# learning_rate=0.01
threshold = 1.0
initial_learning_rate = 0.1
decay_steps = 10000
decay_rate=1/10
global_step = tf.Variable(0,trainable=False)
learning_rate = tf.train.exponential_decay(initial_learning_rate,global_step,decay_steps,decay_rate)
with tf.name_scope("train"):
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9,use_nesterov=True)
    # optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,momentum=0.9, decay=0.9, epsilon=1e-10)
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss,global_step=global_step)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits,Y,1)
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

from tensorflow.examples.tutorials.mnist import input_data

n_epochs = 400
batch_size = 50
data = []
data_2 = []

clip_all_weights = tf.get_collection("max_norm")
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        mini = np.array_split(range(y.shape[0]),batch_size)
        for idx in mini:
            X_batch,y_batch = x_sd[idx],y[idx]
            sess.run(training_op, feed_dict={X: X_batch, Y: y_batch,is_training:True})
            # clipped_weights.eval()
            sess.run(clip_all_weights)
        acc_train = accuracy.eval(feed_dict={X: X_batch, Y: y_batch,is_training:False})
        acc_test = accuracy.eval(feed_dict={X: x_test_sd,Y: y_test,is_training:False})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
        data.append(acc_train)
        data_2.append(acc_test)
    save_path = saver.save(sess, "./my_model_final.ckpt")
import matplotlib.pyplot as plt
plt.plot(range(n_epochs),data,color='blue',marker='o')
plt.plot(range(n_epochs),data_2,color='red',marker='s')
plt.show()
# with tf.Session() as sess:
#     saver.restore(sess, "./my_model_final.ckpt")
#     Z = logits.eval(feed_dict={X: x_test_sd})
#     y_pred = np.argmax(Z, axis=1)
#     from sklearn.metrics import accuracy_score
#     print(accuracy_score(y_test,y_pred))