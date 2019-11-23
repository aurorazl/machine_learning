import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
n_steps = 28
n_inputs = 28
n_neurons = 150
n_outputs = 10
learning_rate = 0.001
n_layers = 3
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])

# basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
# outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

# n_neurons = 100
n_layers = 3

# lstm_cells = [tf.nn.rnn_cell.BasicLSTMCell(num_units=n_neurons) for layer in range(n_layers)]
# lstm_cells = [tf.nn.rnn_cell.LSTMCell(num_units=n_neurons,use_peepholes=True) for layer in range(n_layers)]
lstm_cells = [tf.nn.rnn_cell.GRUCell(num_units=n_neurons) for layer in range(n_layers)]
multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

# top_layer_h_state = states[-1][1] # LSTMCell
top_layer_h_state = states[-1]  # GRUCell只有一个状态向量
# states_concat = tf.concat(axis=1, values=states)
# logits = fully_connected(states_concat, n_outputs, activation_fn=None)
logits = tf.layers.dense(top_layer_h_state, n_outputs, name="softmax")
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

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
sd = StandardScaler()
X_train_std = sd.fit_transform(X_train)
X_test_sd = sd.fit_transform(X_test)

n_epochs = 10
batch_size = 150
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        mini = np.array_split(range(y_train.shape[0]), batch_size)
        for idx in mini:
            X_batch, y_batch = X_train_std[idx], y_train[idx]
            X_batch = X_batch.reshape((-1, n_steps, n_inputs))
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test_sd.reshape((-1, n_steps, n_inputs)), y: y_test})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)