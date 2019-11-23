import tensorflow as tf

n_steps = 2
n_inputs = 3
n_neurons = 5
X0 = tf.placeholder(tf.float32, [None, n_inputs])
X1 = tf.placeholder(tf.float32, [None, n_inputs])
# Wx = tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons],dtype=tf.float32))
# Wy = tf.Variable(tf.random_normal(shape=[n_neurons,n_neurons],dtype=tf.float32))
# b = tf.Variable(tf.zeros([1, n_neurons], dtype=tf.float32))
# Y0 = tf.tanh(tf.matmul(X0, Wx) + b)
# Y1 = tf.tanh(tf.matmul(Y0, Wy) + tf.matmul(X1, Wx) + b)
# basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
# output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, [X0, X1], dtype=tf.float32)
# Y0, Y1 = output_seqs

# X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
# X_seqs = tf.unstack(tf.transpose(X, perm=[1, 0, 2]))
# basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
# output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, X_seqs, dtype=tf.float32)
# outputs = tf.transpose(tf.stack(output_seqs), perm=[1, 0, 2])

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
seq_length = tf.placeholder(tf.int32, [None])
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32,sequence_length=seq_length)

init = tf.global_variables_initializer()

import numpy as np
# # Mini-batch: instance 0,instance 1,instance 2,instance 3
# X0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]]) # t = 0
# X1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]]) # t = 1
# with tf.Session() as sess:
#     init.run()
#     Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0: X0_batch, X1: X1_batch})
#     print(Y0_val)  # output at t = 0
#     print(Y1_val)  # output at t = 1

X_batch = np.array([
# t = 0 t = 1
[[0, 1, 2], [9, 8, 7]], # instance 0
[[3, 4, 5], [0, 0, 0]], # instance 1
[[6, 7, 8], [6, 5, 4]], # instance 2
[[9, 0, 1], [3, 2, 1]], # instance 3
])
seq_length_batch = np.array([2, 1, 2, 2])
with tf.Session() as sess:
    init.run()
    # outputs_val = outputs.eval(feed_dict={X: X_batch,seq_length:seq_length_batch})
    outputs_val, states_val = sess.run(
        [outputs, states], feed_dict=
        {X: X_batch, seq_length: seq_length_batch})
    print(outputs_val)
    print(states_val)