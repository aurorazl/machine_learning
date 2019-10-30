import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

n_steps = 20
n_inputs = 1
n_neurons = 100
n_outputs = 1
n_layers = 3
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])
cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
# cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu),output_size=n_outputs)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

stacked_rnn_outputs = tf.reshape(outputs, [-1, n_neurons])
stacked_outputs = fully_connected(stacked_rnn_outputs, n_outputs,activation_fn=None)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])

learning_rate = 0.001
loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
init = tf.global_variables_initializer()
n_iterations = 10000
batch_size = 50
t_min, t_max = 0, 30
resolution = 0.1
import numpy as np
def time_series(t):
    return t * np.sin(t) / 3 + 2 * np.sin(t*5)
def next_batch(batch_size, n_steps):
    t0 = np.random.rand(batch_size, 1) * (t_max - t_min - n_steps * resolution)
    Ts = t0 + np.arange(0., n_steps + 1) * resolution
    ys = time_series(Ts)
    return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1)
t = np.linspace(t_min, t_max, int((t_max - t_min) / resolution))

n_steps = 20
t_instance = np.linspace(12.2, 12.2 + resolution * (n_steps + 1), n_steps + 1)
import matplotlib.pyplot as plt
# plt.figure(figsize=(11,4))
# plt.subplot(121)
# plt.title("A time series (generated)", fontsize=14)
# plt.plot(t, time_series(t), label=r"$t . \sin(t) / 3 + 2 . \sin(5t)$")
# plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "b-", linewidth=3, label="A training instance")
# plt.legend(loc="lower left", fontsize=14)
# plt.axis([0, 30, -17, 13])
# plt.xlabel("Time")
# plt.ylabel("Value")
#
# plt.subplot(122)
# plt.title("A training instance", fontsize=14)
# plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=10, label="instance")
# plt.plot(t_instance[1:], time_series(t_instance[1:]), "w*", markersize=10, label="target")
# plt.legend(loc="upper left")
# plt.xlabel("Time")


# save_fig("time_series_plot")
# plt.show()

saver = tf.train.Saver()
# with tf.Session() as sess:
#     init.run()
#     for iteration in range(n_iterations):
#         X_batch, y_batch = next_batch(batch_size, n_steps)
#         sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
#         if iteration % 100 == 0:
#             mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
#             print(iteration, "\tMSE:", mse)
#
#     saver.save(sess, "./my_time_series_model")  # not shown in the book

# with tf.Session() as sess:                          # not shown in the book
#     saver.restore(sess, "./my_time_series_model")   # not shown
#
#     X_new = time_series(np.array(t_instance[:-1].reshape(-1, n_steps, n_inputs)))
#     y_pred = sess.run(outputs, feed_dict={X: X_new})
#
# plt.title("Testing the model", fontsize=14)
# plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=10, label="instance")
# plt.plot(t_instance[1:], time_series(t_instance[1:]), "w*", markersize=10, label="target")
# plt.plot(t_instance[1:], y_pred[0,:,0], "r.", markersize=10, label="prediction")
# plt.legend(loc="upper left")
# plt.xlabel("Time")
# plt.show()

# with tf.Session() as sess:                        # not shown in the book
#     saver.restore(sess, "./my_time_series_model") # not shown
#     sequence = [0.] * n_steps
#     for iteration in range(300):
#         X_batch = np.array(sequence[-n_steps:]).reshape(1, n_steps, 1)
#         y_pred = sess.run(outputs, feed_dict={X: X_batch})
#         sequence.append(y_pred[0, -1, 0])
#
# plt.figure(figsize=(8,4))
# plt.plot(np.arange(len(sequence)), sequence, "b-")
# plt.plot(t[:n_steps], sequence[:n_steps], "b-", linewidth=3)
# plt.xlabel("Time")
# plt.ylabel("Value")
# plt.show()

with tf.Session() as sess:
    saver.restore(sess, "./my_time_series_model")

    sequence1 = [0. for i in range(n_steps)]
    for iteration in range(len(t) - n_steps):
        X_batch = np.array(sequence1[-n_steps:]).reshape(1, n_steps, 1)
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        sequence1.append(y_pred[0, -1, 0])

    sequence2 = [time_series(i * resolution + t_min + (t_max-t_min/3)) for i in range(n_steps)]
    for iteration in range(len(t) - n_steps):
        X_batch = np.array(sequence2[-n_steps:]).reshape(1, n_steps, 1)
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        sequence2.append(y_pred[0, -1, 0])

plt.figure(figsize=(11,4))
plt.subplot(121)
plt.plot(t, sequence1, "b-")
plt.plot(t[:n_steps], sequence1[:n_steps], "b-", linewidth=3)
plt.xlabel("Time")
plt.ylabel("Value")

plt.subplot(122)
plt.plot(t, sequence2, "b-")
plt.plot(t[:n_steps], sequence2[:n_steps], "b-", linewidth=3)
plt.xlabel("Time")
plt.show()