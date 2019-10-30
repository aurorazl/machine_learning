# coding=utf-8

# python 12-3图间拷贝.py --job_name=ps --task_index=0 --issync=1
# python 12-3图间拷贝.py --job_name=worker --task_index=0 --issync=1
# python 12-3图间拷贝.py --job_name=worker --task_index=1 --issync=1

import time
import numpy as np
import os
import struct
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import dropout
from sklearn.preprocessing import StandardScaler

# Define parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('learning_rate', 0.00003, 'Initial learning rate.')
tf.app.flags.DEFINE_integer('steps_to_validate', 1000,'Steps to validate and print loss')

# For distributed
tf.app.flags.DEFINE_string("ps_hosts", "localhost:2213","Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "localhost:2214,localhost:2215","Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "worker", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("issync", 0, "是否采用分布式的同步模式，1表示同步模式，0表示异步模式")

# Hyperparameters
learning_rate = FLAGS.learning_rate
steps_to_validate = FLAGS.steps_to_validate
n_inputs = 28*28
n_outputs=10
n_hidden1 = 300
n_hidden2 = 100
keep_prob = 0.5
n_epochs = 400
batch_size = 50

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

def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    worker_count = len(worker_hosts)

    issync = FLAGS.issync
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index,cluster=cluster)):
            global_step = tf.Variable(0, name='global_step', trainable=False)

            X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
            Y = tf.placeholder(tf.int64, shape=(None), name='Y')
            is_training = tf.placeholder(tf.bool, shape=(), name="is_training")
            he_init = tf.contrib.layers.variance_scaling_initializer()
            bn_params = {
                "is_training": is_training,
                "decay": 0.99,
                "updates_collections": None,
                "scale": True
            }

            with tf.name_scope("dnn"):
                with tf.contrib.framework.arg_scope([fully_connected], normalizer_fn=batch_norm,normalizer_params=bn_params):
                    x_drop = dropout(X, keep_prob, is_training=is_training)
                    hidden1 = fully_connected(x_drop, n_hidden1, scope='hidden1', weights_initializer=he_init,
                                              activation_fn=tf.nn.elu)
                    hidden1_drop = dropout(hidden1, keep_prob, is_training=is_training)
                    hidden2 = fully_connected(hidden1_drop, n_hidden2, scope='hidden2', weights_initializer=he_init,
                                              activation_fn=tf.nn.elu)
                    hidden2_drop = dropout(hidden2, keep_prob, is_training=is_training)
                    logits = fully_connected(hidden2_drop, n_outputs, activation_fn=None, scope='outputs')
            with tf.name_scope("loss"):
                xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=logits)
                loss = tf.reduce_mean(xentropy, name="loss")
            with tf.name_scope("train"):
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

            if issync == 1:
                optimizer = tf.train.SyncReplicasOptimizer(optimizer,replicas_to_aggregate=worker_count,total_num_replicas=worker_count,use_locking=True)
                sync_replicas_hook = optimizer.make_session_run_hook(FLAGS.task_index == 0)

            # 更新梯度
            train_op = optimizer.minimize(loss, global_step=global_step)

            # hooks = [tf.train.StopAtStepHook(last_step=10000)]
            hooks = []
            if issync == 1:
                hooks.append(sync_replicas_hook)

            with tf.name_scope('eval'):
                correct = tf.nn.in_top_k(logits, Y, 1)
                accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

            with tf.train.MonitoredTrainingSession(master=server.target, is_chief=(FLAGS.task_index == 0),checkpoint_dir="./train_logs", hooks=hooks) as mon_sess:
                # while not mon_sess.should_stop():
                    for epoch in range(n_epochs):
                        mini = np.array_split(np.array_split(np.arange(y_train.shape[0]),worker_count)[FLAGS.task_index], batch_size)
                        for idx in mini:
                            X_batch, y_batch = X_train_std[idx], y_train[idx]
                            mon_sess.run(train_op, feed_dict={X: X_batch, Y: y_batch, is_training: True})
                        acc_train = accuracy.eval(feed_dict={X: X_batch, Y: y_batch, is_training: False},session=mon_sess)
                        acc_test = accuracy.eval(feed_dict={X: X_test_sd, Y: y_test, is_training: False},session=mon_sess)
                        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

if __name__ == "__main__":
    tf.app.run()