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
tf.app.flags.DEFINE_string("ps_hosts", "localhost:2223","Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "localhost:2224,localhost:2225","Comma-separated list of hostname:port pairs")
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
per_size = 600

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

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(grads,0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    # server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    worker_count = len(worker_hosts)

    issync = FLAGS.issync

    with tf.device(tf.train.replica_device_setter(cluster=cluster)):
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
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # q = tf.RandomShuffleQueue(capacity=per_size*worker_count, min_after_dequeue=0, dtypes=[tf.float32,tf.int32], shapes=[(784),[]], name="q15",shared_name="shared_q15")
        # feature_instances = tf.placeholder(tf.float32, shape=(None,784))
        # target_instances = tf.placeholder(tf.int32, shape=(None))
        # enqueue_many = q.enqueue_many((feature_instances,target_instances))
        # close_q = q.close()
        # run_options = tf.RunOptions()
        # run_options.timeout_in_ms = 2000

    with tf.contrib.framework.arg_scope([fully_connected], normalizer_fn=batch_norm, normalizer_params=bn_params):
        # X,Y = q.dequeue_up_to(per_size)
        x_drop = dropout(X, keep_prob, is_training=is_training)
        hidden1 = fully_connected(x_drop, n_hidden1, scope='hidden1', weights_initializer=he_init,
                                  activation_fn=tf.nn.elu)
        hidden1_drop = dropout(hidden1, keep_prob, is_training=is_training)
        hidden2 = fully_connected(hidden1_drop, n_hidden2, scope='hidden2', weights_initializer=he_init,
                                  activation_fn=tf.nn.elu)
        hidden2_drop = dropout(hidden2, keep_prob, is_training=is_training)
        logits = fully_connected(hidden2_drop, n_outputs, activation_fn=None, scope='outputs')

    tower_grads = []
    for i in range(worker_count):
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % i,cluster=cluster)):
            with tf.name_scope('test_%d' % (i)):
                # 问题点：如何分开不同样本给不同的副本
                # X = tf.split(X,worker_count,0)[i]
                # X = X[:tf.shape(X)//worker_count*(i+1)]
                X = tf.slice(X,tf.shape(X)//worker_count*(i),tf.shape(X)//worker_count)
                xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=logits)
                loss = tf.reduce_mean(xentropy, name="loss")
                grads_and_vars = optimizer.compute_gradients(loss)
                tower_grads.append(grads_and_vars)
    # 异步将下面两个操作去掉，直接使用training_op = optimizer.minimize(loss)，然后每个副本一个线程运行training_op，计算运行在worker上，多线程没问题
    # 各自的for循环迭代不同的样本集来run
    capped_gvs = average_gradients(tower_grads)
    training_op = optimizer.apply_gradients(capped_gvs)

    saver = tf.train.Saver()
    init = tf.initialize_all_variables()

    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(logits, Y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    with tf.Session("grpc://localhost:2221") as sess:
        init.run()
        for epoch in range(n_epochs):
            mini = np.array_split(range(y_train.shape[0]), batch_size*worker_count)
            for idx in mini:
                # sess.run(enqueue_many, feed_dict={feature_instances:X_train_std[idx],target_instances:y_train[idx]})
                # try:
                #     sess.run(training_op,feed_dict={is_training:True},options=run_options)
                # except tf.errors.DeadlineExceededError:
                #     pass
                X_batch, y_batch = X_train_std[idx], y_train[idx]
                sess.run(training_op, feed_dict={X: X_batch, Y: y_batch, is_training: True})
            acc_train = accuracy.eval(feed_dict={X: X_batch, Y: y_batch, is_training: False},session=sess)
            # acc_train = accuracy.eval(feed_dict={is_training: False},session=sess)
            acc_test = accuracy.eval(feed_dict={X: X_test_sd, Y: y_test, is_training: False},session=sess)
            print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

if __name__ == "__main__":
    tf.app.run()