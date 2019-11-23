import os
import glob
from skimage import io,transform
import numpy as np

def read_img(path):
    cate = [path + '/' + x for x in os.listdir(path) if os.path.isdir(path + '/' + x)]
    imgs = []
    labels = []
    for idx, folder in enumerate(cate):
        # if len(list(glob.glob(folder + '/*.jpg')))<2:
        #     continue
        if idx>20:
            break
        for im in glob.glob(folder + '/*.jpg'):
            # print('reading the images:%s' % (im))
            try:
                img = io.imread(im)
            except Exception:
                continue
            img = transform.resize(img, (250, 250, 3))
            # import matplotlib.pyplot as plt
            # plt.imshow(img, plt.cm.gray)
            # plt.show()
            imgs.append(img)
            labels.append(idx)
    return np.array(imgs, np.float32), np.array(labels, np.int32)

path=r"D:\BaiduNetdiskDownload\casia-maxpy-clean\CASIA-maxpy-clean"
X, y = read_img(path)
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
y = lb.fit_transform(y)


num_example=X.shape[0]
# index= np.random.permutation(10)
# data=data[index]
# label=label[index]
from sklearn.model_selection import train_test_split
(X_train, X_test,y_train,y_test) = train_test_split(X, y, test_size=0.25, random_state=0,shuffle=True)

import tensorflow as tf
height = 250
width = 250
channels = 3
n_inputs = height * width

conv1_fmaps = 96
conv1_ksize = [5,5]
conv1_stride = 4
conv1_pad = "SAME"

conv2_fmaps = 128
conv2_ksize = [3,3]
conv2_stride = 2
conv2_pad = "SAME"
conv2_dropout_rate = 0.25

pool3_fmaps = conv2_fmaps

n_fc1 = 128
fc1_dropout_rate = 0.5
n_outputs = y.shape[1]

with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, height, width, channels], name="X")
    y = tf.placeholder(tf.float32, shape=[None,y.shape[1]], name="y")
    training = tf.placeholder_with_default(False, shape=[], name='training')
    phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

# conv1 = tf.layers.conv2d(X, filters=conv1_fmaps, kernel_size=conv1_ksize,
#                          strides=conv1_stride, padding=conv1_pad,
#                          activation=tf.nn.relu, name="conv1")
# pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
# conv2 = tf.layers.conv2d(pool1, filters=conv2_fmaps, kernel_size=conv2_ksize,
#                          strides=conv2_stride, padding=conv2_pad,
#                          activation=tf.nn.relu, name="conv2")
# with tf.name_scope("pool3"):
#     pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
#     conv3 = tf.layers.conv2d(pool2, filters=96, kernel_size=3,
#                              strides=1, padding=conv1_pad,
#                              activation=tf.nn.relu, name="conv3")
#     pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
#     conv4 = tf.layers.conv2d(conv3, filters=256, kernel_size=3,
#                              strides=1, padding=conv1_pad,
#                              activation=tf.nn.relu, name="conv4")
#     conv5 = tf.layers.conv2d(conv4, filters=256, kernel_size=3,
#                              strides=1, padding=conv1_pad,
#                              activation=tf.nn.relu, name="conv5")
#     conv6 = tf.layers.conv2d(conv5, filters=128, kernel_size=3,
#                              strides=1, padding=conv1_pad,
#                              activation=tf.nn.relu, name="conv6")
#     # pool4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
#     pool3_flat = tf.reshape(conv6, shape=[-1, 128 * 10 * 10])
#     pool3_flat_drop = tf.layers.dropout(pool3_flat, conv2_dropout_rate, training=training)
#
# with tf.name_scope("fc1"):
#     fc1 = tf.layers.dense(pool3_flat_drop, n_fc1, activation=tf.nn.relu, name="fc1")
#     fc1_drop = tf.layers.dropout(fc1, fc1_dropout_rate, training=training)

# import models
# import importlib
# network = importlib.import_module("models.inception_resnet_v1")
# keep_probability=0.8
# embedding_size=128
# weight_decay=5e-5
# prelogits, _ = network.inference(X, keep_probability,
#             phase_train=phase_train_placeholder, bottleneck_layer_size=embedding_size,
#             weight_decay=weight_decay)
# with tf.name_scope("output"):
#     logits = tf.layers.dense(prelogits, n_outputs, name="output")
#     Y_proba = tf.nn.softmax(logits, name="Y_proba")

from tensorflow.contrib.slim.nets import inception
import tensorflow.contrib.slim as slim
with slim.arg_scope(inception.inception_v3_arg_scope()):
    logits, end_points = inception.inception_v3(X, num_classes=n_outputs, is_training=training)
inception_saver = tf.train.Saver()
prelogits = tf.squeeze(end_points["PreLogits"], axis=[1, 2])
with tf.name_scope("new_output_layer"):
    flower_logits = tf.layers.dense(prelogits, n_outputs, name="output")
    Y_proba = tf.nn.softmax(flower_logits, name="Y_proba")

with tf.name_scope("train"):
    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=flower_logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    output_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="output")
    training_op = optimizer.minimize(loss,var_list=output_vars)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, tf.argmax(y,1), 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

n_epochs = 10000
batch_size = 50
iteration = 0

best_loss_val = np.infty
check_interval = 500
checks_since_last_progress = 0
max_checks_without_progress = 20
best_model_params = None

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch
def get_model_params():
    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}
def restore_model_params(model_params):
    gvar_names = list(model_params.keys())
    assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + "/Assign")
                  for gvar_name in gvar_names}
    init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
    feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
    tf.get_default_session().run(assign_ops, feed_dict=feed_dict)

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            iteration += 1
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True,phase_train_placeholder:True})
            if iteration % check_interval == 0:
                loss_val = loss.eval(feed_dict={X: X_test, y: y_test,phase_train_placeholder:False})
                if loss_val < best_loss_val:
                    best_loss_val = loss_val
                    checks_since_last_progress = 0
                    best_model_params = get_model_params()
                else:
                    checks_since_last_progress += 1
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch,phase_train_placeholder:False})
        acc_val = accuracy.eval(feed_dict={X: X_test, y: y_test,phase_train_placeholder:False})
        print("Epoch {}, last batch accuracy: {:.4f}%, valid. accuracy: {:.4f}%, valid. best loss: {:.6f}".format(
                  epoch, acc_batch * 100, acc_val * 100, best_loss_val))
        if checks_since_last_progress > max_checks_without_progress:
            print("Early stopping!")
            break

    if best_model_params:
        restore_model_params(best_model_params)
    acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
    print("Final accuracy on test set:", acc_test)
    save_path = saver.save(sess, "./face_model")


