import os
import os.path
import cv2
import glob
import imutils

import cv2
import pickle
import os.path
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense

def resize_to_fit(image, width, height):
    (h, w) = image.shape[:2]
    if w > h:
        image = imutils.resize(image, width=width)
    else:
        image = imutils.resize(image, height=height)
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW,cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))
    return image

LETTER_IMAGES_FOLDER = "generated_captcha_images"

def load_image():
    data = []
    labels = []
    for image_file in paths.list_images(LETTER_IMAGES_FOLDER):
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = resize_to_fit(image,72,24)
        image = np.expand_dims(image, axis=2)
        label = image_file.split(os.path.sep)[-1].split(".")[0]
        data.append(image)
        labels.append([i for i in label])
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    return data,labels

X,y = load_image()
lb = LabelBinarizer()
y_len = y.shape[0]
y=y.reshape(-1)
y = lb.fit_transform(y)
y_wid = y.shape[1]
y = y.reshape(y_len,-1)
(X_train, X_test,y_train,y_test) = train_test_split(X, y, test_size=0.25, random_state=0)

# from sklearn.preprocessing import MinMaxScaler
# sd = MinMaxScaler()
# X_train= sd.fit_transform(X_train)
# X_test = sd.fit_transform(X_test)

import tensorflow as tf
height = 24
width = 72
channels = 1
n_inputs = height * width

conv1_fmaps = 32
conv1_ksize = [1,3]
conv1_stride = 1
conv1_pad = "SAME"

conv2_fmaps = 64
conv2_ksize = [1,3]
conv2_stride = 1
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

conv1 = tf.layers.conv2d(X, filters=conv1_fmaps, kernel_size=conv1_ksize,
                         strides=conv1_stride, padding=conv1_pad,
                         activation=tf.nn.relu, name="conv1")
conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,
                         strides=conv2_stride, padding=conv2_pad,
                         activation=tf.nn.relu, name="conv2")
with tf.name_scope("pool3"):
    pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 6, 1], strides=[1, 2, 6, 1], padding="VALID")
    conv3 = tf.layers.conv2d(pool3, filters=96, kernel_size=3,
                             strides=1, padding=conv1_pad,
                             activation=tf.nn.relu, name="conv3")
    conv4 = tf.layers.conv2d(conv3, filters=48, kernel_size=3,
                             strides=1, padding=conv1_pad,
                             activation=tf.nn.relu, name="conv4")
    pool3_flat = tf.reshape(conv4, shape=[-1, 48 * 12 * 12])
    pool3_flat_drop = tf.layers.dropout(pool3_flat, conv2_dropout_rate, training=training)

with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(pool3_flat_drop, n_fc1, activation=tf.nn.relu, name="fc1")
    fc1_drop = tf.layers.dropout(fc1, fc1_dropout_rate, training=training)

with tf.name_scope("output"):
    logits = tf.layers.dense(fc1_drop, n_outputs, name="output")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")

with tf.name_scope("train"):
    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    predict = tf.reshape(logits, [-1, 4, y_wid])
    real = tf.reshape(y, [-1, 4, y_wid])
    correct= tf.reduce_all(tf.equal(tf.argmax(predict,2), tf.argmax(real,2)),axis=1)
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

# with tf.Session() as sess:
#     init.run()
#     for epoch in range(n_epochs):
#         for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
#             iteration += 1
#             sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})
#             if iteration % check_interval == 0:
#                 loss_val = loss.eval(feed_dict={X: X_test, y: y_test})
#                 if loss_val < best_loss_val:
#                     best_loss_val = loss_val
#                     checks_since_last_progress = 0
#                     best_model_params = get_model_params()
#                 else:
#                     checks_since_last_progress += 1
#         acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
#         acc_val = accuracy.eval(feed_dict={X: X_test, y: y_test})
#         print("Epoch {}, last batch accuracy: {:.4f}%, valid. accuracy: {:.4f}%, valid. best loss: {:.6f}".format(
#                   epoch, acc_batch * 100, acc_val * 100, best_loss_val))
#         if checks_since_last_progress > max_checks_without_progress:
#             print("Early stopping!")
#             break
#
#     if best_model_params:
#         restore_model_params(best_model_params)
#     acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
#     print("Final accuracy on test set:", acc_test)
    # save_path = saver.save(sess, "./captchas_no_split_model")

with tf.Session() as sess:
    saver.restore(sess,"./captchas_no_split_model")
    data,labels=[],[]
    from captcha.image import ImageCaptcha
    obj=ImageCaptcha(width=160, height=60)
    # img = obj.generate_image("BBBB", noise_dot=False, color_draw=(255, 0, 0)).convert('L')
    from check_code import create_validate_code
    img, captcha_str = create_validate_code(draw_lines=False, draw_points=False)
    img.save("test.png")
    print(captcha_str)
    # image = np.array(img.getdata()).reshape((24,72,1))
    # from PIL import Image
    # im = Image.open("test.png")
    # im=im.resize([72,24])
    # im.save("test.png")
    # image = cv2.imread("test.png")
    # image = cv2.resize(image,(72,24))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = resize_to_fit(image, 72, 24)
    from skimage import io, transform
    import matplotlib.pyplot as plt
    img = io.imread("test.png")
    img = transform.resize(img, (24, 72))
    plt.imshow(img, plt.cm.gray)
    plt.show()
    # image = np.expand_dims(img, axis=2)
    image = img[:,:,[0]]
    data.append(image)
    test_data = np.array(data, dtype="float") / 255.0
    y_predict = Y_proba.eval(feed_dict={X: test_data})
    print(lb.inverse_transform(y_predict.reshape(4, -1)))

    # y_predict=Y_proba.eval(feed_dict={X: X_test[[5],:]})
    # print(lb.inverse_transform(y_predict.reshape(4,-1)))
    # print(lb.inverse_transform(y_test[[5]].reshape(4,-1)))