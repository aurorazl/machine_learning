from captcha.audio import AudioCaptcha
from captcha.image import ImageCaptcha

# audio = AudioCaptcha()
# image = ImageCaptcha()
# image.write('1234', 'out.png')

import os
import os.path
import cv2
import glob
import imutils

CAPTCHA_IMAGE_FOLDER = "generated_captcha_images"
OUTPUT_FOLDER = "extracted_letter_images"

def extract_single_letter_from_captchas():
    captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))
    counts = {}
    for (i, captcha_image_file) in enumerate(captcha_image_files):
        print("[INFO] processing image {}/{}".format(i + 1, len(captcha_image_files)))
        filename = os.path.basename(captcha_image_file)
        captcha_correct_text = os.path.splitext(filename)[0]
        image = cv2.imread(captcha_image_file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0]
        letter_image_regions = []
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if w / h > 1.25:
                half_width = int(w / 2)
                letter_image_regions.append((x, y, half_width, h))
                letter_image_regions.append((x + half_width, y, half_width, h))
            else:
                letter_image_regions.append((x, y, w, h))
        if len(letter_image_regions) != 4:
            continue
        letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
        for letter_bounding_box, letter_text in zip(letter_image_regions, captcha_correct_text):
            x, y, w, h = letter_bounding_box
            letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]
            save_path = os.path.join(OUTPUT_FOLDER, letter_text)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            count = counts.get(letter_text, 1)
            p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
            cv2.imwrite(p, letter_image)
            counts[letter_text] = count + 1

# extract_single_letter_from_captchas()

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

LETTER_IMAGES_FOLDER = "extracted_letter_images"

def load_image():
    data = []
    labels = []
    for image_file in paths.list_images(LETTER_IMAGES_FOLDER):
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = resize_to_fit(image,20,20)
        image = np.expand_dims(image, axis=2)
        label = image_file.split(os.path.sep)[-2]
        data.append(image)
        labels.append(label)
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    return data,labels

X,y = load_image()
lb = LabelBinarizer()
y = lb.fit_transform(y)
(X_train, X_test,y_train,y_test) = train_test_split(X, y, test_size=0.25, random_state=0)

# from sklearn.preprocessing import MinMaxScaler
# sd = MinMaxScaler()
# X_train= sd.fit_transform(X_train)
# X_test = sd.fit_transform(X_test)

import tensorflow as tf
height = 20
width = 20
channels = 1
n_inputs = height * width

conv1_fmaps = 32
conv1_ksize = 3
conv1_stride = 1
conv1_pad = "SAME"

conv2_fmaps = 64
conv2_ksize = 3
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
    pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 10 * 10])
    pool3_flat_drop = tf.layers.dropout(pool3_flat, conv2_dropout_rate, training=training)

with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(pool3_flat_drop, n_fc1, activation=tf.nn.relu, name="fc1")
    fc1_drop = tf.layers.dropout(fc1, fc1_dropout_rate, training=training)

with tf.name_scope("output"):
    logits = tf.layers.dense(fc1_drop, n_outputs, name="output")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")

with tf.name_scope("train"):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits,tf.argmax(y,axis=1), 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


n_epochs = 1000
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
#     save_path = saver.save(sess, "./captchas_model")

with tf.Session() as sess:
    saver.restore(sess,"./captchas_model")
    y_predict=Y_proba.eval(feed_dict={X: X_test[[5],:]})
    print(lb.inverse_transform(y_predict))
    print(lb.inverse_transform(y_test[[5]]))