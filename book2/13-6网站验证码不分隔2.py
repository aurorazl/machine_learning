from captcha.image import ImageCaptcha
from PIL import Image
import numpy as np
import random
import string
from PIL import Image,ImageDraw,ImageFont,ImageFilter
from check_code import create_validate_code

class generateCaptcha():
    def __init__(self,
                 width = 144,#验证码图片的宽
                 height = 48,#验证码图片的高
                 char_num = 4,#验证码字符个数
                 characters = string.digits + string.ascii_uppercase + string.ascii_lowercase):#验证码组成，数字+大写字母+小写字母
        self.width = width
        self.height = height
        self.char_num = char_num
        self.characters = characters
        self.classes = len(characters)

    def gen_captcha(self,batch_size = 50):
        X = np.zeros([batch_size,self.height,self.width,1])
        Y = np.zeros([batch_size,self.char_num,self.classes])
        image = ImageCaptcha(width = self.width,height = self.height)

        while True:
            for i in range(batch_size):
                img, captcha_str = create_validate_code(draw_lines=True, draw_points=True)
                # captcha_str = ''.join(random.sample(self.characters, self.char_num))
                # img = image.generate_image(captcha_str,noise_dot=False,color_draw=(255, 0, 0)).convert('L')
                img = np.array(img.getdata())[:,[0]]
                # img = np.array(img.getdata())
                X[i] = np.reshape(img, [self.height, self.width, 1]) / 255.0
                for j, ch in enumerate(captcha_str):
                    Y[i, j, self.characters.find(ch)] = 1
            Y = np.reshape(Y,(batch_size,self.char_num*self.classes))
            yield X,Y

    def decode_captcha(self,y):
        y = np.reshape(y,(len(y),self.char_num,self.classes))
        return ''.join(self.characters[x] for x in np.argmax(y,axis = 2)[0,:])

    def get_parameter(self):
        return self.width,self.height,self.char_num,self.characters,self.classes

    def gen_test_captcha(self):
        image = ImageCaptcha(width = self.width,height = self.height)
        captcha_str = ''.join(random.sample(self.characters,self.char_num))
        img = image.generate_image(captcha_str)
        img.save(captcha_str + '.jpg')

import tensorflow as tf
import math

import tensorflow as tf
import numpy as np
import string

captcha = generateCaptcha()
width,height,char_num,characters,classes = captcha.get_parameter()

import tensorflow as tf
channels = 1
n_inputs = height * width

conv1_fmaps = 32
conv1_ksize = [1,3]
conv1_stride = 4
conv1_pad = "SAME"

conv2_fmaps = 64
conv2_ksize = [1,3]
conv2_stride = 1
conv2_pad = "SAME"
conv2_dropout_rate = 0.25

pool3_fmaps = conv2_fmaps

n_fc1 = 128
fc1_dropout_rate = 0.5
n_outputs = char_num*classes
with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, height, width, channels], name="X")
    y = tf.placeholder(tf.float32, shape=[None,char_num*classes], name="y")
    training = tf.placeholder_with_default(False, shape=[], name='training')

conv1 = tf.layers.conv2d(X, filters=conv1_fmaps, kernel_size=conv1_ksize,
                         strides=conv1_stride, padding=conv1_pad,
                         activation=tf.nn.relu, name="conv1")
# pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,
                         strides=conv2_stride, padding=conv2_pad,
                         activation=tf.nn.relu, name="conv2")

with tf.name_scope("pool3"):
    pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    conv3 = tf.layers.conv2d(pool3, filters=96, kernel_size=3,
                             strides=1, padding=conv1_pad,
                             activation=tf.nn.relu, name="conv3")
    conv4 = tf.layers.conv2d(conv3, filters=48, kernel_size=3,
                             strides=1, padding=conv1_pad,
                             activation=tf.nn.relu, name="conv4")
    # conv5 = tf.layers.conv2d(conv4, filters=64, kernel_size=3,
    #                          strides=1, padding=conv1_pad,
    #                          activation=tf.nn.relu, name="conv5")
    pool3_flat = tf.reshape(conv4, shape=[-1, 48 * 18 * 6])
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
    predict = tf.reshape(logits, [-1, 4, classes])
    real = tf.reshape(y, [-1, 4, classes])
    correct= tf.reduce_all(tf.equal(tf.argmax(predict,2), tf.argmax(real,2)),axis=1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
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

from sklearn.model_selection import train_test_split
# X_data,y_data = next(captcha.gen_captcha(10000))
# (X_train, X_test,y_train,y_test) = train_test_split(X_data, y_data, test_size=0.25, random_state=0)

# if __name__ == '__main__':
#     n_epochs = 10000
#     batch_size = 50
#     iteration = 0
#     best_loss_val = np.infty
#     check_interval = 500
#     checks_since_last_progress = 0
#     max_checks_without_progress = 20
#     best_model_params = None
#
#     with tf.Session() as sess:
#         init.run()
#         for epoch in range(n_epochs):
#             # for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
#             for i in range(200):
#                 X_batch, y_batch=next(captcha.gen_captcha(50))
#                 iteration += 1
#                 sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})
#                 if iteration % check_interval == 0:
#                     X_test, y_test = next(captcha.gen_captcha(100))
#                     loss_val = loss.eval(feed_dict={X: X_test, y: y_test})
#                     if loss_val < best_loss_val:
#                         best_loss_val = loss_val
#                         checks_since_last_progress = 0
#                         best_model_params = get_model_params()
#                     else:
#                         checks_since_last_progress += 1
#             acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
#             X_test, y_test = next(captcha.gen_captcha(100))
#             acc_val = accuracy.eval(feed_dict={X: X_test, y: y_test})
#             print("Epoch {}, last batch accuracy: {:.4f}%, valid. accuracy: {:.4f}%, valid. best loss: {:.6f}".format(
#                 epoch, acc_batch * 100, acc_val * 100, best_loss_val))
#             if checks_since_last_progress > max_checks_without_progress:
#                 print("Early stopping!")
#                 break
#
#         if best_model_params:
#             restore_model_params(best_model_params)
#         acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
#         print("Final accuracy on test set:", acc_test)
#         save_path = saver.save(sess, "./captchas_no_split_model_160_60")

with tf.Session() as sess:
    saver.restore(sess,"./captchas_no_split_model_160_60")
    data,labels=[],[]
    from check_code import create_validate_code
    img, captcha_str = create_validate_code(draw_lines=True, draw_points=True)
    img.save("test.png")
    print("str is {}".format(captcha_str))
    from skimage import io, transform
    import matplotlib.pyplot as plt
    img = io.imread("test.png")
    # img = transform.resize(img, (24, 72))
    # plt.imshow(img, plt.cm.gray)
    # plt.show()
    # image = np.expand_dims(img, axis=2)
    image = img[:,:,[0]]
    data.append(image)
    test_data = np.array(data, dtype="float") / 255.0
    y_predict = Y_proba.eval(feed_dict={X: test_data})
    for i in np.argmax(np.reshape(y_predict,(-1,4,classes)),2):
        s = ''
        for j in i:
            s += characters[j]
        print(s)