import tensorflow as tf

with tf.device('/cpu:0'):
    a = tf.Variable(3)
    b = tf.Variable(4)
c = a*b

config = tf.ConfigProto()
config.log_device_placement = True
init = tf.global_variables_initializer()
with tf.Session(config=config) as sess:
    init.run()
    sess.run(c)