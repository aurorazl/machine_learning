import tensorflow as tf
import sys

x = tf.Variable(0.0, name="x")

# with tf.variable_scope("my_problem_1"):
#     x = tf.Variable(0.0, name="x")

# with tf.container("myproblem"):
#     x = tf.Variable(0.0, name="x")

increment_x = tf.assign(x, x + 1)
tf.Session.reset("grpc://localhost:2221", ["myproblem"])
with tf.Session("grpc://localhost:2221") as sess:
    if sys.argv[1:]==["init"]:
        sess.run(x.initializer)
    sess.run(increment_x)
    print(x.eval())