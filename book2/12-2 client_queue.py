import tensorflow as tf
import sys

# x = tf.Variable(0.0, name="x")

# with tf.variable_scope("my_problem_1"):
#     x = tf.Variable(0.0, name="x")

with tf.container("myproblem"):
    x = tf.Variable(0.0, name="x")

# q = tf.FIFOQueue(capacity=10,dtypes=[tf.float32],shapes=[[2]],name="q",shared_name='shared_q')
# q = tf.FIFOQueue(capacity=10,dtypes=[tf.int32,tf.float32],shapes=[[],[3,2]],name="q2",shared_name='shared_q2')
# q = tf.RandomShuffleQueue(capacity=50, min_after_dequeue=10,dtypes=[tf.float32], shapes=[(1)],name="q4", shared_name="shared_q4")
q = tf.PaddingFIFOQueue(capacity=50, dtypes=[tf.float32], shapes=[(None, None)],name="q5", shared_name="shared_q5")
# training_instance=tf.placeholder(tf.float32,shape=(2))
# enqueue = q.enqueue([training_instance])

# training_instances = tf.placeholder(tf.float32, shape=(None,1))
# training_instances = tf.placeholder(tf.float32, shape=(None, 2))
# enqueue_many = q.enqueue_many(training_instances)

# a = tf.placeholder(tf.int32, shape=())
# b = tf.placeholder(tf.float32, shape=(3, 2))
# enqueue = q.enqueue((a, b))

v = tf.placeholder(tf.float32, shape=(None, None))
enqueue = q.enqueue([v])

close_q = q.close()

with tf.Session("grpc://localhost:2221") as sess:
    # sess.run(enqueue,feed_dict={training_instance:[1.,2.]})
    # sess.run(enqueue,feed_dict={training_instance:[3.,4.]})
    # sess.run(enqueue_many,feed_dict={training_instances:[[5.,6.],[7.,8.],[1.,2.],[3.,4.]]})
    # sess.run(enqueue_many,feed_dict={training_instances:[[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15],[16],[17],[18],[19],[20],[21],[22]]})
    # sess.run(enqueue, feed_dict={a: 10, b: [[1., 2.], [3., 4.], [5., 6.]]})
    sess.run(enqueue, feed_dict={v: [[1., 2.], [3., 4.], [5., 6.]]})  # 3x2
    sess.run(enqueue, feed_dict={v: [[1.]]})  # 1x1
    sess.run(enqueue, feed_dict={v: [[7., 8., 9., 5.], [6., 7., 8., 9.]]})  # 2x4
    print(q.size().eval())
    sess.run(close_q)