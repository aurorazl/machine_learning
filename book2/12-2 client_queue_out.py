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
# dequeue = q.dequeue()
batch_size = 3
dequeue_batch = q.dequeue_many(batch_size)

# dequeue = q.dequeue()
# dequeue_a, dequeue_b = q.dequeue()
# dequeue_as, dequeue_bs = q.dequeue_many(batch_size)

with tf.Session("grpc://localhost:2221") as sess:
    print(sess.run(dequeue_batch,))
    # a_val, b_val = sess.run([dequeue_a, dequeue_b])
    # print(a_val)  # 10
    # print(b_val)  # [[1., 2.], [3., 4.], [5., 6.]]
    # print(sess.run(dequeue_a))

    # print(sess.run([dequeue_as, dequeue_bs]))