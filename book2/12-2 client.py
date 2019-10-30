import tensorflow as tf

# a = tf.constant(1.0)
# b = a + 2
# c = a * 3
# with tf.Session("grpc://localhost:2220") as sess:
#         print(c.eval()) # 9.0

with tf.device("/job:ps/task:0/cpu:0"):
    a = tf.constant(1.0)
with tf.device("/job:worker/task:0/cpu:0"):
    b = a + 2
c = a + b
# with tf.Session("grpc://localhost:2221") as sess:
#     print(c.eval())

cluster_spec = tf.train.ClusterSpec({
        "ps": [
                "localhost:2220", # /job:ps/task:0
                ],
        "worker": [
                "localhost:2221", # /job:worker/task:0
                "localhost:2222", # /job:worker/task:1
                ]})
# with tf.device(tf.train.replica_device_setter(cluster=cluster_spec)):
#     v1 = tf.Variable(1.0) # pinned to /job:ps/task:0
#     v2 = tf.Variable(2.0) # pinned to /job:ps/task:1
#     v3 = tf.Variable(3.0) # pinned to /job:ps/task:0
#     v4 = tf.Variable(4.0) # pinned to /job:ps/task:1
#     v5 = tf.Variable(5.0) # pinned to /job:ps/task:0

with tf.device(tf.train.replica_device_setter(ps_tasks=1)):
    v1 = tf.Variable(1.0) # pinned to /job:ps/task:0 (+ defaults to /cpu:0)
    v2 = tf.Variable(2.0) # pinned to /job:ps/task:1 (+ defaults to /cpu:0)
    v3 = tf.Variable(3.0) # pinned to /job:ps/task:0 (+ defaults to /cpu:0)
    s = v1 + v2 # pinned to /job:worker (+ defaults to task:0/gpu:0)
    with tf.device("/cpu:0"):
        p1 = 2 * s # pinned to /job:worker/gpu:1 (+ defaults to /task:0)
        with tf.device("/task:1"):
            p2 = 3 * s # pinned to /job:worker/task:1/gpu:1

with tf.Session("grpc://localhost:2221") as sess:
    print(c.eval())