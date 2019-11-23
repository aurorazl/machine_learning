import tensorflow as tf

cluster_spec = tf.train.ClusterSpec({
        "ps": [
                "localhost:2223", # /job:ps/task:0
                ],
        "worker": [
                "localhost:2224", # /job:worker/task:0
                "localhost:2225", # /job:worker/task:1
                ]})
server = tf.train.Server(cluster_spec, job_name="worker", task_index=1)
server.join() # blocks until the server stops (i.e., never)