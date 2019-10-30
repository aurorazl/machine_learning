import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
m,n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m,1)),housing.data]
scaled_housing_data_plus_bias = StandardScaler().fit_transform(housing_data_plus_bias)
n_epochs = 10
learn_rate=0.01
# x = tf.constant(scaled_housing_data_plus_bias,dtype=tf.float32,name='x')
# y = tf.constant(housing.target.reshape(-1,1),dtype=tf.float32,name='y')
x = tf.placeholder(tf.float32,shape=(None,n+1),name='x')
y = tf.placeholder(tf.float32,shape=(None,1),name='y')
theta = tf.Variable(tf.random_uniform([n+1,1],-1.0,1.0),name='theta')
y_pred = tf.matmul(x,theta,name='predictions')
with tf.name_scope("loss") as scope:
    error=y_pred-y
    mse = tf.reduce_mean(tf.square(error),name='mse')
# gradients = 2/m * tf.matmul(tf.transpose(x),error)
# gradients = tf.gradients(mse,[theta])[0]
# training_op = tf.assign(theta,theta-learn_rate*gradients)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate)
training_op = optimizer.minimize(mse)
init = tf.global_variables_initializer()
batch_size = 100
n_batches =int(np.ceil(m/batch_size))
# with tf.Session() as sess:
#     sess.run(init)
#     for epoch in range(n_epochs):
#         if epoch%100==0:
#             print(epoch,mse.eval())
#         sess.run(training_op)
#     best_theta = theta.eval()

def fetch_batch(epoch,batch_index,batch_size):
    know = np.random.seed(epoch*n_batches+batch_index)
    indices = np.random.randint(m,size=batch_size)
    x_batch = scaled_housing_data_plus_bias[indices]
    y_batch = housing.target.reshape(-1,1)[indices]
    return x_batch,y_batch
# saver = tf.train.Saver()
from datetime import datetime
now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
root_logdir = 'tf_logs'
logdir = "{}/run-{}/".format(root_logdir,now)
mse_summary = tf.summary.scalar("MSE",mse)
file_writer = tf.summary.FileWriter(logdir,tf.get_default_graph())
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            x_batch,y_batch = fetch_batch(epoch,batch_index,batch_size)
            if batch_index%10==0:
                summary_str = mse_summary.eval(feed_dict={x:x_batch,y:y_batch})
                step = epoch*n_batches + batch_index
                file_writer.add_summary(summary_str,step)
            sess.run(training_op,feed_dict={x:x_batch,y:y_batch})
    best_theta = theta.eval()
    # saver.save(sess, "my_model_final.ckpt")
print(best_theta)
file_writer.close()