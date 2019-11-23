import numpy as np
from sklearn.datasets import load_sample_images
import tensorflow as tf
import matplotlib.pyplot as plt
# Load sample images
dataset = np.array(load_sample_images().images, dtype=np.float32)
batch_size, height, width, channels = dataset.shape
print(batch_size, height, width, channels)
# Create 2 filters
filters_test = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
filters_test[:, 3, :, 0] = 1 # vertical line
filters_test[3, :, :, 1] = 1 # horizontal line
# Create a graph with input X plus a convolutional layer applying the 2 filters
X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
# max_pool = tf.nn.max_pool(X, ksize=[1,2,2,1], strides=[1,2,2,1],padding="VALID")
# convolution = tf.nn.conv2d(X, filters_test, strides=[1,2,2,1], padding="SAME")
conv1 = tf.layers.conv2d(X, filters=7, kernel_size=3,
                         strides=1, padding="SAME",
                         activation=tf.nn.relu, name="conv1")
init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    output = sess.run(conv1, feed_dict={X: dataset})
    print(output.shape)
    # plt.imshow(output[0, :, :, 1]) # plot 1st image's 2nd feature map
    # plt.show()

def plot_image(image):
    plt.imshow(image, cmap="gray", interpolation="nearest")
    plt.axis("off")

# for image_index in (0, 1):
#     for feature_map_index in (0, 1):
#         plt.imshow(output[image_index, :, :, feature_map_index])
#         plt.show()

