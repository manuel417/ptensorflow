import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#x setup x
x = tf.placeholder(tf.float32, [None, 784])

#set W - weights matrix
#One row of W is params for one classifier for a given class
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#x[i] is a row vector
# invert the order to get effect of transpose
y = tf.nn.softmax(tf.matmul(x, W) + b)

# loss function
y_ = tf.placeholder(tf.float32, [None, 10])

#cross_entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#training
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#session
sess = tf.InteractiveSession()

#global variables
tf.global_variables_initializer().run()

#train this stuff
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict= {x: batch_xs, y_: batch_ys})


#evaluate
correct_prediction = tf.equal(tf.arg_max(y,1), tf.argmax(y_, 1))

#cart T/F vector to bit vector
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("Accuracy: ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
