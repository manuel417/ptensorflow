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


