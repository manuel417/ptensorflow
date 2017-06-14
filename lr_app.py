import tensorflow as tf
import numpy as np

#declare the features for LR
features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

#Estimator - this another name for the classifier
estimator  = tf.contrib.learn.LinearRegressor(feature_columns=features)

#set_up_data_sets
x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])
input_fn  = tf.contrib.learn.io.numpy_input_fn({"x" : x}, y, batch_size=4, num_epochs=1000)

#epoch - passes over dataset
#batch_size - number of data examples processes in a step
# set- number of gradient descent steps (loops) to fit parameters

# Fit the data
estimator.fit(input_fn=input_fn, steps=1000)

# evaluate model
print(estimator.evaluate(input_fn=input_fn))

