import tensorflow as tf

node1  = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # implicity float 32
print(node1, node2)

#run the session
sess = tf.Session()
print(sess.run([node1, node2]))

# add nodes
node3 = tf.add(node1, node2)
print("node3: ", node3)
print("sess.run(node3): ", sess.run(node3))

#promises
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b

#pass parameters as dictionary
print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a : [1,3], b: [2,4]}))

# more complex
add_triple = adder_node * 3.
print(sess.run(add_triple, {a: 3, b: 4}))
print(sess.run(add_triple, {a: [1, 3], b: [2, 4]}))

# variables
W = tf.Variable([0.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

#init variables
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(linear_model, {x: [10]}))
print(sess.run(linear_model, {x: 10}))
print(sess.run(linear_model, {x: [2, 4 , 10, 100]}))

# bring y variables
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1,2,3,4], y:[0,-1,-2,-3]}))

# change model
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1,2,3,4], y: [0,-1,-2,-3]}))

#Gradient descent
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init) #reset all variable to default so we see how they get corrected
for i in range(1000):
    sess.run(train, {x: [1,2,3,4], y:[0,-1,-2,-3]})

print(sess.run([W,b]))
