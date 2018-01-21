import tensorflow as tf

# TensorFlow session
sess = tf.Session()

# Data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]

# Parameters
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# Linear Regression Model
linear_model = W * x + b
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

# Training
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# Initialize predefined variables
init = tf.global_variables_initializer()
sess.run(init)

print("BEFORE TRAINING")
print("W, b: ", sess.run([W, b]))
print("Loss: ", sess.run(loss, {x: x_train, y: y_train}))
print("Predictions: ", sess.run(linear_model, {x: x_train}))
print()

for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})

print("AFTER TRAINING")
print("W, b: ", sess.run([W, b]))
print("Loss: ", sess.run(loss, {x: x_train, y: y_train}))
print("Predictions: ", sess.run(linear_model, {x: x_train}))