"""
TensorFlow basics
"""

import tensorflow as tf


# Our first computational graph
a = tf.constant(3.0, tf.float32)
b = tf.constant(4.0)

print(a, b)

# Run a session
sess = tf.Session()
print(sess.run([a, b]))

# Placeholders - external inputs, it's a contract
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

# Variable
c = tf.Variable(1, name="var1")

print('\r\nExecute graph')
adder_node = a + b
print(sess.run(adder_node, {a: 5, b: 1.3}))
print(sess.run(adder_node, {a: [1,2], b: [3,5]}))

adder_mul_three = adder_node * 3
print(sess.run(adder_mul_three, {a: 2, b:2}))





