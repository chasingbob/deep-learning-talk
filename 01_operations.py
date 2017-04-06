"""
Example showing basic matrix operations in TensorFlow

"""


import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

sess = tf.Session()

# 2x2 Matrix
A = tf.convert_to_tensor(np.array([[1., 2.], [4., 5.]]))

# 2x2 Matrix
B = tf.convert_to_tensor(np.array([[-1., -3.], [2., -1.]]))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

add = A + B
mul = tf.matmul(A, B)
mul2 = tf.matmul(a,b)

print('Add:')
print(sess.run(add))

print('\r\nMultiply 1:')
print(sess.run(mul))

print('\r\nMultiply 2')
print(sess.run(mul2, {a: [[1,2,3],[4,5,6]], b: [[7,8],[9,10],[11,12]]}))


