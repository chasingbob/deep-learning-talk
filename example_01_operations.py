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

add = A + B
mul = tf.matmul(A, B)

print(sess.run(add))
print(sess.run(mul))

