'''
A linear regression example implemented using TensorFlow
'''

from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
rnd = np.random

plt.ion()
plt.show()

learning_rate = 0.01
training_epochs = 1000
display_step = 50

# Training Data
train_X = np.asarray([x for x in range(20)]) * 0.1
train_Y = np.linspace(0,4.5,20)
noise = np.random.normal(-0.2, 0.2, 20)
train_Y += noise

n_samples = train_X.shape[0]

#plt.scatter(train_X, train_Y, color='b')
#input("Press ENTER to continue")

# Placeholders
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Init model with random weights
W = tf.Variable(rnd.randn())
b = tf.Variable(rnd.randn())

# linear model (y = Wx + b)
pred = tf.add(tf.multiply(X, W), b)

# Mean squared error (MSE)
loss = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
loss_vec = []
steps = []
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(1000):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            l = sess.run(loss, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "loss=", "{:.9f}".format(l), \
                "W=", sess.run(W), "b=", sess.run(b))
            loss_vec.append(l)
            steps.append(epoch)

    print("Optimization Finished!")
    training_loss = sess.run(loss, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_loss, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

    plt.plot(steps, loss_vec, color='r', label='Loss')
    plt.legend()
    plt.show()


    input('Press ENTER')
