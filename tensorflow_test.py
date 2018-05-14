import tensorflow as tf
import numpy as np
import numpy

A = [[1, 3, 4, 5, 6]]
B = [[1, 3, 4, 3, 2]]

with tf.Session() as sess:
    correct_prediction = numpy.equal(A, B)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy))
