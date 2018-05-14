from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import tensorflow as tf

# user input
from argparse import ArgumentParser
import numpy as np
# refernce argument values
from deep_image_load_data import load_data
from image_cnn_model import Image_CNN

MODEL_DIRECTORY = "model"
TEST_BATCH_SIZE = 5000


# test with test data given by mnist_data.py
def test(data_dir, model_directory, batch_size, n_label):
    # Import data
    test_data, test_labels, _, _ = load_data(
        data_dir, n_label, validation_rate=0, is_expanding=False)

    is_training = tf.placeholder(tf.bool, name='MODE')

    # tf Graph input
    x = tf.placeholder(tf.float32, np.append(None, test_data[0].shape))
    y_ = tf.placeholder(tf.float32, [None, n_label])
    # Predict
    y = Image_CNN(x, n_label, is_training=is_training)

    # Add ops to save and restore all the variables
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer(), feed_dict={is_training: False})

    # Restore variables from disk
    saver = tf.train.Saver()

    # Calculate accuracy for all mnist test images
    test_size = len(test_data)
    print("size of test data: %d" % len(test_data))
    if test_size < batch_size:
        batch_size = test_size
    total_batch = int(test_size / batch_size)

    saver.restore(sess, model_directory)

    acc_buffer = []

    for i in range(total_batch):
        # Compute the offset of the current minibatch in the data.
        offset = (i * batch_size) % (test_size)
        batch_xs = test_data[offset:(offset + batch_size)]
        batch_ys = test_labels[offset:(offset + batch_size)]

        y_final = sess.run(y, feed_dict={x: batch_xs, y_: batch_ys, is_training: False})
        correct_prediction = numpy.equal(numpy.argmax(y_final, 1), numpy.argmax(batch_ys, 1))
        acc_buffer.append(numpy.sum(correct_prediction) / batch_size)

    print("test accuracy for the stored model: %g" % numpy.mean(acc_buffer))


# build parser
def build_parser():
    parser = ArgumentParser()

    parser.add_argument('--model-dir',
                        dest='model_directory', help='directory where model to be tested is stored',
                        metavar='MODEL_DIRECTORY', required=True)
    parser.add_argument('--test-data-dir',
                        dest='test_data_dir', help='the directory of test data',
                        metavar='TEST_DATA_DIR', required=True)
    parser.add_argument('--n_label', type=int,
                        dest='n_label', help='the number of labels',
                        metavar='N_LABEL', required=True)
    parser.add_argument('--batch-size', type=int,
                        dest='batch_size', help='batch size for test',
                        metavar='BATCH_SIZE', required=True)
    return parser


if __name__ == '__main__':
    # Parse argument
    parser = build_parser()
    options = parser.parse_args()

    model_directory = options.model_directory
    test_data_dir = options.test_data_dir
    n_label = options.n_label
    batch_size = options.batch_size
    test(test_data_dir, model_directory + '/model.ckpt', batch_size, n_label)
    # test('dset2', 'model' + '/model.ckpt', 500, 2)
