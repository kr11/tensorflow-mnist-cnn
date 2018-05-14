# Some code was borrowed from https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/models/image/mnist/convolutional.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from argparse import ArgumentParser

import tensorflow as tf
import tensorflow.contrib.slim as slim
import time

from deep_image_load_data import load_data
from image_cnn_model import Image_CNN

MODEL_DIRECTORY = "model/model.ckpt"
LOGS_DIRECTORY = "logs/train"

# Params for Train
training_epochs = 10  # 10 for augmented training data, 20 for training data
TRAIN_BATCH_SIZE = 50
display_step = 10
validation_step = 50


# Params for test
# TEST_BATCH_SIZE = 5000


def train(batch_size, data_dir, n_label, is_expanding):
    # Some parameters

    # Prepare mnist data
    train_data, train_labels, validation_data, validation_labels = load_data(
        data_dir, n_label, validation_rate=0.2, is_expanding=is_expanding)
    train_size = len(train_data)
    if train_size == 0:
        raise Exception("no train data!")
    # Boolean for MODE of train or test
    is_training = tf.placeholder(tf.bool, name='MODE')

    # tf Graph input
    x = tf.placeholder(tf.float32, np.append(None, train_data[0].shape))
    y_ = tf.placeholder(tf.float32, [None, n_label])

    # Predict
    y = Image_CNN(x, n_label)

    # Get loss of model
    with tf.name_scope("LOSS"):
        loss = slim.losses.softmax_cross_entropy(y, y_)

    # Create a summary to monitor loss tensor
    tf.summary.scalar('loss', loss)

    # Define optimizer
    with tf.name_scope("ADAM"):
        # Optimizer: set up a variable that's incremented once per batch and
        # controls the learning rate decay.
        batch = tf.Variable(0)

        learning_rate = tf.train.exponential_decay(
            1e-4,  # Base learning rate.
            batch * batch_size,  # Current index into the dataset.
            train_size,  # Decay step.
            0.95,  # Decay rate.
            staircase=True)
        # Use simple momentum for the optimization.
        # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=batch)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=batch)

    # Create a summary to monitor learning_rate tensor
    tf.summary.scalar('learning_rate', learning_rate)

    # Get accuracy of model
    with tf.name_scope("ACC"):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Create a summary to monitor accuracy tensor
    tf.summary.scalar('acc', accuracy)

    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()

    # Add ops to save and restore all the variables
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer(), feed_dict={is_training: True})

    # Training cycle
    total_batch = int(train_size / batch_size)

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(LOGS_DIRECTORY, graph=tf.get_default_graph())

    # Save the maximum accuracy value for validation data
    max_acc = 0.

    # Loop for epoch
    for epoch in range(training_epochs):

        # Random shuffling
        data_labels = list(zip(train_data, train_labels))
        np.random.shuffle(data_labels)
        # train_data_ = train_total_data[:, :-num_labels]
        # train_labels_ = train_total_data[:, -num_labels:]
        train_data[:], train_labels[:] = zip(*data_labels)
        # Loop over all batches
        for i in range(total_batch):
            # Compute the offset of the current minibatch in the data.
            offset = (i * batch_size) % (train_size)
            batch_xs = train_data[offset:(offset + batch_size)]
            batch_ys = train_labels[offset:(offset + batch_size)]

            # Run optimization op (backprop), loss op (to get loss value)
            # and summary nodes
            train_loss, _, train_accuracy, summary = sess.run([loss, train_step, accuracy, merged_summary_op],
                                                              feed_dict={x: batch_xs, y_: batch_ys, is_training: True})

            # Write logs at every iteration
            summary_writer.add_summary(summary, epoch * total_batch + i)

            # Display logs
            if i % display_step == 0:
                print("Epoch:", '%04d,' % (epoch + 1),
                      "batch_index %4d/%4d, training accuracy %.5f, loss %.5f" % (
                          i, total_batch, train_accuracy, train_loss))

            # Get accuracy for validation data
            if i % validation_step == 0:
                # Calculate accuracy
                validation_accuracy, valid_loss = sess.run([accuracy, loss],
                                                           feed_dict={x: validation_data, y_: validation_labels,
                                                                      is_training: False})

                print("Epoch:", '%04d,' % (epoch + 1),
                      "batch_index %4d/%4d, validation accuracy %.5f, loss %.5f" %
                      (i, total_batch, validation_accuracy, valid_loss))

            # Save the current model if the maximum accuracy is updated
            if validation_accuracy > max_acc:
                max_acc = validation_accuracy
                save_path = saver.save(sess, MODEL_DIRECTORY)
                print("Model updated and saved in file: %s" % save_path)

    print("Optimization Finished!, max acc: %.5f" % max_acc)

    # Restore variables from disk
    saver.restore(sess, MODEL_DIRECTORY)

# build parser
def build_parser():
    parser = ArgumentParser()

    parser.add_argument('--train-data-dir',
                        dest='train_data_dir', help='the directory of test data',
                        metavar='TRAIN_DATA_DIR', required=True)
    parser.add_argument('--n_label', type=int,
                        dest='n_label', help='the number of labels',
                        metavar='N_LABEL', required=True)
    parser.add_argument('--batch-size', type=int,
                        dest='batch_size', help='batch size for test',
                        metavar='BATCH_SIZE', required=True)
    parser.add_argument('--use-data-aug', type=bool,
                        dest='use_data_aug', help='use data augmentation',
                        metavar='USE_DATA_AUG', required=True)
    return parser


if __name__ == '__main__':
    """
    python image_cnn_train.py --train-data-dir dset1/train --n_label 65 --batch-size 50 --use-data-aug False
    """
    # train_data_dir = 'dset2/train'
    # batch_size = 65
    # batch_size = TRAIN_BATCH_SIZE
    # use_data_aug = False

    # Parse argument
    parser = build_parser()
    options = parser.parse_args()
    train_data_dir = options.train_data_dir
    n_label = options.n_label
    batch_size = options.batch_size
    use_data_aug = options.use_data_aug

    start = time.time()
    train(batch_size, train_data_dir, n_label, use_data_aug)
    end = time.time()
    print("total time cost: %d" % (end - start))
