from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath('../safeai')))

import tensorflow as tf
import numpy as np
from keras.datasets.cifar10 import load_data
from safeai.models.classification import build_CNN_classifier

def next_batch(num, data, labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def main():
    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    y = tf.placeholder(tf.float32, shape=[None, 10])
    keep_prob = tf.placeholder(tf.float32)

    (x_train, y_train), (x_test, y_test) = load_data()
    y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10),axis=1)
    y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 10),axis=1)

    y_pred, logits = build_CNN_classifier(x)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
    train_step = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(10000):
            batch = next_batch(128, x_train, y_train_one_hot.eval())

            
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
                loss_print = loss.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})

                print("Epoch: %d, train data accuracy: %f, loss: %f" % (i, train_accuracy, loss_print))
            
            sess.run(train_step, feed_dict={x: batch[0], y: batch[1], keep_prob: 0.8})

      
        test_accuracy = 0.0  
        for i in range(10):
            test_batch = next_batch(1000, x_test, y_test_one_hot.eval())
            test_accuracy = test_accuracy + accuracy.eval(feed_dict={x: test_batch[0], y: test_batch[1], keep_prob: 1.0})
        test_accuracy = test_accuracy / 10;
        print("test data accuracy: %f" % test_accuracy)
        
if __name__ == '__main__':
    main()        