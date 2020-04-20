#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Simone Boglio
"""

import tensorflow as tf
import numpy as np
import time


class INeuRec():
    def __init__(self, tf_session, num_neurons, num_factors, dropout_percentage, learning_rate, regularization_rate, max_epochs,
                 batch_size, display_epoch=None, display_step=None, verbose=True):
        self.sess = tf_session
        self.num_neurons = num_neurons
        self.num_factors = num_factors
        self.dropout_percentage = dropout_percentage
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.display_epoch = display_epoch
        self.display_step = display_step
        self.reg_rate = regularization_rate
        self.verbose = verbose
        self.current_epoch = 0
        self.total_time = 0

    def fit(self, urm):

        start_time = time.time()

        if self.verbose: print('INeuRec: init..')

        self.num_users, self.num_items = urm.shape
        self.num_training = self.num_users * self.num_items

        # use these 4 lines for speedup the slow part
        train = urm.T.todense()

        # arrays used in sampling
        self.rating = urm.todense().flatten().A1
        self.user = np.repeat(np.arange(0, self.num_users), self.num_items)
        self.item = np.tile(np.arange(0, self.num_items).transpose(), self.num_users)

        self.cf_user_input = tf.placeholder(dtype=tf.int32, shape=[None], name='cf_user_input')
        self.cf_item_input = tf.placeholder(dtype=tf.int32, shape=[None], name='cf_item_input')
        self.y = tf.placeholder("float", [None], 'y')
        self.isTrain = tf.placeholder(tf.bool, shape=())
        self.keep_rate_net = tf.placeholder(tf.float32)
        self.keep_rate_input = tf.placeholder(tf.float32)
        self.keep_rate_output = tf.placeholder(tf.float32)

        hidden_dim_1 = self.num_neurons
        hidden_dim_2 = self.num_neurons
        hidden_dim_3 = self.num_neurons
        hidden_dim_4 = self.num_neurons
        hidden_dim_5 = self.num_factors
        hidden_dim_6 = self.num_factors

        R = tf.constant(train, dtype=tf.float32)

        P = tf.Variable(tf.random_normal([self.num_users, hidden_dim_6], stddev=0.005))

        self.user_factor = tf.nn.embedding_lookup(P, self.cf_user_input)

        item_factor = tf.cond(self.isTrain, lambda: tf.nn.dropout(tf.nn.embedding_lookup(R, self.cf_item_input), 1-self.dropout_percentage),
                              lambda: tf.nn.embedding_lookup(R, self.cf_item_input))


        W1 = tf.Variable(tf.random_normal([self.num_users, hidden_dim_1]))
        W2 = tf.Variable(tf.random_normal([hidden_dim_1, hidden_dim_2]))
        W3 = tf.Variable(tf.random_normal([hidden_dim_2, hidden_dim_3]))
        W4 = tf.Variable(tf.random_normal([hidden_dim_3, hidden_dim_4]))
        W5 = tf.Variable(tf.random_normal([hidden_dim_4, hidden_dim_5]))

        b1 = tf.Variable(tf.random_normal([hidden_dim_1]))
        b2 = tf.Variable(tf.random_normal([hidden_dim_2]))
        b3 = tf.Variable(tf.random_normal([hidden_dim_3]))
        b4 = tf.Variable(tf.random_normal([hidden_dim_4]))
        b5 = tf.Variable(tf.random_normal([hidden_dim_5]))

        layer_1 = tf.sigmoid(tf.matmul(item_factor, W1) + b1)
        layer_2 = tf.sigmoid(tf.matmul(layer_1, W2) + b2)
        layer_3 = tf.sigmoid(tf.matmul(layer_2, W3) + b3)
        layer_4 = tf.sigmoid(tf.matmul(layer_3, W4) + b4)
        self.layer_5 = tf.sigmoid(tf.matmul(layer_4, W5) + b5)

        self.pred_y = tf.reduce_sum(tf.nn.dropout(tf.multiply(self.user_factor, self.layer_5), 1), 1)  # tf.reshape(output, [-1])

        self.loss = tf.reduce_sum(tf.square(self.y - self.pred_y)) + \
                    self.reg_rate * (tf.norm(P) + tf.norm(W1) + tf.norm(W2) + tf.norm(W3) + tf.norm(W4) + tf.norm(W5))


        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        # initialize model
        init = tf.global_variables_initializer()
        self.total_batch = int(self.num_training / self.batch_size)
        self.sess.run(init)

        if self.verbose: print('INeuRec: init done')

        self.total_time = time.time() - start_time

    def run_epoch(self):

        start_time = time.time()

        idxs = np.random.permutation(self.num_training)

        user_random = self.user[idxs].tolist()
        item_random = self.item[idxs].tolist()
        rating_random = self.rating[idxs].tolist()

        for i in range(self.total_batch):
            batch_user = user_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_item = item_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_rating = rating_random[i * self.batch_size:(i + 1) * self.batch_size]

            _, c = self.sess.run([self.optimizer, self.loss], feed_dict={self.cf_user_input: batch_user,
                                                                         self.cf_item_input: batch_item,
                                                                         self.y: batch_rating,
                                                                         self.isTrain: True})
            avg_cost = c
            if self.verbose and self.display_step is not None and i % self.display_step == 0:
                print("Batch: %04d; Epoch: %04d; cost= %.9f" % (i, self.current_epoch, np.mean(avg_cost)))

        self.current_epoch += 1

        self.total_time += time.time() - start_time

        if self.verbose and self.display_epoch is not None and self.current_epoch % self.display_epoch == 0:
            print("Epoch: %04d; cost= %.9f; exe_time= %s" % (self.current_epoch, np.mean(avg_cost), str(round(self.total_time, 2))))

        return avg_cost

    #
    # def run(self):
    #     for i in range(self.epochs):
    #         _ = self.run_epoch()
    #     if self.verbose: print("INeuRec: training completed")


    def get_factors(self):

        learned_item_factors = self.sess.run(self.layer_5,
                                                  feed_dict={self.cf_item_input: np.arange(self.num_items),
                                                             self.isTrain: False})
        learned_user_factors = self.sess.run(self.user_factor,
                                                  feed_dict={self.cf_user_input: np.arange(self.num_users),
                                                             self.isTrain: False})

        return learned_user_factors, learned_item_factors

