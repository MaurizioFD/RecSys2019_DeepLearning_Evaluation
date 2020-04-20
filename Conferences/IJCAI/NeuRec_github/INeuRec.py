import tensorflow as tf
import numpy as np
import time
import random
import math

from Conferences.IJCAI.NeuRec_github.eval import *


class INeuRec():
    def __init__(self, sess, num_users, num_items, num_training, num_factors, learning_rate, reg_rate, epochs,
                 batch_size, display_step):
        self.num_users = num_users
        self.num_items = num_items
        self.num_training = num_training
        self.num_factors = num_factors
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.display_step = display_step
        self.reg_rate = reg_rate
        self.sess = sess

    def run(self, train_data, train_user_item_matrix, unique_users, neg_train_matrix, test_matrix):

        self.cf_user_input = tf.placeholder(dtype=tf.int32, shape=[None], name='cf_user_input')
        self.cf_item_input = tf.placeholder(dtype=tf.int32, shape=[None], name='cf_item_input')
        self.y = tf.placeholder("float", [None], 'y')
        isTrain = tf.placeholder(tf.bool, shape=())
        self.keep_rate_net = tf.placeholder(tf.float32)
        self.keep_rate_input = tf.placeholder(tf.float32)
        self.keep_rate_output = tf.placeholder(tf.float32)

        hidden_dim_1 = 150
        hidden_dim_2 = 150
        hidden_dim_3 = 150
        hidden_dim_4 = 150
        hidden_dim_5 = 40
        hidden_dim_6 = 40

        R = tf.constant(train_user_item_matrix, dtype=tf.float32)

        P = tf.Variable(tf.random_normal([self.num_users, hidden_dim_6], stddev=0.005))

        user_factor = tf.nn.embedding_lookup(P, self.cf_user_input)

        item_factor = tf.cond(isTrain, lambda: tf.nn.dropout(tf.nn.embedding_lookup(R, self.cf_item_input), 0.97),
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
        layer_5 = tf.sigmoid(tf.matmul(layer_4, W5) + b5)

        self.pred_y = tf.reduce_sum(tf.nn.dropout(tf.multiply(user_factor, layer_5), 1), 1)  # tf.reshape(output, [-1])

        self.loss = tf.reduce_sum(tf.square(self.y - self.pred_y)) + \
                    self.reg_rate * (tf.norm(P) + tf.norm(W1) + tf.norm(W2) + tf.norm(W3) + tf.norm(W4) + tf.norm(W5))


        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        # initialize model
        init = tf.global_variables_initializer()
        total_batch = int(self.num_training / self.batch_size)
        print(total_batch)
        self.sess.run(init)

        temp = train_data.tocoo()

        item = temp.row.reshape(-1)
        user = temp.col.reshape(-1)
        rating = temp.data

        # train and test the model
        for epoch in range(self.epochs):
            idxs = np.random.permutation(self.num_training)
            user_random = list(user[idxs])
            item_random = list(item[idxs])
            rating_random = list(rating[idxs])

            for i in range(total_batch):
                start_time = time.time()
                batch_user = user_random[i * self.batch_size:(i + 1) * self.batch_size]
                batch_item = item_random[i * self.batch_size:(i + 1) * self.batch_size]
                batch_rating = rating_random[i * self.batch_size:(i + 1) * self.batch_size]

                _, c = self.sess.run([self.optimizer, self.loss], feed_dict={self.cf_user_input: batch_user,
                                                                             self.cf_item_input: batch_item,
                                                                             self.y: batch_rating,
                                                                             isTrain: True})
                avg_cost = c
                if i % self.display_step == 0:
                    print("Index: %04d; Epoch: %04d; cost= %.9f" % (i + 1, epoch, np.mean(avg_cost)))

            if (epoch) % (2) == 0 and epoch >= 0:

                pred_ratings_10 = {}
                pred_ratings_5 = {}
                pred_ratings = {}
                ranked_list = {}
                count = 0
                p_at_5 = []
                p_at_10 = []
                r_at_5 = []
                r_at_10 = []
                map = []
                mrr = []
                ndcg = []
                ndcg_at_5 = []
                ndcg_at_10 = []

                learned_item_factors = self.sess.run(layer_5, feed_dict={self.cf_item_input: np.arange(self.num_items),
                                                                         isTrain: False})
                learned_user_factors = self.sess.run(user_factor,
                                                     feed_dict={self.cf_user_input: np.arange(self.num_users),
                                                                isTrain: False})

                results = np.dot(learned_user_factors, np.transpose(learned_item_factors))

                for u in unique_users:
                    count += 1
                    user_neg_items = neg_train_matrix[u]
                    item_ids = []
                    scores = []

                    for j in user_neg_items:
                        item_ids.append(j)
                        scores.append(results[u, j])

                    neg_item_index = list(zip(item_ids, scores))

                    ranked_list[u] = sorted(neg_item_index, key=lambda tup: tup[1], reverse=True)
                    pred_ratings[u] = [r[0] for r in ranked_list[u]]
                    pred_ratings_5[u] = pred_ratings[u][:5]
                    pred_ratings_10[u] = pred_ratings[u][:10]

                    p_5, r_5, ndcg_5 = precision_recall_ndcg_at_k(5, pred_ratings_5[u], test_matrix[u])
                    p_at_5.append(p_5)
                    r_at_5.append(r_5)
                    ndcg_at_5.append(ndcg_5)
                    p_10, r_10, ndcg_10 = precision_recall_ndcg_at_k(10, pred_ratings_10[u], test_matrix[u])
                    p_at_10.append(p_10)
                    r_at_10.append(r_10)
                    ndcg_at_10.append(ndcg_10)
                    map_u, mrr_u, ndcg_u = map_mrr_ndcg(pred_ratings[u], test_matrix[u])
                    map.append(map_u)
                    mrr.append(mrr_u)
                    ndcg.append(ndcg_u)

                print("-------------------------------")
                print("precision@10:" + str(np.mean(p_at_10)))
                print("recall@10:" + str(np.mean(r_at_10)))
                print("precision@5:" + str(np.mean(p_at_5)))
                print("recall@5:" + str(np.mean(r_at_5)))
                print("map:" + str(np.mean(map)))
                print("mrr:" + str(np.mean(mrr)))
                print("ndcg:" + str(np.mean(ndcg)))
