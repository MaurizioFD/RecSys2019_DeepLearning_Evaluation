# coding:utf8
import numpy as np
import tensorflow as tf
import time
from datetime import datetime
import cPickle as pickle
import math
import heapq
from processing_www_data import read_data, get_user_item_interaction_map

epochs = 200
learn_rate = 0.0001
batch_size = 32

num_negs = 1
neg_sample_size = 32 * num_negs
last_layer_size = 64

# *****************************************************************************************************
file_name = '../dataset/Amazon_ratings_Digital_Music_pruned.txt'
ratings, u_max_num, v_max_num = read_data(file_name)
user_map_item, latest_item_interaction, pruned_all_ratings = get_user_item_interaction_map(ratings)
# *****************************************************************************************************

pruned_user_map_item = {}
pruned_item_map_user = {}
for u, v, r, t in pruned_all_ratings:
    if u not in pruned_user_map_item:
        pruned_user_map_item[u] = {}
    if v not in pruned_item_map_user:
        pruned_item_map_user[v] = {}
    pruned_user_map_item[u][v] = r
    pruned_item_map_user[v][u] = r


# DSSM one hots
def inference_neural_DSSM_onehot(one_hot_u, one_hot_v):
    u_w1 = tf.get_variable("u_w1", shape=(v_max_num, 300), initializer=tf.contrib.layers.xavier_initializer())
    u_b1 = tf.get_variable("u_b1", shape=[300], initializer=tf.contrib.layers.xavier_initializer())
    u_w2 = tf.get_variable("u_w2", shape=(300, last_layer_size), initializer=tf.contrib.layers.xavier_initializer())
    u_b2 = tf.get_variable("u_b2", shape=[last_layer_size], initializer=tf.contrib.layers.xavier_initializer())

    v_w1 = tf.get_variable("v_w1", shape=(u_max_num, 300), initializer=tf.contrib.layers.xavier_initializer())
    v_b1 = tf.get_variable("v_b1", shape=[300], initializer=tf.contrib.layers.xavier_initializer())
    v_w2 = tf.get_variable("v_w2", shape=(300, last_layer_size), initializer=tf.contrib.layers.xavier_initializer())
    v_b2 = tf.get_variable("v_b2", shape=[last_layer_size], initializer=tf.contrib.layers.xavier_initializer())

    net_u_1 = tf.nn.relu(tf.matmul(one_hot_u, u_w1) + u_b1)
    net_u_2 = tf.matmul(net_u_1, u_w2) + u_b2

    net_v_1 = tf.nn.relu(tf.matmul(one_hot_v, v_w1) + v_b1)
    net_v_2 = tf.matmul(net_v_1, v_w2) + v_b2

    fen_zhi = tf.reduce_sum(net_u_2 * net_v_2, 1, keep_dims=True)

    norm_u = tf.sqrt(tf.reduce_sum(tf.square(net_u_2), 1, keep_dims=True))
    norm_v = tf.sqrt(tf.reduce_sum(tf.square(net_v_2), 1, keep_dims=True))
    fen_mu = norm_u * norm_v

    return tf.nn.relu(fen_zhi / fen_mu), []


class MySampler():
    def __init__(self, all_ratings, u_max_num, v_max_num):
        self.sample_con = {}
        self.sample_con_size = 0

        self.all_ratings_map_u = {}
        for u, v, r, t in all_ratings:
            if u not in self.all_ratings_map_u:
                self.all_ratings_map_u[u] = {}
            self.all_ratings_map_u[u][v] = 1

        self.u_max_num = u_max_num
        self.v_max_num = v_max_num

    def smple_one(self):
        u_rand_num = int(np.random.rand() * self.u_max_num)
        v_rand_num = int(np.random.rand() * self.v_max_num)
        if u_rand_num == 0:
            u_rand_num += 1
        if v_rand_num == 0:
            v_rand_num += 1

        if u_rand_num in self.all_ratings_map_u and v_rand_num not in self.all_ratings_map_u[u_rand_num]:
            return u_rand_num, v_rand_num
        else:
            return self.smple_one()


def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1.0
    return 0.0


def getNDCG(ranklist, gtItem):
    for i in xrange(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i + 2)
    return 0.0


# training
def train_matrix_factorization_With_Feed_Neural():
    top_k = 10
    final_ndcg_metric_list = []
    final_hr_metric_list = []

    my_sample = MySampler(pruned_all_ratings, u_max_num, v_max_num)

    one_hot_u = tf.placeholder(tf.float32, [None, v_max_num])
    one_hot_v = tf.placeholder(tf.float32, [None, u_max_num])
    true_u_v = tf.placeholder(tf.float32, [None, 1])
    pred_val, network_params = inference_neural_DSSM_onehot(one_hot_u, one_hot_v)

    one_constant = tf.constant(1.0, shape=[1, 1])
    gmf_loss = tf.reduce_mean(-true_u_v * tf.log(pred_val + 1e-10) - (one_constant - true_u_v) * tf.log(one_constant - pred_val + 1e-10))
    train_step = tf.train.AdamOptimizer(learn_rate).minimize(gmf_loss)

    batch_u = np.zeros((batch_size + neg_sample_size, v_max_num)).astype('float32')
    batch_v = np.zeros((batch_size + neg_sample_size, u_max_num)).astype('float32')
    batch_true_u_v = np.zeros((batch_size + neg_sample_size, 1)).astype('float32')

    batch_u_test = np.zeros((100, v_max_num)).astype('float32')
    batch_v_test = np.zeros((100, u_max_num)).astype('float32')
    map_index_u = {}
    map_index_v = {}

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    sess = tf.InteractiveSession(config=config)
    tf.initialize_all_variables().run()

    print "DSSM ONE HOT"
    print "batch      size: ", batch_size
    print "neg sample size: ", neg_sample_size
    print "learn      rate: ", learn_rate
    print "file       name: ", file_name
    print "last layer size: ", last_layer_size
    print "2 layers"
    for epoch in range(epochs):

        np.random.shuffle(pruned_all_ratings)
        one_epoch_loss = 0.0
        one_epoch_batchnum = 0.0
        for index in range(len(pruned_all_ratings) / batch_size):
            train_sample_index = 0
            for u_i, v_i, r_i, t_i in pruned_all_ratings[index * batch_size:(index + 1) * batch_size]:
                for v_in in pruned_user_map_item[u_i]:
                    batch_u[train_sample_index][v_in] = pruned_user_map_item[u_i][v_in]
                for u_in in pruned_item_map_user.get(v_i, []):
                    batch_v[train_sample_index][u_in] = pruned_item_map_user[v_i][u_in]
                batch_true_u_v[train_sample_index][0] = 1.0
                map_index_u[train_sample_index] = u_i
                map_index_v[train_sample_index] = v_i
                train_sample_index += 1
            for sam in range(neg_sample_size):
                u_i, v_i = my_sample.smple_one()
                for v_in in pruned_user_map_item[u_i]:
                    batch_u[train_sample_index][v_in] = pruned_user_map_item[u_i][v_in]
                for u_in in pruned_item_map_user.get(v_i, []):
                    batch_v[train_sample_index][u_in] = pruned_item_map_user[v_i][u_in]
                batch_true_u_v[train_sample_index][0] = 0.0
                map_index_u[train_sample_index] = u_i
                map_index_v[train_sample_index] = v_i
                train_sample_index += 1

            _, loss_val, pred_value = sess.run([train_step, gmf_loss, pred_val], feed_dict={one_hot_u: batch_u, one_hot_v: batch_v, true_u_v: batch_true_u_v})
            one_epoch_loss += loss_val
            # print batch_true_u_v
            # print loss_val, pred_value
            one_epoch_batchnum += 1.0

            for j in range(train_sample_index):
                u_key = map_index_u[j]
                v_key = map_index_v[j]
                for v_i in pruned_user_map_item[u_key]:
                    batch_u[j][v_i] = 0.0
                for u_in in pruned_item_map_user.get(v_key, []):
                    batch_v[j][u_in] = 0.0

            if index == len(pruned_all_ratings) / batch_size -1:
                # print "epoch: ", epoch, " end"
                format_str = '%s: %d epoch, iteration averge loss = %.4f '
                print (format_str % (datetime.now(), epoch, one_epoch_loss / one_epoch_batchnum))

                # 计算 NDCG@10 与 HR@10
                # evaluate_1
                # evaluate_2
                hr_list = []
                ndcg_list = []
                for u_i in latest_item_interaction:
                    v_latest = latest_item_interaction[u_i]

                    # print u_i, v_latest
                    v_random = [v_latest]
                    i = 1
                    while i < 100:
                        rand_num = int(np.random.rand() * (v_max_num - 1) + 1)
                        if rand_num not in user_map_item[u_i] and rand_num not in v_random and rand_num in pruned_item_map_user:
                            v_random.append(rand_num)
                            i += 1

                    for i in range(100):
                        for v_in in pruned_user_map_item[u_i]:
                            batch_u_test[i][v_in] = pruned_user_map_item[u_i][v_in]
                        for u_in in pruned_item_map_user.get(v_random[i], []):
                            batch_v_test[i][u_in] = pruned_item_map_user[v_random[i]][u_in]

                    pred_value = sess.run([pred_val], feed_dict={one_hot_u: batch_u_test, one_hot_v: batch_v_test})
                    pre_real_val = np.array(pred_value).reshape((-1))

                    for i in range(100):
                        for v_in in pruned_user_map_item[u_i]:
                            batch_u_test[i][v_in] = 0.0
                        for u_in in pruned_item_map_user.get(v_random[i], []):
                            batch_v_test[i][u_in] = 0.0

                    items = v_random
                    gtItem = items[0]
                    # Get prediction scores
                    map_item_score = {}
                    for i in xrange(len(items)):
                        item = items[i]
                        map_item_score[item] = pre_real_val[i]

                    # Evaluate top rank list
                    # print map_item_score
                    ranklist = heapq.nlargest(top_k, map_item_score, key=map_item_score.get)
                    hr_list.append(getHitRatio(ranklist, gtItem))
                    ndcg_list.append(getNDCG(ranklist, gtItem))
                
                hr_val, ndcg_val = np.array(hr_list).mean(), np.array(ndcg_list).mean()
                final_hr_metric_list.append(hr_val)
                final_ndcg_metric_list.append(ndcg_val)             
                print "RESULT: ", hr_val, ndcg_val
        
    print 'BEST RESULT: ', max(final_hr_metric_list), max(final_ndcg_metric_list)


if __name__ == "__main__":
    train_matrix_factorization_With_Feed_Neural()
