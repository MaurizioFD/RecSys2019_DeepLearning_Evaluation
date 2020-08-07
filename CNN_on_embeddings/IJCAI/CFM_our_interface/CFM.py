'''
Tensorflow Implementation of Convolutional Factorization Machines (CFM) model in:
Xin Xin, Bo Chen et al. CFM: Convolutional Factorization Machines for Context-Aware Recommendation. In IJCAI 2019.

@author:
Xin Xin
Bo Chen
'''
import math
import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
import argparse
import CNN_on_embeddings.IJCAI.CFM_our_interface.LoadData as DATA
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import logging

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run CFM.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='lastfm',
                        help='Choose a dataset. frappe, lastfm, movielens')
    parser.add_argument('--epoch', type=int, default=300,
                        help='Number of epochs.')
    parser.add_argument('--pretrain', type=int, default=1,
                        help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save the model to pretrain file')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--lamda', type=float, default=0,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='Keep probility (1-dropout_ratio). 1: no dropout')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--loss_type', nargs='?', default='log_loss',
                        help='Specify a loss type (square_loss or log_loss).')
    parser.add_argument('--optimizer', nargs='?', default='AdagradOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show the results per X epochs (0, 1 ... any positive integer)')
    parser.add_argument('--batch_norm', type=int, default=0,
                        help='Whether to perform batch normaization (0 or 1)')
    parser.add_argument('--net_channel', nargs='?', default='[32,32,32,32,32,32]',
                        help='net_channel, should be 6 layers here')
    parser.add_argument('--regs', nargs='?', default='[10,1]',
                        help='Regularization for fully-connected weights, CNN filter weights.')
    parser.add_argument('--attention_size', type=int, default=32,
                        help='dimension of attention_size')
    parser.add_argument('--attentive_pooling', type=bool, default=False,
                        help='the flag of attentive_pooling')
    parser.add_argument('--num_field', type=int, default=4,
                        help='number of fields')

    return parser.parse_args()


def _get_interaction_map(relation, map_mode):

    # If using only the diagonal, remove everything not in the diagonal
    if map_mode == "main_diagonal":
        print("CFM: Using main diagonal elements.")
        relation_main_diagonal = tf.zeros_like(relation)
        relation_main_diagonal = tf.linalg.set_diag(relation_main_diagonal, tf.linalg.diag_part(relation))
        relation = relation_main_diagonal

    elif map_mode == "off_diagonal":
        print("CFM: Using off diagonal elements.")
        relation = tf.linalg.set_diag(relation, tf.zeros_like(tf.linalg.diag_part(relation)))

    else:
        print("CFM: Using all map elements.")

    return relation




class CFM(BaseEstimator, TransformerMixin):
    def __init__(self, data, user_field_M, item_field_M, pretrain_flag, pretrained_FM_folder_path, hidden_factor, num_field,
                 batch_size, learning_rate,
                 lamda_bilinear, keep, optimizer_type, batch_norm, verbose, regs, attention_size, attentive_pooling,
                 net_channel, map_mode, permutation:list=None, random_seed=None):
        # bind params to class
        self.data = data
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_factor = hidden_factor
        self.pretrained_FM_folder_path = pretrained_FM_folder_path
        self.pretrain_flag = pretrain_flag
        self.user_field_M = user_field_M
        self.item_field_M = item_field_M
        self.lamda_bilinear = lamda_bilinear
        self.keep = keep
        # self.epoch = epoch
        self.random_seed = random_seed
        self.optimizer_type = optimizer_type
        self.batch_norm = batch_norm
        self.verbose = verbose
        self.attention_size = attention_size
        self.num_field = num_field
        self.attentive_pooling = attentive_pooling
        self.net_channel = net_channel

        self.regs = regs
        reg = eval(regs)
        self.gamma_bilinear = reg[0]
        self.lambda_weight = reg[1]

        # the type of experiment to do with the interaction cube
        self.map_mode = map_mode

        # permutation array
        self.permutation = permutation
        # constuct a matrix to permutate the embeddings
        if self.permutation is not None:
            # check if the permutation array is correct
            assert len(self.permutation) == hidden_factor,\
                'Invalid permutation array length, it must be equal to the hidden factor size ({}), {} passed instead'.format(
                    hidden_factor, len(self.permutation))
            setdiff = np.setdiff1d(np.arange(hidden_factor), permutation)
            assert setdiff.shape[0] == 0,\
                'Invalid permutation array, not all indices are present, {} are missing'.format(str(setdiff))
        self._perm_mx = np.zeros((self.hidden_factor, self.hidden_factor), dtype=np.uint8)
        self._perm_mx[self.permutation, np.arange(self._perm_mx.shape[0])] = 1

        # init all variables in a tensorflow graph
        self._init_graph()

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Set graph level random seed
            if self.random_seed is not None:
                tf.set_random_seed(self.random_seed)

            # Input data
            self.user_features = tf.placeholder(tf.int32, shape=[None, None])
            self.positive_features = tf.placeholder(tf.int32, shape=[None, None])
            self.negative_features = tf.placeholder(tf.int32, shape=[None, None])
            self.dropout_keep = tf.placeholder(tf.float32)
            self.train_phase = tf.placeholder(tf.bool)

            self.permutation_matrix = tf.placeholder(tf.float32, shape=[None, None])

            # Variables.
            self.weights = self._initialize_weights()       # contains user and item embedding matrices
            self.nc = eval(self.net_channel)
            iszs = [1] + self.nc[:-1]
            oszs = self.nc
            self.P = []
            for i in range(5):
                self.P.append(self._conv_weight(2, iszs[i], oszs[i]))  # first 5 layers
            self.P.append(self._conv_weight(1, iszs[5], oszs[5]))
            self.W = self.weight_variable([self.nc[-1], 1])  # last layer
            self.b = self.weight_variable([1])

            ##################### CFM Model. #####################
            # Embedding Layer
            self.user_feature_embeddings = tf.nn.embedding_lookup(self.weights['user_feature_embeddings'],
                                                                  self.user_features)

            self.positive_item_embeddings = tf.nn.embedding_lookup(self.weights['item_feature_embeddings'],
                                                                   self.positive_features)

            # Positive part
            # Self-Attention Pooling Layer - for field with multi-hot feature
            if self.attentive_pooling:
                self.user_feature_embeddings_uid = tf.expand_dims(self.user_feature_embeddings[:, 0, :], 1)
                self.user_feature_embeddings_history = self.user_feature_embeddings[:, 1:,
                                                       :]  # history items of user (multi-hot)

                self.attention_sum = tf.tanh(tf.layers.dense(self.user_feature_embeddings_history, self.attention_size,
                                                             name='w_item',
                                                             reuse=tf.AUTO_REUSE, use_bias=False))
                self.attention_scalar = tf.layers.dense(self.attention_sum, 1, name='attention_weight',
                                                        reuse=tf.AUTO_REUSE,
                                                        use_bias=False)
                self.attention = tf.expand_dims(tf.nn.softmax(tf.squeeze(self.attention_scalar, [2])), 1)
                self.user_item_latent = tf.matmul(self.attention, self.user_feature_embeddings_history)
                self.user_feature_embeddings = tf.concat([self.user_feature_embeddings_uid, self.user_item_latent], 1)

            # Interaction Cube
            self.positive_embeddings = tf.concat(
                [self.user_feature_embeddings, self.positive_item_embeddings], 1)

            self.split = tf.split(
                axis=1, num_or_size_splits=self.num_field, value=self.positive_embeddings)  # split as field
            # build interaction cube
            for i in range(0, self.num_field):
                for j in range(i + 1, self.num_field):
                    content = 'self.split[' + str(i) + ']'
                    self.x = eval(content)
                    content1 = 'self.split[' + str(j) + ']'
                    self.y = eval(content1)

                    # permutate both embeddings before to be outer multiplied
                    if self.permutation is not None:
                        self.x = tf.multiply(self.permutation_matrix, self.x)
                        self.y = tf.multiply(self.permutation_matrix, self.y)

                    self.relation = tf.matmul(tf.transpose(self.x, perm=[0, 2, 1]), self.y)

                    self.relation = _get_interaction_map(self.relation, self.map_mode)

                    self.net_input = tf.expand_dims(self.relation, 1)
                    if i == 0 and j == 1:
                        self.cube = self.net_input
                    else:
                        self.cube = tf.concat([self.cube, self.net_input], 1)
            self.positive_cube = tf.expand_dims(self.cube, -1)

            # 3D Convolution Layers
            self.layer = []
            positive_input = self.positive_cube
            for p in self.P:
                # convolution
                self.layer.append(self._conv_layer(positive_input, p))
                positive_input = self.layer[-1]
            self.dropout_positive = tf.nn.dropout(self.layer[-1], self.dropout_keep)
            self.interaction_positive = tf.matmul(tf.reshape(self.dropout_positive, [-1, self.nc[-1]]), self.W) + self.b
            self.user_feature_bias = tf.reduce_sum(
                tf.nn.embedding_lookup(self.weights['user_feature_bias'], self.user_features), 1)  # None * 1
            self.item_feature_bias_positive = tf.reduce_sum(
                tf.nn.embedding_lookup(self.weights['item_feature_bias'], self.positive_features), 1)  # None * 1
            self.positive = tf.add_n(
                [self.interaction_positive, self.user_feature_bias, self.item_feature_bias_positive])  # None * 1

            # Negative part
            self.negative_item_embeddings = tf.nn.embedding_lookup(self.weights['item_feature_embeddings'],
                                                                   self.negative_features)
            self.negative_embeddings = tf.concat(
                [self.user_feature_embeddings, self.negative_item_embeddings], 1)

            self.split = tf.split(
                axis=1, num_or_size_splits=self.num_field, value=self.negative_embeddings)
            for i in range(0, self.num_field):
                for j in range(i + 1, self.num_field):
                    content = 'self.split[' + str(i) + ']'
                    self.x = eval(content)
                    content1 = 'self.split[' + str(j) + ']'
                    self.y = eval(content1)

                    # permutate both embeddings before to be outer multiplied
                    if self.permutation is not None:
                        self.x = tf.multiply(self.permutation_matrix, self.x)
                        self.y = tf.multiply(self.permutation_matrix, self.y)

                    self.relation = tf.matmul(tf.transpose(self.x, perm=[0, 2, 1]), self.y)

                    self.relation = _get_interaction_map(self.relation, self.map_mode)

                    self.net_input = tf.expand_dims(self.relation, 1)
                    if i == 0 and j == 1:
                        self.cube = self.net_input
                    else:
                        self.cube = tf.concat([self.cube, self.net_input], 1)
            self.negative_cube = tf.expand_dims(self.cube, -1)

            self.layer = []
            negative_input = self.negative_cube
            for p in self.P:
                self.layer.append(self._conv_layer(negative_input, p))
                negative_input = self.layer[-1]
            self.dropout_negative = tf.nn.dropout(self.layer[-1], self.dropout_keep)
            self.interaction_negative = tf.matmul(tf.reshape(self.dropout_negative, [-1, self.nc[-1]]), self.W) + self.b
            self.item_feature_bias_negative = tf.reduce_sum(
                tf.nn.embedding_lookup(self.weights['item_feature_bias'], self.negative_features), 1)  # None * 1
            self.negative = tf.add_n(
                [self.interaction_negative, self.user_feature_bias, self.item_feature_bias_negative])  # None * 1

            # Compute the loss.
            self.loss = -tf.log(tf.sigmoid(self.positive - self.negative))
            self.loss = tf.reduce_sum(self.loss)
            # self.loss = self.loss + self.gamma_bilinear * self._regular([(self.W, self.b)]) \
            #                 + self.lambda_weight * (self._regular(self.P) + self._regular([(self.W, self.b)]))

            # Optimizer.
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'AdagradOptimizer':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'GradientDescentOptimizer':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(
                    self.loss)
            elif self.optimizer_type == 'MomentumOptimizer':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def _regular(self, params):
        res = 0
        for param in params:
            res += tf.reduce_sum(tf.square(param[0])) + tf.reduce_sum(tf.square(param[1]))
        return res

    def _conv_weight(self, deep, isz, osz):
        return (self.weight_variable([deep, 2, 2, isz, osz]), self.bias_variable([osz]))

    def _conv_layer(self, input, P):
        '''
        Convolution layer of 3D CNN
        :param input:
        :param P: weights and bias
        :return: convolution result
        '''
        conv = tf.nn.conv3d(input, P[0], strides=[1, 1, 2, 2, 1],
                            padding='VALID')
        return tf.nn.relu(conv + P[1])  # bias_add and activate

    def _initialize_weights(self):
        '''
        Weights of embeddings
        '''
        all_weights = dict()
        if self.pretrain_flag > 0:
            weight_saver = tf.train.import_meta_graph(self.pretrained_FM_folder_path + '.meta')
            pretrain_graph = tf.get_default_graph()
            user_feature_embeddings = pretrain_graph.get_tensor_by_name('user_feature_embeddings:0')
            item_feature_embeddings = pretrain_graph.get_tensor_by_name('item_feature_embeddings:0')
            user_feature_bias = pretrain_graph.get_tensor_by_name('user_feature_bias:0')
            item_feature_bias = pretrain_graph.get_tensor_by_name('item_feature_bias:0')
            with tf.Session() as sess:
                weight_saver.restore(sess, self.pretrained_FM_folder_path)
                ue, ie, ub, ib = sess.run(
                    [user_feature_embeddings, item_feature_embeddings, user_feature_bias, item_feature_bias])
            all_weights['user_feature_embeddings'] = tf.Variable(ue, dtype=tf.float32)
            all_weights['item_feature_embeddings'] = tf.Variable(ie, dtype=tf.float32)
            all_weights['user_feature_bias'] = tf.Variable(ub, dtype=tf.float32)
            all_weights['item_feature_bias'] = tf.Variable(ib, dtype=tf.float32)
        else:
            all_weights['user_feature_embeddings'] = tf.Variable(
                tf.random_normal([self.user_field_M, self.hidden_factor], 0.0, 0.1),
                name='user_feature_embeddings')  # user_field_M * K
            all_weights['item_feature_embeddings'] = tf.Variable(
                tf.random_normal([self.item_field_M, self.hidden_factor], 0.0, 0.1),
                name='item_feature_embeddings')  # item_field_M * K
            all_weights['user_feature_bias'] = tf.Variable(
                tf.random_uniform([self.user_field_M, 1], 0.0, 0.1), name='user_feature_bias')  # user_field_M * 1
            all_weights['item_feature_bias'] = tf.Variable(
                tf.random_uniform([self.item_field_M, 1], 0.0, 0.1), name='item_feature_bias')  # item_field_M * 1
        return all_weights

    def batch_norm_layer(self, x, train_phase, scope_bn):
        # Note: the decay parameter is tunable
        bn_train = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def partial_fit(self, data):
        '''
        Fit a batch and obtain batch loss
        :param data: batch data
        :return: loss and opt
        '''
        feed_dict = {self.user_features: data['X_user'],
                     self.positive_features: data['X_positive'],
                     self.negative_features: data['X_negative'],
                     self.permutation_matrix: self._perm_mx,    # set the permutation matrix
                     self.dropout_keep: self.keep,
                     self.train_phase: True}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def get_random_block_from_data(self, train_data, batch_size):
        '''
        Generate a random block of training data
        :param train_data: train dataset
        :param batch_size: batch size
        :return: a batch data
        '''
        X_user, X_positive, X_negative = [], [], []
        all_items = self.data.binded_items.values()
        # get sample
        while len(X_user) < batch_size:
            index = np.random.randint(0, len(train_data['X_user']))
            X_user.append(train_data['X_user'][index])
            X_positive.append(train_data['X_item'][index])

            # uniform sampler
            user_features = "-".join([str(item) for item in train_data['X_user'][index][0:]])
            user_id = self.data.binded_users[user_features]  # get userID
            pos = self.data.user_positive_list[user_id]  # get positive list for the userID
            neg = np.random.randint(len(all_items))  # uniform sample a negative itemID from negative set
            while (neg in pos):
                neg = np.random.randint(len(all_items))
            negative_feature = self.data.item_map[neg].strip().split('-')  # get negative item feature
            X_negative.append([int(item) for item in negative_feature[0:]])
        return {'X_user': X_user, 'X_positive': X_positive, 'X_negative': X_negative}

    def _run_epoch(self, Train_data):
        total_loss = 0
        total_batch = int(len(Train_data['X_user']) / self.batch_size)

        total_batch_iterator = tqdm(range(total_batch)) if self.verbose else range(total_batch)

        for i in total_batch_iterator:
            # generate a batch
            batch_xs = self.get_random_block_from_data(Train_data, self.batch_size)
            # Fit training
            loss = self.partial_fit(batch_xs)
            total_loss += loss
        return total_loss
    #
    # def train(self, Train_data):
    #     '''
    #     Model training
    #     :param Train_data: train dataset
    #     :return:
    #     '''
    #     for epoch in range(self.epoch):
    #         total_loss = self._run_epoch(Train_data)
    #         logger.info("the total loss in %d th iteration is: %f" % (epoch, total_loss))
    #
    #         # if (epoch + 1) % 10 == 0:
    #         #     model.evaluate()
    #     if self.pretrain_flag < 0:
    #         print("Save model to file as pretrain.")
    #         print('Save succesfully in {}'.format(self.save_session()))

    # def evaluate(self):
    #     '''
    #     Model evaluation
    #     :return:
    #     '''
    #     self.graph.finalize()
    #     count = [0, 0, 0, 0]
    #     rank = [[], [], [], []]
    #     topK = [5, 10, 15, 20]
    #     for index in tqdm(range(len(data.Test_data['X_user']))):
    #         user_features = data.Test_data['X_user'][index]
    #         item_features = data.Test_data['X_item'][index]
    #         scores = model.get_scores_per_user(user_features)
    #         # get true item score
    #         true_item_id = data.binded_items["-".join([str(item) for item in item_features[0:]])]
    #         true_item_score = scores[true_item_id]
    #         # delete visited scores
    #         user_id = data.binded_users["-".join([str(item) for item in user_features[0:]])]  # get userID
    #         visited = data.user_positive_list[user_id]  # get positive list for the userID
    #         scores = np.delete(scores, visited)
    #         # whether hit
    #         sorted_scores = sorted(scores, reverse=True)
    #
    #         label = []
    #         for i in range(len(topK)):
    #             label.append(sorted_scores[topK[i] - 1])
    #             if true_item_score >= label[i]:
    #                 count[i] = count[i] + 1
    #                 rank[i].append(sorted_scores.index(true_item_score) + 1)
    #
    #     for i in range(len(topK)):
    #         mrr = 0
    #         ndcg = 0
    #         hit_rate = float(count[i]) / len(data.Test_data['X_user'])
    #         for item in rank[i]:
    #             mrr = mrr + float(1.0) / item
    #             ndcg = ndcg + float(1.0) / np.log2(item + 1)
    #         mrr = mrr / len(data.Test_data['X_user'])
    #         ndcg = ndcg / len(data.Test_data['X_user'])
    #         k = (i + 1) * 5
    #         logger.info("top:%f" % k)
    #         logger.info("the Hit Rate is: %f" % hit_rate)
    #         logger.info("the MRR is: %f" % mrr)
    #         logger.info("the NDCG is: %f" % ndcg)
    #
    # def get_scores_per_user_original(self, user_feature):
    #     '''
    #     Evaluate the results for an user context
    #     :param user_feature: user feature vector
    #     :return: scorelist of a user
    #     '''
    #     scorelist = []
    #
    #     all_items = data.binded_items.values()
    #
    #     if len(all_items) % self.batch_size == 0:
    #         batch_count = len(all_items) / self.batch_size
    #         flag = 0
    #     else:
    #         batch_count = math.ceil(len(all_items) / self.batch_size)
    #         flag = 1
    #     j = 0
    #     for i in range(int(batch_count)):
    #         X_user, X_item = [], []
    #         if flag == 1 and i == batch_count - 1:
    #             k = len(all_items)
    #         else:
    #             k = j + self.batch_size
    #         for itemID in range(j, k):
    #             X_user.append(user_feature)
    #             item_feature = [int(feature) for feature in data.item_map[itemID].strip().split('-')[0:]]
    #             X_item.append(item_feature)
    #
    #         feed_dict = {self.user_features: X_user,
    #                      self.positive_features: X_item,
    #                      self.train_phase: False,
    #                      self.dropout_keep: 1.0}
    #         scores = self.sess.run((self.positive), feed_dict=feed_dict)
    #         scores = scores.reshape(len(X_user))
    #         scorelist = np.append(scorelist, scores)
    #         j = j + self.batch_size
    #     return scorelist


    def get_scores_per_user(self, user_feature):
        '''
        Evaluate the results for an user context
        :param user_feature: user feature vector
        :return: scorelist of a user
        '''
        scorelist = []
        all_items_count = self.data.item_bind_M

        if all_items_count % self.batch_size == 0:
            batch_count = int(all_items_count / self.batch_size)
            flag = 0
        else:
            batch_count = int(math.ceil(all_items_count / self.batch_size))
            flag = 1

        X_user_batch = [user_feature] * self.batch_size

        j = 0
        for i in range(batch_count):
            if flag == 1 and i == batch_count - 1:
                k = all_items_count
                X_user_batch = [user_feature] * (k - j)
            else:
                k = j + self.batch_size

            X_item_batch = self.data.all_items_features_list[j:k]

            feed_dict = {self.user_features: X_user_batch,
                         self.positive_features: X_item_batch,
                         self.permutation_matrix: self._perm_mx,    # set the permutation matrix
                         self.train_phase: False,
                         self.dropout_keep: 1.0}
            scores = self.sess.run((self.positive), feed_dict=feed_dict)
            scores = scores.reshape(len(X_user_batch))
            scorelist = np.append(scorelist, scores)

            j += self.batch_size
        return scorelist


    def save_session(self, folder_path):
        self.saver.save(self.sess, folder_path)
        return

    def restore_session(self, folder_path):
        self.saver.restore(self.sess, folder_path)

if __name__ == '__main__':
    # Logging
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('../../../result_experiments/cfm.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)

    # Data loading
    args = parse_args()
    data = DATA.LoadData("../CFM_github/" + args.path, args.dataset)

    if args.verbose > 0:
        print(
            "FM: dataset=%s, factors=%d,  #epoch=%d, batch=%d, lr=%.4f, lambda=%.1e,optimizer=%s, batch_norm=%d, keep=%.2f"
            % (args.dataset, args.hidden_factor, args.epoch, args.batch_size, args.lr, args.lamda, args.optimizer,
               args.batch_norm, args.keep_prob))

    save_file = 'pretrain-FM-%s/%s_%d' % (args.dataset, args.dataset, args.hidden_factor)

    # CFM model
    model = CFM(data.user_field_M, data.item_field_M, args.pretrain, save_file, args.hidden_factor, args.num_field,
                args.epoch, args.batch_size, args.lr, args.lamda, args.keep_prob, args.optimizer, args.batch_norm,
                args.verbose, args.regs, args.attention_size, args.attentive_pooling)
    # Training
    model.train(data.Train_data)
