import math
import os
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from time import time
import argparse
import LoadData as DATA
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import logging

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run NFM.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--topk', nargs='?', default=10,
                        help='Topk recommendation list')
    parser.add_argument('--dataset', nargs='?', default='lastfm',
                        help='Choose a dataset.')
    parser.add_argument('--epoch', type=int, default=300,
                        help='Number of epochs.')
    parser.add_argument('--pretrain', type=int, default=1,
                        help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save the model to pretrain file')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors.')
    parser.add_argument('--lamda', type=float, default=0,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--keep_prob', nargs='?', default='[0.8,0.6]',
                        help='Keep probility (1-dropout_ratio) for the Bi-Interaction layer. 1: no dropout')
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
    parser.add_argument('--layers', nargs='?', default='[64]',
                        help="Size of each layer.")

    return parser.parse_args()


class FM(BaseEstimator, TransformerMixin):
    def __init__(self, user_field_M, item_field_M, pretrain_flag, save_file, hidden_factor, layers, epoch, batch_size, learning_rate,
                 lamda_bilinear, keep, optimizer_type, batch_norm, verbose, random_seed=2019):
        # bind params to class
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_factor = hidden_factor
        self.save_file = save_file
        self.pretrain_flag = pretrain_flag
        self.user_field_M=user_field_M
        self.item_field_M=item_field_M
        self.lamda_bilinear = lamda_bilinear
        self.keep = keep
        self.epoch = epoch
        self.random_seed = random_seed
        self.optimizer_type = optimizer_type
        self.batch_norm = batch_norm
        self.verbose = verbose
        self.layers = layers
        self.batch_norm = batch_norm
        self.no_dropout = np.array([1 for i in range(len(keep))])

        # init all variables in a tensorflow graph
        self._init_graph()

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():  # , tf.device('/cpu:0'):
            # Set graph level random seed
            tf.set_random_seed(self.random_seed)
            # Input data.
            self.user_features = tf.placeholder(tf.int32, shape=[None, None])
            self.positive_features= tf.placeholder(tf.int32, shape=[None, None])
            self.negative_features= tf.placeholder(tf.int32, shape=[None, None])
            self.dropout_keep = tf.placeholder(tf.float32)
            self.train_phase = tf.placeholder(tf.bool)

            # Variables.
            self.weights = self._initialize_weights()

            # Model.
            # _________ sum_square part for positive (u,i)_____________
            self.user_feature_embeddings = tf.nn.embedding_lookup(self.weights['user_feature_embeddings'], self.user_features)
            self.positive_feature_embeddings = tf.nn.embedding_lookup(self.weights['item_feature_embeddings'], self.positive_features)
            self.summed_user_emb = tf.reduce_sum(self.user_feature_embeddings, 1)
            self.summed_item_positive_emb = tf.reduce_sum(self.positive_feature_embeddings, 1)
            self.summed_positive_emb=tf.add(self.summed_user_emb,self.summed_item_positive_emb)
            self.summed_positive_emb_square = tf.square(self.summed_positive_emb)

            self.squared_user_emb=tf.square(self.user_feature_embeddings)
            self.squared_item_positiv_emb=tf.square(self.positive_feature_embeddings)
            self.squared_user_emb_sum=tf.reduce_sum(self.squared_user_emb, 1)
            self.squared_item_positive_emb_sum = tf.reduce_sum(self.squared_item_positiv_emb, 1)
            self.squared_positive_emb_sum=tf.add(self.squared_user_emb_sum,self.squared_item_positive_emb_sum)

            # ________ FM part for positive (u,i)__________
            self.FM_positive = 0.5 * tf.subtract(self.summed_positive_emb_square, self.squared_positive_emb_sum)  # None * K
            if self.batch_norm:
                self.FM_positive = self.batch_norm_layer(self.FM_positive, train_phase=self.train_phase, scope_bn='bn_fm')
            self.FM_positive = tf.nn.dropout(self.FM_positive, self.dropout_keep[-1])  # dropout at the FM layer
            for i in range(0, len(self.layers)):
                self.FM_positive = tf.add(tf.matmul(self.FM_positive, self.weights['layer_%d' % i]),
                                 self.weights['bias_%d' % i])  # None * layer[i] * 1
                if self.batch_norm:
                    self.FM_positive = self.batch_norm_layer(self.FM_positive, train_phase=self.train_phase,
                                                    scope_bn='bn_%d' % i)  # None * layer[i] * 1
                self.FM_positive = tf.nn.relu(self.FM_positive)
                self.FM_positive = tf.nn.dropout(self.FM_positive, self.dropout_keep[i])  # dropout at each Deep layer
            self.FM_positive = tf.matmul(self.FM_positive, self.weights['prediction'])  # None * 1
            # _________positive_________
            self.Bilinear_positive = tf.reduce_sum(self.FM_positive, 1, keepdims=True)  # None * 1
            self.user_feature_bias = tf.reduce_sum(tf.nn.embedding_lookup(self.weights['user_feature_bias'], self.user_features),
                                              1)  # None * 1
            self.item_feature_bias_positive = tf.reduce_sum(tf.nn.embedding_lookup(self.weights['item_feature_bias'], self.positive_features),
                1)  # None * 1
            #Bias = self.weights['bias'] * tf.ones_like(self.train_labels)  # None * 1
            self.positive = tf.add_n([self.Bilinear_positive, self.user_feature_bias, self.item_feature_bias_positive])  # None * 1

            # _________ sum_square part for negative (u,j)_____________
            self.negative_feature_embeddings = tf.nn.embedding_lookup(self.weights['item_feature_embeddings'],
                                                                 self.negative_features)
            self.summed_item_negative_emb = tf.reduce_sum(self.negative_feature_embeddings, 1)
            self.summed_negative_emb = tf.add(self.summed_user_emb, self.summed_item_negative_emb)
            self.summed_negative_emb_square = tf.square(self.summed_negative_emb)

            self.squared_item_negative_emb = tf.square(self.negative_feature_embeddings)
            self.squared_item_negative_emb_sum = tf.reduce_sum(self.squared_item_negative_emb, 1)
            self.squared_negative_emb_sum = tf.add(self.squared_user_emb_sum, self.squared_item_negative_emb_sum)

            # ________ FM part for negative (u,j)__________
            self.FM_negative = 0.5 * tf.subtract(self.summed_negative_emb_square, self.squared_negative_emb_sum)  # None * K
            if self.batch_norm:
                self.FM_negative = self.batch_norm_layer(FM, train_phase=self.train_phase, scope_bn='bn_fm')
            self.FM_negative = tf.nn.dropout(self.FM_negative, self.dropout_keep[-1])  # dropout at the FM layer
            for i in range(0, len(self.layers)):
                self.FM_negative = tf.add(tf.matmul(self.FM_negative, self.weights['layer_%d' % i]),
                                 self.weights['bias_%d' % i])  # None * layer[i] * 1
                if self.batch_norm:
                    self.FM_negative = self.batch_norm_layer(self.FM_negative, train_phase=self.train_phase,
                                                    scope_bn='bn_%d' % i)  # None * layer[i] * 1
                self.FM_negative = tf.nn.relu(self.FM_negative)
                self.FM_negative = tf.nn.dropout(self.FM_negative, self.dropout_keep[i])  # dropout at each Deep layer
            self.FM_negative = tf.matmul(self.FM_negative, self.weights['prediction'])  # None * 1
            # _________negative_________
            self.Bilinear_negative = tf.reduce_sum(self.FM_negative,1,keepdims=True)  # None * 1
            self.item_feature_bias_negative = tf.reduce_sum(
                tf.nn.embedding_lookup(self.weights['item_feature_bias'], self.negative_features),
                1)  # None * 1
            self.negative = tf.add_n([self.Bilinear_negative, self.user_feature_bias, self.item_feature_bias_negative])  # None * 1

            # Compute the loss.
            self.loss=-tf.log(tf.sigmoid(self.positive-self.negative))
            self.loss=tf.reduce_sum(self.loss)
            # Optimizer.
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'AdagradOptimizer':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'GradientDescentOptimizer':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'MomentumOptimizer':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)


            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()  # shape is an array of tf.Dimension
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print ("#params: %d" % total_parameters)

    def _initialize_weights(self):
        all_weights = dict()
        if self.pretrain_flag > 0:
            weight_saver = tf.train.import_meta_graph(self.save_file + '.meta')
            pretrain_graph = tf.get_default_graph()
            user_feature_embeddings = pretrain_graph.get_tensor_by_name('user_feature_embeddings:0')
            item_feature_embeddings = pretrain_graph.get_tensor_by_name('item_feature_embeddings:0')
            user_feature_bias = pretrain_graph.get_tensor_by_name('user_feature_bias:0')
            item_feature_bias = pretrain_graph.get_tensor_by_name('item_feature_bias:0')
            #bias = pretrain_graph.get_tensor_by_name('bias:0')
            with tf.Session() as sess:
                weight_saver.restore(sess, self.save_file)
                ue, ie, ub,ib  = sess.run([user_feature_embeddings,item_feature_embeddings,user_feature_bias,item_feature_bias])
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
            #all_weights['bias'] = tf.Variable(tf.constant(0.0), name='bias')  # 1 * 1
        num_layer = len(self.layers)
        if num_layer > 0:
            glorot = np.sqrt(2.0 / (self.hidden_factor + self.layers[0]))
            all_weights['layer_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.hidden_factor, self.layers[0])), dtype=np.float32)
            all_weights['bias_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.layers[0])),
                                                dtype=np.float32)  # 1 * layers[0]
            for i in range(1, num_layer):
                glorot = np.sqrt(2.0 / (self.layers[i - 1] + self.layers[i]))
                all_weights['layer_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.layers[i - 1], self.layers[i])),
                    dtype=np.float32)  # layers[i-1]*layers[i]
                all_weights['bias_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.layers[i])), dtype=np.float32)  # 1 * layer[i]
                # prediction layer
            glorot = np.sqrt(2.0 / (self.layers[-1] + 1))
            all_weights['prediction'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.layers[-1], 1)),
                                                    dtype=np.float32)  # layers[-1] * 1
        else:
            all_weights['prediction'] = tf.Variable(
                np.ones((self.hidden_factor, 1), dtype=np.float32))  # hidden_factor * 1
        return all_weights

    def batch_norm_layer(self, x, train_phase, scope_bn):
        # Note: the decay parameter is tunable
        bn_train = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def partial_fit(self, data):  # fit a batch
        feed_dict = {self.user_features: data['X_user'], self.positive_features: data['X_positive'],
                     self.negative_features: data['X_negative'], self.dropout_keep: self.keep,
                     self.train_phase: True}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def get_random_block_from_data(self, train_data, batch_size):  # generate a random block of training data
        X_user, X_positive, X_negative = [], [], []
        all_items = data.binded_items.values()
        #get sample
        while len(X_user) < batch_size:
            index = np.random.randint(0, len(train_data['X_user']))
            X_user.append(train_data['X_user'][index])
            X_positive.append(train_data['X_item'][index])
            #uniform sampler
            user_features="-".join([str(item) for item in train_data['X_user'][index][0:]])
            user_id=data.binded_users[user_features] # get userID
            pos=data.user_positive_list[user_id]   #get positive list for the userID
            #candidates = list(set(all_items) -set(pos))  #get negative set
            neg = np.random.randint(len(all_items))#uniform sample a negative itemID from negative set
            while(neg in pos):
                neg = np.random.randint(len(all_items))
            negative_feature=data.item_map[neg].strip().split('-') #get negative item feature
            X_negative.append([int(item) for item in negative_feature[0:]])
        return {'X_user': X_user,'X_positive': X_positive,'X_negative': X_negative}


    def train(self, Train_data):  # fit a dataset
        for epoch in range(self.epoch):
            total_loss=0
            total_batch = int(len(Train_data['X_user']) / self.batch_size)
            for i in range(total_batch):
                # generate a batch
                batch_xs = self.get_random_block_from_data(Train_data, self.batch_size)
                # Fit training
                loss=self.partial_fit(batch_xs)
                total_loss= total_loss+loss
            logger.info("the total loss in %d th iteration is: %f" %(epoch, total_loss))
            if (epoch + 1) % 10 == 0:
                model.evaluate()
        # if self.pretrain_flag < 0:
        #     print ("Save model to file as pretrain.")
        #     self.saver.save(self.sess, self.save_file)

    def evaluate(self):
        self.graph.finalize()
        count = [0, 0, 0, 0]
        rank = [[], [], [], []]
        topK = [5, 10, 15, 20]
        for index in range(len(data.Test_data['X_user'])):
            user_features = data.Test_data['X_user'][index]
            item_features = data.Test_data['X_item'][index]
            scores=model.get_scores_per_user(user_features)
            #get true item score
            true_item_id = data.binded_items["-".join([str(item) for item in item_features[0:]])]
            true_item_score = scores[true_item_id]
            #delete visited scores
            user_id = data.binded_users["-".join([str(item) for item in user_features[0:]])]  # get userID
            visited = data.user_positive_list[user_id]  # get positive list for the userID
            scores = np.delete(scores, visited)
            #whether hit
            sorted_scores = sorted(scores, reverse=True)

            label = []
            for i in range(len(topK)):
                label.append(sorted_scores[topK[i] - 1])
                if true_item_score >= label[i]:
                    count[i] = count[i] + 1
                    rank[i].append(sorted_scores.index(true_item_score) + 1)

        for i in range(len(topK)):
            mrr = 0
            ndcg = 0
            hit_rate = float(count[i]) / len(data.Test_data['X_user'])
            for item in rank[i]:
                mrr = mrr + float(1.0) / item
                ndcg = ndcg + float(1.0) / np.log2(item + 1)
            mrr = mrr / len(data.Test_data['X_user'])
            ndcg = ndcg / len(data.Test_data['X_user'])
            k = (i + 1) * 5
            logger.info("top:%f" % k)
            logger.info("the Hit Rate is: %f" % hit_rate)
            logger.info("the MRR is: %f" % mrr)
            logger.info("the NDCG is: %f" % ndcg)


    def get_scores_per_user(self, user_feature):  # evaluate the results for an user context, return scorelist
        #num_example = len(Testdata['Y'])
        # get score list for a userID, store in scorelist, indexed by itemID
        #scorelist=[]
        X_user, X_item= [],[]
        #X_item = []
        #Y=[[1]]
        all_items = data.binded_items.values()
        #true_item_id=data.binded_items[item]
        #user_feature_embeddings = tf.nn.embedding_lookup(self.weights['user_feature_embeddings'],X_user)
        for itemID in range(len(all_items)):
            X_user.append(user_feature)
            item_feature=[int(feature) for feature in data.item_map[itemID].strip().split('-')[0:]]
            X_item.append(item_feature)
        feed_dict = {self.user_features: X_user, self.positive_features: X_item,self.train_phase: False, self.dropout_keep: self.no_dropout}
        scores=self.sess.run((self.positive),feed_dict=feed_dict)
        scores=scores.reshape(len(all_items))
        return scores

if __name__ == '__main__':
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('neuralfm.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # fh.setFormatter(formatter)
    # ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    # Data loading
    args = parse_args()
    data = DATA.LoadData(args.path, args.dataset)
    if args.verbose > 0:
        print( "FM: dataset=%s, factors=%d,  #epoch=%d, batch=%d, lr=%.4f, lambda=%.1e,optimizer=%s, batch_norm=%d, keep=%s"
                % (args.dataset, args.hidden_factor, args.epoch, args.batch_size, args.lr, args.lamda, args.optimizer, args.batch_norm, args.keep_prob))

    # save_file = '../pretrain/%s_%d' % (args.dataset, args.hidden_factor)
    save_file = 'pretrain-FM-%s/%s_%d' % (args.dataset, args.dataset, args.hidden_factor)
    # Training
    t1 = time()
    model = FM(data.user_field_M, data.item_field_M, args.pretrain, save_file, args.hidden_factor, eval(args.layers), args.epoch,
               args.batch_size, args.lr, args.lamda,  eval(args.keep_prob), args.optimizer, args.batch_norm, args.verbose)
    # model.evaluate()
    model.train(data.Train_data)
    # model.evaluate()
