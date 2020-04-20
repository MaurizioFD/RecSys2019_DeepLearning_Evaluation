from __future__ import absolute_import
from __future__ import division
import os
import math
import numpy as np
import tensorflow as tf
from multiprocessing import Pool
from multiprocessing import cpu_count
import argparse
import logging
from time import time
from time import strftime
from time import localtime
import scipy.sparse as sps
from tqdm import tqdm

# use conv instead of pooling

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

_user_input = None
_item_input_pos = None
_batch_size = None
_index = None
_model = None
_sess = None
_dataset = None
_K = None
_feed_dict = None
_output = None

_eval_feed_dicts = None
_samples = None
_max_ndcg = None
_max_res = None



TRAIN_KEEP_PROB = 1
TEST_KEEP_PROB = 1


#---------- data preparation -------

# data sampling and shuffling

# input: dataset(Mat, List, Rating, Negatives), batch_choice, num_negatives
# output: [_user_input_list, _item_input_pos_list]
def sampling(dataset):
    _user_input, _item_input_pos = [], []
    for (u, i) in dataset.trainMatrix.keys():
        # positive instance
        _user_input.append(u)
        _item_input_pos.append(i)
    return _user_input, _item_input_pos

def shuffle(samples, batch_size, dataset, model):
    global _user_input
    global _item_input_pos
    global _batch_size
    global _index
    global _model
    global _dataset
    _user_input, _item_input_pos = samples
    _batch_size = batch_size
    _index = np.arange(len(_user_input))
    _model = model
    _dataset = dataset
    np.random.shuffle(_index)
    num_batch = len(_user_input) // _batch_size
    pool = Pool(cpu_count())
    res = pool.map(_get_train_batch, range(num_batch))
    pool.close()
    pool.join()
    user_list = [r[0] for r in res]
    item_pos_list = [r[1] for r in res]
    user_dns_list = [r[2] for r in res]
    item_dns_list = [r[3] for r in res]
    return user_list, item_pos_list, user_dns_list, item_dns_list

def _get_train_batch(i):
    global _dataset
    user_batch, item_batch = [], []
    user_neg_batch, item_neg_batch = [], []
    begin = i * _batch_size
    for idx in range(begin, begin + _batch_size):
        user_batch.append(_user_input[_index[idx]])
        item_batch.append(_item_input_pos[_index[idx]])
        for dns in range(_model.dns):
            user = _user_input[_index[idx]]
            user_neg_batch.append(user)
            # negtive k
            # add this try except because not all users could have a test item, so in this case the while condition doesnt't fail (Simone Boglio)
            # try:
            #     gtItem = _dataset.testRatings[user][1]
            # except IndexError:
            #     gtItem = -1
            j = np.random.randint(_dataset.num_items)
            # the condition "j == gtItem" could lead to a potential overfitting with large number of epochs (Simone Boglio)
            while j in _dataset.trainList[_user_input[_index[idx]]]:# or j == gtItem:
                j = np.random.randint(_dataset.num_items)
            item_neg_batch.append(j)
    return np.array(user_batch)[:,None], np.array(item_batch)[:,None], \
           np.array(user_neg_batch)[:,None], np.array(item_neg_batch)[:,None]

#---------- model definition -------

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# prediction model
class ConvNCF:
    def __init__(self, num_users, num_items, args):
        self.num_items = num_items
        self.num_users = num_users
        self.embedding_size = args.embed_size
        self.lr_embed = args.lr_embed
        self.lr_net = args.lr_net
        self.hidden_size = args.hidden_size
        self.nc = args.net_channel
        regs = args.regs
        self.lambda_bilinear = regs[0]
        self.gamma_bilinear = regs[1]
        self.lambda_weight = regs[2]
        self.dns = args.dns
        self.train_auc = args.train_auc
        self.prepared = False

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape = [None, 1], name = "user_input")
            self.item_input_pos = tf.placeholder(tf.int32, shape = [None, 1], name = "item_input_pos")
            self.item_input_neg = tf.placeholder(tf.int32, shape = [None, 1], name = "item_input_neg")
            self.keep_prob = tf.placeholder(tf.float32, name = "keep_prob")

    def _conv_weight(self, isz, osz):
        return (weight_variable([2,2,isz,osz]), bias_variable([osz]))

    def _conv_layer(self, input, P):
        conv = tf.nn.conv2d(input, P[0], strides=[1, 2, 2, 1], padding='SAME')
        return tf.nn.relu(conv + P[1])

    def _create_variables(self):
        with tf.name_scope("embedding"):
            self.embedding_P = tf.Variable(tf.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0, stddev=0.01),
                                                                name='embedding_P', dtype=tf.float32)  #(users, embedding_size)
            self.embedding_Q = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
                                                                name='embedding_Q', dtype=tf.float32)  #(items, embedding_size)

            # here should have 6 iszs due to the size of outer products is 64x64
            iszs = [1] + self.nc[:-1]
            oszs = self.nc
            self.P = []
            for isz, osz in zip(iszs, oszs):
                self.P.append(self._conv_weight(isz, osz))

            self.W = weight_variable([self.nc[-1], 1])
            self.b = weight_variable([1])

    def _create_inference(self, item_input):
        with tf.name_scope("inference"):
            # embedding look up
            self.embedding_p = tf.nn.embedding_lookup(self.embedding_P, self.user_input)
            self.embedding_q = tf.nn.embedding_lookup(self.embedding_Q, item_input)

            # outer product of P_u and Q_i
            self.relation = tf.matmul(tf.transpose(self.embedding_p, perm=[0, 2, 1]), self.embedding_q)
            self.net_input = tf.expand_dims(self.relation, -1)

            # CNN
            self.layer = []
            input = self.net_input
            for p in self.P:
                self.layer.append(self._conv_layer(input, p))
                input = self.layer[-1]

            # prediction
            self.dropout = tf.nn.dropout(self.layer[-1], self.keep_prob)
            self.output_layer = tf.matmul(tf.reshape(self.dropout,[-1,self.nc[-1]]), self.W) + self.b

            return self.embedding_p, self.embedding_q, self.output_layer


    def _regular(self, params):
        res = 0
        for param in params:
            res += tf.reduce_sum(tf.square(param[0])) + tf.reduce_sum(tf.square(param[1]))
        return res

    def _create_loss(self):
        with tf.name_scope("loss"):
            # BPR loss for L(Theta)
            self.p1, self.q1, self.output = self._create_inference(self.item_input_pos)
            self.p2, self.q2, self.output_neg = self._create_inference(self.item_input_neg)
            self.result = self.output - self.output_neg
            self.loss = tf.reduce_sum(tf.log(1 + tf.exp(-self.result)))

            self.opt_loss = self.loss + self.lambda_bilinear * ( tf.reduce_sum(tf.square(self.p1)) \
                                    + tf.reduce_sum(tf.square(self.q2)) + tf.reduce_sum(tf.square(self.q1)))\
                                    + self.gamma_bilinear * self._regular([(self.W, self.b)]) \
                                    + self.lambda_weight * (self._regular(self.P) + self._regular([(self.W, self.b)]))

    # used at the first time when emgeddings are pretrained yet network are randomly initialized
    # if not, the parameters may be NaN.
    def _create_pre_optimizer(self):
        self.pre_opt = tf.train.AdagradOptimizer(learning_rate=0.01).minimize(self.loss)

    def _create_optimizer(self):
        # seperated optimizer
        var_list1 = [self.embedding_P, self.embedding_Q]
        #[self.W1,self.W2,self.W3,self.W4,self.b1,self.b2,self.b3,self.b4,self.P1,self.P2,self.P3]
        var_list2 = list(set(tf.trainable_variables()) - set(var_list1))
        opt1 = tf.train.AdagradOptimizer(self.lr_embed)
        opt2 = tf.train.AdagradOptimizer(self.lr_net)
        grads = tf.gradients(self.opt_loss, var_list1 + var_list2)
        grads1 = grads[:len(var_list1)]
        grads2 = grads[len(var_list1):]
        train_op1 = opt1.apply_gradients(zip(grads1, var_list1))
        train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
        self.optimizer = tf.group(train_op1, train_op2)


    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_loss()
        self._create_pre_optimizer()
        self._create_optimizer()

    def load_parameter_MF(self, sess, path):
        ps = np.load(path)
        assert ps is not None, 'ConvNCF: Pre-trained net-weights from MF_BPR LOAD FAIL'
        ap = tf.assign(self.embedding_P, ps[0])
        aq = tf.assign(self.embedding_Q, ps[1])
        #ah = tf.assign(self.h, np.diag(ps[2][:,0]).reshape(4096,1))
        sess.run([ap,aq])
        print("ConvNCF: Pre-trained net-weights from MF_BPR loaded successfully...")

    def load_parameter_logloss(self, sess, path):
        ps = np.load(path).tolist()
        ap = tf.assign(self.embedding_P, ps['P'])
        aq = tf.assign(self.embedding_Q, ps['Q'])
        sess.run([ap,aq])
        print("logloss parameter loaded")

    def save_net_parameters(self, sess, path):
        pass

    def get_optimizer(self):
        if self.prepared:  # common optimize
            return self.optimizer
        else:
            # do a previous optimizing with none regularizations if it is the first time to optimize the neural network.
            # if not, the parameters may be NaN.
            return self.pre_opt

def get_session():
    return _sess


def dxyeval(sess, model, dataset):
    eval_feed_dicts = init_eval_model(model, dataset)
    hr, ndcg, auc, train_auc = evaluate(model, sess, dataset, eval_feed_dicts)
    res = "Pretrain: HR = %.4f, NDCG = %.4f AUC = %.4f train_AUC = %.4f" % (hr, ndcg, auc, train_auc)
    print (res)

#---------- training -------

# training
def initialize(model, dataset, args, saver = None): # saver is an object to save pq
    global _sess
    global _ckpt_save_path, _ckpt_save_file ,_saver_ckpt
    _sess= tf.Session()
    # initialized the save op
    #_ckpt_save_path = "Pretrain/dropout/%s_embed_%s/" %  (args.dataset, "_".join(map(str,model.nc)))
    #_ckpt_save_file = str(TRAIN_KEEP_PROB) + '_'.join(map(str, args.regs))
    #if not os.path.exists(_ckpt_save_path):
    #    os.makedirs(_ckpt_save_path)
    # initialize saver
    #_saver_ckpt = tf.train.Saver(tf.trainable_variables())

    # pretrain or not
    _sess.run(tf.global_variables_initializer())

    # restore the weights when pretrained
    if args.pretrain:
        path = args.path_partial_results
        #saver_ckpt.restore(sess, "Pretrain/MF_BPR/embed_32_32_32_32_32_32/1e-06_0_10-1440")
        model.load_parameter_MF(_sess, path + "best_model_latent_factors.npy")
        #dxyeval(_sess,model,dataset) # this is an evaluation


    global _eval_feed_dicts
    global _samples
    global _max_ndcg
    global _max_res

    # initialize for Evaluate
    #####_eval_feed_dicts = init_eval_model(model, dataset)

    # sample the data
    _samples = sampling(dataset)

    #initialize the max_ndcg to memorize the best result
    _max_ndcg = 0
    _max_res = " "


def run_epoch(model, dataset, epoch_count, args, original_evaluation=False, verbose=True):
    global _sess
    global _max_ndcg
    global _max_res
    global _ckpt_save_path, _ckpt_save_file, _saver_ckpt

    if verbose: print("Start epoch: {}".format(epoch_count))
    # initialize for training batches
    batch_begin = time()
    batches = shuffle(_samples, args.batch_size, dataset, model)
    batch_time = time() - batch_begin

    # compute the accuracy before training
    _, prev_acc = 0,0 #training_loss_acc(model, sess, prev_batch)

    # training the model
    train_begin = time()
    train_batches = training_batch(model,_sess, batches)
    train_time = time() - train_begin

    if verbose: print("batch cost: {} train cost: {}".format(batch_time,train_time))

    if not original_evaluation:
        return batch_time, train_time

    if  original_evaluation and epoch_count % args.verbose == 0:
        print("Epoch {}: start evaluation inside model..".format(epoch_count))
        _, ndcg, cur_res = output_evaluate(model, _sess, dataset, train_batches, _eval_feed_dicts,
                        epoch_count, batch_time, train_time, prev_acc)

        # print and log the best result
        if _max_ndcg < ndcg:
            _max_ndcg = ndcg
            _max_res = cur_res
            #_saver_ckpt.save(_sess, _ckpt_save_path+_ckpt_save_file, global_step=epoch_count)
            print("saved best: epoch {}".format(epoch_count))
        print("Epoch {}: end evaluation inside model..".format(epoch_count))
        return batch_time, train_time


def compute_score_all_users(ids_user, model, dataset, verbose=True):
    feed_dicts = init_eval_model(model, dataset)
    users = []
    items = []
    predictions = []
    for u in tqdm(ids_user, disable=not verbose ,desc="ConvNCF compute ratings:"):
        user_input, item_input = feed_dicts[u]
        feed_dict = {_model.user_input: user_input, _model.item_input_pos: item_input, _model.keep_prob: TEST_KEEP_PROB}
        prediction = _sess.run(_model.output, feed_dict)
        predictions += prediction.flatten().tolist()
        users += user_input.flatten().tolist()
        items += item_input.flatten().tolist()
    eurm = sps.csr_matrix((predictions,(users,items)), shape=(dataset.num_users,dataset.num_items))
    return eurm

def compute_score_single_user(id_user, ids_item):
    item_input = np.array(ids_item)
    user_input = np.full(item_input.shape[0], id_user)
    item_input = item_input.reshape((item_input.shape[0],1))
    user_input = user_input.reshape((item_input.shape[0],1))
    feed_dict = {_model.user_input: user_input, _model.item_input_pos: item_input, _model.keep_prob: TEST_KEEP_PROB}
    predictions = _sess.run(_model.output, feed_dict).flatten()
    return predictions




def close_session(verbose=False):
    global _sess
    global _max_res
    if verbose and _max_res!=" ": print("best: {}".format(_max_res))
    if _max_res!=" ":logging.info("best:" + _max_res)
    _sess.close()

def output_evaluate(model, sess, dataset, train_batches, eval_feed_dicts, epoch_count, batch_time, train_time, prev_acc):
    loss_begin = time()
    train_loss, post_acc = training_loss_acc(model, sess, train_batches)
    loss_time = time() - loss_begin

    eval_begin = time()
    hr, ndcg, auc, train_auc = evaluate(model, sess, dataset, eval_feed_dicts)
    eval_time = time() - eval_begin

    res = "Epoch %d [%.1fs + %.1fs]: HR = %.4f, NDCG = %.4f AUC = %.4f train_AUC = %.4f [%.1fs]" \
        " ACC = %.4f train_loss = %.4f ACC = %.4f [%.1fs]" % ( epoch_count, batch_time, train_time,
                    hr, ndcg, auc, train_auc, eval_time, prev_acc, train_loss, post_acc, loss_time)

    logging.info(res)
    print (res)

    return post_acc, ndcg, res

# input: batch_index (shuffled), model, sess, batches
# do: train the model optimizer
def training_batch(model, sess, batches):
    user_input, item_input_pos, user_dns_list, item_dns_list = batches
    # dns for every mini-batch
    # dns = 1, i.e., BPR
    if model.dns == 1:
        item_input_neg = item_dns_list
        # for BPR training
        for i in range(len(user_input)):
            feed_dict = {model.user_input: user_input[i],
                         model.item_input_pos: item_input_pos[i],
                         model.item_input_neg: item_input_neg[i],
                         model.keep_prob: TRAIN_KEEP_PROB}
            sess.run(model.get_optimizer(), feed_dict)
        model.prepared = True
    # dns > 1, i.e., BPR-dns
    elif model.dns > 1:
        item_input_neg = []
        for i in range(len(user_input)):
            # get the output of negtive sample
            feed_dict = {model.user_input: user_dns_list[i],
                         model.item_input_neg: item_dns_list[i],
                         model.keep_prob: TRAIN_KEEP_PROB}
            output_neg = sess.run(model.output_neg, feed_dict)
            # select the best negtive sample as for item_input_neg
            item_neg_batch = []
            for j in range(0, len(output_neg), model.dns):
                item_index = np.argmax(output_neg[j : j + model.dns])
                item_neg_batch.append(item_dns_list[i][j : j + model.dns][item_index][0])
            item_neg_batch = np.array(item_neg_batch)[:,None]
            # for mini-batch BPR training
            feed_dict = {model.user_input: user_input[i],
                         model.item_input_pos: item_input_pos[i],
                         model.item_input_neg: item_neg_batch,
                         model.keep_prob: TRAIN_KEEP_PROB}
            sess.run(model.get_optimizer(), feed_dict)
            item_input_neg.append(item_neg_batch)
    return user_input, item_input_pos, item_input_neg

# input: model, sess, batches
# output: training_loss
def training_loss_acc(model, sess, train_batches):
    train_loss = 0.0
    acc = 0
    num_batch = len(train_batches[1])
    user_input, item_input_pos, item_input_neg = train_batches
    for i in range(len(user_input)):
        # print user_input[i][0]. item_input_pos[i][0], item_input_neg[i][0]
        feed_dict = {model.user_input: user_input[i],
                     model.item_input_pos: item_input_pos[i],
                     model.item_input_neg: item_input_neg[i],
                         model.keep_prob: TEST_KEEP_PROB}

        loss, output_pos, output_neg = sess.run([model.loss, model.output, model.output_neg], feed_dict)
        train_loss += loss
        acc += ((output_pos - output_neg) > 0).sum() / len(output_pos)
    return train_loss / num_batch, acc / num_batch

def init_eval_model(model, dataset):
    global _dataset
    global _model
    _dataset = dataset
    _model = model

    pool = Pool(cpu_count())
    feed_dicts = pool.map(_evaluate_input, range(_dataset.num_users))
    pool.close()
    pool.join()

    print("already load the evaluate model...")
    return feed_dicts

def _evaluate_input(user):
    # generate items_list
    item_input = _dataset.testNegatives[user] # read negative samples from files
    test_item = _dataset.testRatings[user][1]
    item_input.append(test_item)
    user_input = np.full(len(item_input), user, dtype='int32')[:, None]
    item_input = np.array(item_input)[:,None]
    return user_input, item_input


def evaluate(model, sess, dataset, feed_dicts):
    global _model
    global _K
    global _sess
    global _dataset
    global _feed_dicts
    global _output
    _dataset = dataset
    _model = model
    _sess = sess
    _K = 10
    _feed_dicts = feed_dicts

    res = []
    for user in range(_dataset.num_users):
        res.append(_eval_by_user(user))
    res = np.array(res)
    result = (res.mean(axis = 0)).tolist()
    print (result)

    hr, ndcg, auc, train_auc = result[3:6] + [0]

    return hr, ndcg, auc, train_auc

def scoreK(K, position, negs):
    hr = position < K
    if hr:
        ndcg = math.log(2) / math.log(position + 2)
    else:
        ndcg = 0
    auc = 1 - (position * 1. / negs)  # formula: [#(Xui>Xuj) / #(Items)] = [1 - #(Xui<=Xuj) / #(Items)]
    return hr, ndcg, auc

def _eval_by_user(user):

    if _model.train_auc:
        # get predictions of positive samples in training set
        train_item_input = _dataset.trainList[user]
        train_user_input = np.full(len(train_item_input), user, dtype='int32')[:, None]
        train_item_input = np.array(train_item_input)[:, None]
        feed_dict = {_model.user_input: train_user_input, _model.item_input_pos: train_item_input,
                         _model.keep_prob: TEST_KEEP_PROB}

        train_predict = _sess.run(_model.output, feed_dict)

    # get prredictions of data in testing set
    user_input, item_input = _feed_dicts[user]
    feed_dict = {_model.user_input: user_input, _model.item_input_pos: item_input,
                         _model.keep_prob: TEST_KEEP_PROB}

    predictions = _sess.run(_model.output, feed_dict)

    nan_pos = np.argwhere(np.isnan(predictions))
    if len(nan_pos) > 0:
        print ("contain nan {}".format(nan_pos))
        exit()
    neg_predict, pos_predict = predictions[:-1], predictions[-1]
    position = (neg_predict >= pos_predict).sum()

    ret = []
    ret += scoreK(5, position, len(predictions))
    ret += scoreK(10, position, len(predictions))
    ret += scoreK(20, position, len(predictions))
    return ret
    # calculate AUC for training set
    train_auc = 0
    if _model.train_auc:
        for train_res in train_predict:
            train_auc += (train_res >= neg_predict).sum() / len(neg_predict)
        train_auc /= len(train_predict)

    # calculate HR@K, NDCG@K, AUC
    hr = position < _K
    if hr:
        ndcg = math.log(2) / math.log(position+2)
    else:
        ndcg = 0
    auc = 1 - (position * 1. / len(neg_predict))  # formula: [#(Xui>Xuj) / #(Items)] = [1 - #(Xui<=Xuj) / #(Items)]
    return hr, ndcg, auc, train_auc


def init_logging(args):
    regs = args.regs
    path = args.path_partial_results + "%s/" % (strftime('%d-%m-%Y', localtime()))
    if not os.path.exists(path):
        os.makedirs(path)
    fpath = path + "%s_%s" % (args.dataset, strftime('%H_%M_%S', localtime()))
    logging.basicConfig(filename=fpath,
                        level=logging.INFO)
    print ("log to: {}".format(fpath))
    logging.info("begin training %s model ......" % args.model)
    logging.info("dataset:%s  embedding_size:%d   dns:%d    batch_size:%d, regs:%.8f, %.8f  learning_rate:(%.4f, %.4f)"
                 % (args.dataset, args.embed_size, args.dns, args.batch_size, regs[0], regs[1], args.lr_embed, args.lr_net))
    print ("dataset:%s  embedding_size:%d   dns:%d   batch_size:%d, regs:%.8f, %.8f  learning_rate:(%.4f, %.4f)"
                 % (args.dataset, args.embed_size, args.dns, args.batch_size, regs[0], regs[1], args.lr_embed, args.lr_net))




