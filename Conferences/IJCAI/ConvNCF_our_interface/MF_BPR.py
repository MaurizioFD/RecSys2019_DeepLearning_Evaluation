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
# _exclude_gtItem = None
_user_exclude_validation = set()

#---------- data preparation -------
# data sampling and shuffling

# input: dataset(Mat, List, Rating, Negatives), batch_choice
# output: [_user_input_list, _item_input_pos_list]
def sampling(dataset):
    _user_input, _item_input_pos = [], []
    for (u, i) in dataset.trainMatrix.keys():
        # positive instance
        _user_input.append(u)
        _item_input_pos.append(i)
    return _user_input, _item_input_pos

def shuffle(samples, batch_size, dataset, model):#, exclude_gtItem=True):
    global _user_input
    global _item_input_pos
    global _batch_size
    global _index
    global _model
    global _dataset
    # global _exclude_gtItem
    # _exclude_gtItem = exclude_gtItem
    _user_input, _item_input_pos = samples
    _batch_size = batch_size
    _index = np.array(range(len(_user_input)))
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
            # if _exclude_gtItem:
            #     try:
            #         gtItem = _dataset.testRatings[user][1]
            #     except IndexError:  # User does not have item in test set
            #         gtItem = -1
            # else:
            #     # trick to avoid original paper overfitting
            #     gtItem = -1
            #
            # Never check whether a train negative item is also in test
            j = np.random.randint(_dataset.num_items)
            while j in _dataset.trainList[_user_input[_index[idx]]]:# or j == gtItem:
                j = np.random.randint(_dataset.num_items)
            item_neg_batch.append(j)
    return np.array(user_batch)[:,None], np.array(item_batch)[:,None], \
           np.array(user_neg_batch)[:,None], np.array(item_neg_batch)[:,None]

#---------- model definition -------
class GMF:
    def __init__(self, num_users, num_items, args):
        self.num_items = num_items
        self.num_users = num_users
        self.embedding_size = args.embed_size
        self.learning_rate = args.lr
        regs = args.regs
        self.lambda_bilinear = regs[0]
        self.gamma_bilinear = regs[1]
        self.dns = args.dns
        self.train_auc = args.train_auc

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape = [None, 1], name = "user_input")
            self.item_input_pos = tf.placeholder(tf.int32, shape = [None, 1], name = "item_input_pos")
            self.item_input_neg = tf.placeholder(tf.int32, shape = [None, 1], name = "item_input_neg")
    def _create_variables(self):
        with tf.name_scope("embedding"):
            self.embedding_P = tf.Variable(tf.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0, stddev=0.01),
                                                                name='embedding_P', dtype=tf.float32)  #(users, embedding_size)
            self.embedding_Q = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
                                                                name='embedding_Q', dtype=tf.float32)  #(items, embedding_size)

            self.h = tf.constant(1.0, tf.float32, [self.embedding_size, 1], name = "h")

    def _create_inference(self, item_input):
        with tf.name_scope("inference"):
            # embedding look up
            self.embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P, self.user_input), 1)
            self.embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, item_input), 1) #(b, embedding_size)

            return self.embedding_p, self.embedding_q,\
                tf.matmul(self.embedding_p*self.embedding_q, self.h)  #(b, embedding_size) * (embedding_size, 1)

    def _create_loss(self):
        with tf.name_scope("loss"):
            # loss for L(Theta)
            self.p1, self.q1, self.output = self._create_inference(self.item_input_pos)
            self.p2, self.q2, self.output_neg = self._create_inference(self.item_input_neg)
            self.result = self.output - self.output_neg
            self.loss = tf.reduce_sum(tf.log(1 + tf.exp(-self.result)))

            self.opt_loss = self.loss + self.lambda_bilinear * ( tf.reduce_sum(tf.square(self.p1))
                                    + tf.reduce_sum(tf.square(self.q2)) + tf.reduce_sum(tf.square(self.q1)))

    def _create_optimizer(self):
        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(self.opt_loss)

    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_loss()
        self._create_optimizer()

    def saveParams(self, sess, fname, args):
        path = args.path_partial_results
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(path+fname, sess.run([self.embedding_P,self.embedding_Q,self.h]))

    def load_parameter_MF(self, sess, path):
        ps = np.load(path)
        ap = tf.assign(self.embedding_P, ps[0])
        aq = tf.assign(self.embedding_Q, ps[1])
        #ah = tf.assign(self.h, np.diag(ps[2][:, 0]).reshape(4096, 1))
        sess.run([ap, aq])
        print ("parameter loaded")

#---------- training process -------

def training(model, dataset, args, saver = None): # saver is an object to save pq
    _user_exclude_validation = set()
    with tf.Session() as sess:
        # initialized the save op
        # ckpt_save_path = "Pretrain/MF_BPR/embed_%d/" %  args.embed_size
        # ckpt_restore_path = "Pretrain/MF_BPR/embed_%d/" %  args.embed_size
        # if not os.path.exists(ckpt_save_path):
        #     os.makedirs(ckpt_save_path)
        # if not os.path.exists(ckpt_restore_path):
        #     os.makedirs(ckpt_restore_path)

        # initialize saver
        saver_ckpt = tf.train.Saver({'embedding_P':model.embedding_P,'embedding_Q':model.embedding_Q})

        # pretrain or not
        sess.run(tf.global_variables_initializer())

        # restore the weights when pretrained
        # if args.pretrain:
        #     ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_restore_path + 'checkpoint'))
        #     if ckpt and ckpt.model_checkpoint_path:
        #         saver_ckpt.restore(sess, ckpt.model_checkpoint_path)
        #         logging.info("using pretrained variables")
        #         print ("using pretrained variables")

        # initialize the weights
        # else:
        logging.info("initialized")
        print ("initialized")

        # initialize for Evaluate
        eval_feed_dicts = init_eval_model(model, dataset)

        # sample the data
        samples = sampling(dataset)

        #initialize the max_ndcg to memorize the best result
        max_ndcg = 0
        max_res = " "

        # train by epoch
        for epoch_count in range(args.epochs):

            # initialize for training batches
            batch_begin = time()
            batches = shuffle(samples, args.batch_size, dataset, model)#, args.exclude_gtItem)
            batch_time = time() - batch_begin

            # compute the accuracy before training
            prev_batch = batches[0], batches[1], batches[3]
            _, prev_acc = training_loss_acc(model, sess, prev_batch)

            # training the model
            train_begin = time()
            train_batches = training_batch(model, sess, batches)
            train_time = time() - train_begin

            if epoch_count % args.verbose == 0 or epoch_count == args.epochs-1:
                _, ndcg, cur_res = output_evaluate(model, sess, dataset, train_batches, eval_feed_dicts,
                                epoch_count, batch_time, train_time, prev_acc)

                # print and log the best result
                if max_ndcg < ndcg:
                    max_ndcg = ndcg
                    max_res = cur_res
                    model.saveParams(sess, "best_%s_MFBPR.npy" % args.dataset, args)
                    print("New Best saved..")

            # save the embedding weights
            # if args.ckpt and epoch_count%100 == 0:
            #     saver_ckpt.save(sess, ckpt_save_path+'weights', global_step=epoch_count)

        model.saveParams(sess, "final_%s_MFBPR.npy" % args.dataset, args)
        print ("best: {}".format(max_res))
        logging.info("best:" + max_res)
    #add this line to reset default graph after close session
    tf.reset_default_graph()


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
                         model.item_input_neg: item_input_neg[i]}
            sess.run(model.optimizer, feed_dict)
    # dns > 1, i.e., BPR-dns
    elif model.dns > 1:
        item_input_neg = []
        for i in range(len(user_input)):
            # get the output of negtive sample
            feed_dict = {model.user_input: user_dns_list[i],
                         model.item_input_neg: item_dns_list[i]}
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
                         model.item_input_neg: item_neg_batch}
            sess.run(model.optimizer, feed_dict)
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
                     model.item_input_neg: item_input_neg[i]}

        loss, output_pos, output_neg = sess.run([model.loss, model.output, model.output_neg], feed_dict)
        train_loss += loss
        acc += ((output_pos - output_neg) > 0).sum() / len(output_pos)
    return train_loss / num_batch, acc / num_batch

#---------- evaluation -------

def output_evaluate(model, sess, dataset, train_batches, eval_feed_dicts, epoch_count, batch_time, train_time, prev_acc):
    loss_begin = time()
    train_loss, post_acc = 0,0 #training_loss_acc(model, sess, train_batches)
    loss_time = time() - loss_begin

    eval_begin = time()
    hr, ndcg, auc, train_auc = evaluate(model, sess, dataset, eval_feed_dicts)
    eval_time = time() - eval_begin

    res = "Epoch %d [%.1fs + %.1fs]: HR = %.4f, NDCG = %.4f AUC = %.4f train_AUC = %.4f [%.1fs]" \
        " ACC = %.4f train_loss = %.4f ACC = %.4f [%.1fs]" % ( epoch_count, batch_time, train_time,
                    hr, ndcg, auc, train_auc, eval_time, prev_acc, train_loss, post_acc, loss_time)
    # res = "Epoch %d [%.1fs + %.1fs]: HR = %.4f, NDCG = %.4f AUC = %.4f [%.1fs]" % (epoch_count, batch_time, train_time, hr, ndcg, auc, eval_time)

    logging.info(res)
    print (res)

    return post_acc, ndcg, res

def init_eval_model(model, dataset):
    global _dataset
    global _model
    _dataset = dataset
    _model = model

    pool = Pool(cpu_count())
    feed_dicts = pool.map(_evaluate_input, range(_dataset.num_users))
    pool.close()
    pool.join()

    return feed_dicts

def _evaluate_input(user):
    # generate items_list
    item_input = _dataset.testNegatives[user] # read negative samples from files
    try:
        test_item = _dataset.testRatings[user][1]
        item_input.append(test_item)
    except IndexError:
        _user_exclude_validation.add(user)
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
        if(user not in _user_exclude_validation):
            res.append(_eval_by_user(user))
    res = np.array(res)
    hr, ndcg, auc, train_auc = (res.mean(axis = 0)).tolist()

    return hr, ndcg, auc, train_auc

def _eval_by_user(user):

    if _model.train_auc:
        # get predictions of positive samples in training set
        train_item_input = _dataset.trainList[user]
        train_user_input = np.full(len(train_item_input), user, dtype='int32')[:, None]
        train_item_input = np.array(train_item_input)[:, None]
        feed_dict = {_model.user_input: train_user_input, _model.item_input_pos: train_item_input}

        train_predict = _sess.run(_model.output, feed_dict)

    # get prredictions of data in testing set
    user_input, item_input = _feed_dicts[user]
    feed_dict = {_model.user_input: user_input, _model.item_input_pos: item_input}

    predictions = _sess.run(_model.output, feed_dict)

    neg_predict, pos_predict = predictions[:-1], predictions[-1]
    position = (neg_predict >= pos_predict).sum()

    # calculate AUC for training set
    train_auc = 0
    if _model.train_auc:
        for train_res in train_predict:
            train_auc += (train_res > neg_predict).sum() / len(neg_predict)
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
    path = args.path_partial_results +  "%s/" % (strftime('%d-%m-%Y', localtime()))
    if not os.path.exists(path):
        os.makedirs(path)

    fpath = path + "%s_%s" % (args.dataset,strftime('%H_%M_%S', localtime()))
    logging.basicConfig(filename=fpath,
                        level=logging.INFO)
    print ("log to: {}".format(fpath))
    logging.info("begin training %s model ......" % args.model)
    logging.info("dataset: %s, embedding_size: %d, dns: %d, batch_size: %d, regs[u,i]: [%.3f, %.3f], learning_rate: %.3f"
                 % (args.dataset, args.embed_size, args.dns, args.batch_size, regs[0], regs[1], args.lr))
    print ("dataset: %s, embedding_size: %d, dns: %d, batch_size: %d, regs[u,i]: [%.3f, %.3f], learning_rate: %.3f"
                 % (args.dataset, args.embed_size, args.dns, args.batch_size, regs[0], regs[1], args.lr))




