#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 24/02/19

@author: Simone Boglio
"""

import numpy as np
import Conferences.IJCAI.ConvNCF_our_interface.MF_BPR as MF_BPR

class DatasetInterface:

    def __init__(self, URM_train):#, URM_test, URM_negative):

        self.num_users, self.num_items = URM_train.shape

        self.trainMatrix = URM_train.todok()

        m_train = URM_train.tocsr()
        train_list = []
        for u in range(m_train.shape[0]):
            items = m_train.indices[m_train.indptr[u]:m_train.indptr[u + 1]].tolist()
            train_list.append(items)
        self.trainList = train_list

        # m_test = URM_test.tocoo()
        # test_users = np.unique(m_test.row)
        # # original paper sampling avoid to sample item if in test set (but in this way they use the test set as knowledge)
        # self.testRatings = np.array([m_test.row, m_test.col]).T.tolist()
        #
        # m_test = URM_test.tocsr()
        # test_list = []
        # for u in range(m_test.shape[0]):
        #     items = m_test.indices[m_test.indptr[u]:m_test.indptr[u + 1]].tolist()
        #     assert len(items) <= 1, 'atm is supported only leave one out'
        #     if len(items) == 0:
        #         test_list.append([u])
        #     else:
        #         test_list.append([u, items[0]])
        # self.testRatings = test_list

        self.testRatings = None


        # m_negative = URM_negative.tocsr()
        # negative_list = []
        # for u in range(m_negative.shape[0]):
        #     items = m_negative.indices[m_negative.indptr[u]:m_negative.indptr[u + 1]].tolist()
        #     negative_list.append(items)
        # self.testNegatives = negative_list


        self.testNegatives = None

        # assert len(self.testRatings) == len(self.testNegatives) and len(self.testRatings) == self.num_users

class ArgsInterface:

    def __init__(self):
        pass



import platform
from Base.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from Base.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Base.DataIO import DataIO

class MFBPR_Wrapper(BaseMatrixFactorizationRecommender, Incremental_Training_Early_Stopping):

    RECOMMENDER_NAME = "MF_BPR_Wrapper"


    def __init__(self, URM_train):
        super(MFBPR_Wrapper, self).__init__(URM_train)

        assert platform.system() != "Windows", "Current recommender is implemented using global variables and will not work under Windows"

        self._item_indices = np.arange(0, self.n_items, dtype=np.int32)

    def _print(self, string):
        print("{}: {}".format(self.RECOMMENDER_NAME, string))



    def _init_model(self):

        MF_BPR.tf.reset_default_graph()

        MF_BPR.init_logging(self.args)

        # initialize dataset
        self.dataset = DatasetInterface(URM_train=self.URM_train)#, URM_test=URM_test, URM_negative=URM_negative)

        # initialize models
        self.model_GMF = MF_BPR.GMF(self.dataset.num_users, self.dataset.num_items, self.args)
        self.model_GMF.build_graph()

        self.sess = MF_BPR.tf.Session()
        self.sess.run(MF_BPR.tf.global_variables_initializer())

        # sample the data
        self.samples = MF_BPR.sampling(self.dataset)




    def fit(self,
            batch_size=512,
            epochs=500,
            embed_size=64,
            negative_sample_per_positive=1,
            regularization_users=0.01,
            regularization_items=0.0,
            learning_rate=0.05,
            epoch_verbose=25,
            train_auc_verbose=0,
            path_partial_results = None,
            **earlystopping_kwargs
            ):

        assert path_partial_results is not None
        self.path_partial_results = path_partial_results

        self.args = ArgsInterface()
        self.args.dataset = 'no_dataset_name'
        self.args.model = self.RECOMMENDER_NAME
        self.args.verbose = epoch_verbose
        self.args.batch_size = batch_size
        self.args.epochs = epochs
        self.args.embed_size = embed_size
        self.args.dns = negative_sample_per_positive
        self.args.regs = [regularization_users, regularization_items]
        self.args.task = ''
        self.args.lr = learning_rate
        self.args.pretrain = 0
        self.args.ckpt = 0
        self.args.train_auc = train_auc_verbose
        self.args.path_partial_results = path_partial_results

        self._init_model()

        self._print("Start training...")

        self._update_best_model()

        self._train_with_early_stopping(epochs,
                                        algorithm_name = self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        self.epochs_best_MFBPR = self.epochs_best

        self._print("Training complete")

        self.sess.close()
        self.load_model(self.path_partial_results, file_name="best_model")


    def _dealloc_global_variables(self):

        del MF_BPR._user_input
        del MF_BPR._item_input_pos
        del MF_BPR._batch_size
        del MF_BPR._index
        del MF_BPR._model
        del MF_BPR._sess
        del MF_BPR._dataset

        try:
            del MF_BPR._K
        except:
            pass

        try:
            del MF_BPR._feed_dict
        except:
            pass

        try:
            del MF_BPR._output
        except:
            pass

        try:
            del MF_BPR._user_exclude_validation
        except:
            pass



    def _prepare_model_for_validation(self):
        pass


    def _update_best_model(self):
        self.model_GMF.saveParams(self.sess, "_latent_factors", self.args)

        # Load latent factors
        latent_factors = np.load(self.args.path_partial_results + "_latent_factors.npy", allow_pickle=True)
        self.USER_factors, self.ITEM_factors = latent_factors[0], latent_factors[1]

        self.save_model(self.path_partial_results, file_name="best_model")


    def _run_epoch(self, num_epoch):

        # initialize for training batches
        batches = MF_BPR.shuffle(self.samples, self.args.batch_size, self.dataset, self.model_GMF)  # , args.exclude_gtItem)

        # training the model
        _ = MF_BPR.training_batch(self.model_GMF, self.sess, batches)

