#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 18/12/18

@author: Maurizio Ferrari Dacrema
"""


from Base.BaseCBFRecommender import BaseItemCBFRecommender
from Base.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from Base.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Base.BaseTempFolder import BaseTempFolder
from Base.Recommender_utils import check_matrix
from Base.Recommender_utils import get_unique_temp_folder

import numpy as np
import tensorflow as tf
import os, shutil
import scipy.sparse as sps

from Conferences.KDD.CollaborativeVAE_github.lib.vae import VariationalAutoEncoder
from Conferences.KDD.CollaborativeVAE_github.lib.utils import init_logging, logging
from Conferences.KDD.CollaborativeVAE_github.lib.cvae import CVAE, Params




class CollaborativeVAE_RecommenderWrapper(BaseItemCBFRecommender, BaseMatrixFactorizationRecommender, Incremental_Training_Early_Stopping, BaseTempFolder):

    RECOMMENDER_NAME = "CollaborativeVAE_RecommenderWrapper"

    def __init__(self, URM_train, ICM_train):
        super(CollaborativeVAE_RecommenderWrapper, self).__init__(URM_train, ICM_train)


    def fit(self,
            epochs = 100,
            learning_rate_vae = 1e-2,
            learning_rate_cvae = 1e-3,
            num_factors = 50,
            dimensions_vae = [200, 100],
            epochs_vae = [50, 50],
            batch_size = 128,
            lambda_u = 0.1,
            lambda_v = 10,
            lambda_r = 1,
            a = 1,
            b = 0.01,
            M = 300,
            temp_file_folder = "result_experiments\CollaborativeVAE_RecommenderWrapper",

            **earlystopping_kwargs
            ):


        self.temp_file_folder = self._get_unique_temp_folder(input_temp_file_folder=temp_file_folder)

        print("{}: Pretraining with VariationalAutoEncoder".format(self.RECOMMENDER_NAME))

        random_seed = 0
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)

        activations=['sigmoid', 'sigmoid']

        init_logging(self.temp_file_folder + "_log.txt")

        logging.info('loading data')

        _, input_dim = self.ICM_train.shape

        self.ICM_train = self.ICM_train.toarray()
        idx = np.random.rand(self.ICM_train.shape[0]) < 0.8
        train_X = self.ICM_train[idx, :]
        test_X = self.ICM_train[~idx, :]

        logging.info('initializing sdae model')

        tf.reset_default_graph()

        model = VariationalAutoEncoder(input_dim = input_dim,
                                       dims=dimensions_vae,
                                       z_dim=num_factors,
                                       activations=activations,
                                       epoch=epochs_vae,
                                       noise='mask-0.3',
                                       loss='cross-entropy',
                                       lr=learning_rate_vae,
                                       batch_size = batch_size,
                                       print_step=1,
                                       weight_path=self.temp_file_folder + "_pretrain")

        logging.info('fitting data starts...')
        model.fit(train_X, test_X)







        print("{}: Finalizing with VariationalAutoEncoder".format(self.RECOMMENDER_NAME))

        self._params = Params()
        self._params.lambda_u = lambda_u
        self._params.lambda_v = lambda_v
        self._params.lambda_r = lambda_r
        self._params.a = a
        self._params.b = b
        self._params.M = M
        self._params.n_epochs = epochs


        # These are the train instances as a list of lists
        self._train_users = []

        self.URM_train = sps.csr_matrix(self.URM_train)

        for user_index in range(self.n_users):

            start_pos = self.URM_train.indptr[user_index]
            end_pos = self.URM_train.indptr[user_index +1]

            user_profile = self.URM_train.indices[start_pos:end_pos]
            self._train_users.append(list(user_profile))


        self._train_items = []

        self.URM_train = sps.csc_matrix(self.URM_train)

        for user_index in range(self.n_items):

            start_pos = self.URM_train.indptr[user_index]
            end_pos = self.URM_train.indptr[user_index +1]

            item_profile = self.URM_train.indices[start_pos:end_pos]
            self._train_items.append(list(item_profile))





        self.URM_train = sps.csr_matrix(self.URM_train)

        tf.reset_default_graph()

        self.model = CVAE(num_users = self.n_users,
                          num_items = self.n_items,
                          num_factors=num_factors,
                          params=self._params,
                          input_dim = input_dim,
                          dims=dimensions_vae,
                          n_z=num_factors,
                          activations=activations,
                          loss_type='cross-entropy',
                          lr=learning_rate_cvae,
                          random_seed=random_seed,
                          print_step=10,
                          verbose=False)

        #weight_path = "model/pretrain"
        self.model.load_model(weight_path=self.temp_file_folder + "_pretrain")


        self._update_best_model()

        self.model.m_theta[:] = self.model.transform(self.ICM_train)
        self.model.m_V[:] = self.model.m_theta

        self._train_with_early_stopping(epochs,
                                        algorithm_name = self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)


        self._print("Training complete")

        self.USER_factors = self.USER_factors_best
        self.ITEM_factors = self.ITEM_factors_best

        self._clean_temp_folder(temp_file_folder=self.temp_file_folder)




    def _prepare_model_for_validation(self):
        pass


    def _update_best_model(self):
        self.USER_factors_best = self.model.m_U.copy()
        self.ITEM_factors_best = self.model.m_V.copy()





    def _run_epoch(self, currentEpoch):


        n = self.ICM_train.shape[0]

        # for epoch in range(self._params.n_epochs):
        num_iter = int(n / self._params.batch_size)
        # gen_loss = self.cdl_estimate(data_x, params.cdl_max_iter)
        gen_loss = self.model.cdl_estimate(self.ICM_train, num_iter)
        self.model.m_theta[:] = self.model.transform(self.ICM_train)
        likelihood = self.model.pmf_estimate(self._train_users, self._train_items, None, None, self._params)
        loss = -likelihood + 0.5 * gen_loss * n * self._params.lambda_r

        self.USER_factors = self.model.m_U.copy()
        self.ITEM_factors = self.model.m_V.copy()

        logging.info("[#epoch=%06d], loss=%.5f, neg_likelihood=%.5f, gen_loss=%.5f" % (
            currentEpoch, loss, -likelihood, gen_loss))



