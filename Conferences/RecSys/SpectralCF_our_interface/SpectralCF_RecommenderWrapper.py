#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 18/12/18

@author: Maurizio Ferrari Dacrema
"""


from Base.BaseRecommender import BaseRecommender
from Base.BaseTempFolder import BaseTempFolder
from Base.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Base.DataIO import DataIO

import numpy as np
import scipy.sparse as sps

import tensorflow as tf
import random as rd
import os, shutil, math

from Conferences.RecSys.SpectralCF_our_interface.SpectralCF import SpectralCF




class Data(object):
    def __init__(self, URM_train, batch_size):
        self.batch_size = batch_size

        URM_train = sps.csr_matrix(URM_train)
        self.n_users, self.n_items = URM_train.shape

        self.R = np.zeros((self.n_users, self.n_items), dtype=np.float32)

        self._users_with_interactions = np.ediff1d(URM_train.indptr)>=1
        self._users_with_interactions = np.arange(self.n_users, dtype=np.int64)[self._users_with_interactions]
        self._users_with_interactions = list(self._users_with_interactions)

        self.train_items, self.test_set = {}, {}

        for user_index in range(self.n_users):

            start_pos = URM_train.indptr[user_index]
            end_pos = URM_train.indptr[user_index+1]

            train_items = URM_train.indices[start_pos:end_pos]

            self.R[user_index][train_items] = 1
            self.train_items[user_index] = list(train_items)



    def sample(self):
        if self.batch_size <= self.n_users:
            #users = rd.sample(range(self.n_users), self.batch_size)
            users = rd.sample(self._users_with_interactions, self.batch_size)

        else:
            #users = [rd.choice(range(self.n_users)) for _ in range(self.batch_size)]
            users = [rd.choice(self._users_with_interactions) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]#np.nonzero(self.graph[u,:])[0].tolist()
            if len(pos_items) >= num:
                return rd.sample(pos_items, num)
            else:
                return [rd.choice(pos_items) for _ in range(num)]

        def sample_neg_items_for_u(u, num):
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))#np.nonzero(self.graph[u,:] == 0)[0].tolist()
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items





class SpectralCF_RecommenderWrapper(BaseRecommender, Incremental_Training_Early_Stopping, BaseTempFolder):


    RECOMMENDER_NAME = "SpectralCF_RecommenderWrapper"

    def __init__(self, URM_train):
        super(SpectralCF_RecommenderWrapper, self).__init__(URM_train)

        self._train = sps.dok_matrix(self.URM_train)


    def _compute_item_score(self, user_id_array, items_to_compute=None):

        if len(user_id_array) < self.batch_size:
            user_batch = np.zeros((self.batch_size), dtype=np.int64)
            user_batch[0:len(user_id_array)] = user_id_array

        else:
            user_batch = user_id_array


        # If user id batch is too long, slice the current user_id_array in blocks that the model can handle
        if len(user_id_array) > self.batch_size:

            n_split = math.ceil(len(user_id_array)/ self.batch_size)

            split_list = np.array_split(user_id_array, n_split)

            item_scores_to_compute = - np.ones((len(user_id_array), self.n_items)) * np.inf

            start_pos = 0
            for user_id_array_split in split_list:
                item_scores_to_compute[start_pos:start_pos+len(user_id_array_split),:] = self._compute_item_score(user_id_array_split, items_to_compute=items_to_compute)
                start_pos+=len(user_id_array_split)

        else:

            item_scores_to_compute = self.sess.run(self.model.all_ratings, {self.model.users: user_batch})


        if len(user_id_array) < self.batch_size:
            item_scores_to_compute = item_scores_to_compute[0:len(user_id_array),:]


        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.n_items)) * np.inf
            item_scores[:, items_to_compute] = item_scores_to_compute[:, items_to_compute]
        else:
            item_scores = item_scores_to_compute

        return item_scores




    def fit(self,
            epochs = 200,
            batch_size = 1024,
            embedding_size = 16,
            decay = 0.001,
            k = 3,
            learning_rate = 1e-3,
            temp_file_folder = None,
            **earlystopping_kwargs
            ):


        self.temp_file_folder = self._get_unique_temp_folder(input_temp_file_folder=temp_file_folder)


        self.k = k
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.decay = decay
        self.batch_size = batch_size



        print("SpectralCF_RecommenderWrapper: Instantiating model...")

        tf.reset_default_graph()

        self.data_generator = Data(self.URM_train, batch_size=self.batch_size)

        self.model = SpectralCF(K=self.k,
                           graph = self.URM_train.toarray(),
                           n_users = self.n_users,
                           n_items = self.n_items,
                           emb_dim = self.embedding_size,
                           lr = self.learning_rate,
                           decay = self.decay,
                           batch_size = self.batch_size)

        self.model.compute_eigenvalues()

        # Keep it to avoid recomputing every time the model is loaded
        # self.model_lamda = self.model.lamda.copy()
        # self.model_U = self.model.U

        self.model.build_graph()


        print("SpectralCF_RecommenderWrapper: Instantiating model... done!")
        # print(self.model.model_name)

        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # self.sess = tf.Session(config=config)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


        print("SpectralCF_RecommenderWrapper: Training SpectralCF...")

        self._update_best_model()

        self._train_with_early_stopping(epochs,
                                        algorithm_name = self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        self.sess.close()
        self.sess = tf.Session()

        self.load_model(self.temp_file_folder, file_name="_best_model")


        self._print("Training complete")


        self._clean_temp_folder(temp_file_folder=self.temp_file_folder)



    def _prepare_model_for_validation(self):
        pass


    def _update_best_model(self):
        self.save_model(self.temp_file_folder, file_name="_best_model")



    def _run_epoch(self, currentEpoch):

        users, pos_items, neg_items = self.data_generator.sample()

        _, loss = self.sess.run([self.model.updates, self.model.loss],
                                           feed_dict={self.model.users: users, self.model.pos_items: pos_items,
                                                      self.model.neg_items: neg_items})

        print("SpectralCF_RecommenderWrapper: Epoch {}, loss {:.2E}".format(currentEpoch+1, loss))

        if not np.isfinite(loss):
            self._clean_temp_folder(temp_file_folder=self.temp_file_folder)

            assert False, "SpectralCF_RecommenderWrapper: loss is not a finite number, terminating!"




















    def save_model(self, folder_path, file_name = None):


        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {"k": self.k,
                              "embedding_size": self.embedding_size,
                              "learning_rate": self.learning_rate,
                              "decay": self.decay,
                              "batch_size": self.batch_size,
                              # "model_lamda": self.model_lamda,
                              # "model_U": self.model_U,
                              }

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)

        saver = tf.train.Saver()

        saver.save(self.sess, folder_path + file_name + "_session")


        self._print("Saving complete")




    def load_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Loading model from file '{}'".format(folder_path + file_name))

        dataIO = DataIO(folder_path=folder_path)
        data_dict = dataIO.load_data(file_name=file_name)

        for attrib_name in data_dict.keys():
             self.__setattr__(attrib_name, data_dict[attrib_name])


        tf.reset_default_graph()

        self.data_generator = Data(self.URM_train, batch_size=self.batch_size)

        self.model = SpectralCF(K=self.k,
                           graph = self.URM_train.toarray(),
                           n_users = self.n_users,
                           n_items = self.n_items,
                           emb_dim = self.embedding_size,
                           lr = self.learning_rate,
                           decay = self.decay,
                           batch_size = self.batch_size)

        self.model.compute_eigenvalues()
        self.model.build_graph()



        saver = tf.train.Saver()
        self.sess = tf.Session()

        saver.restore(self.sess, folder_path + file_name + "_session")


        self._print("Loading complete")

