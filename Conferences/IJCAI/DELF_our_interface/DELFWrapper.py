#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/07/19

@author: Simone Boglio
"""

from Base.BaseRecommender import BaseRecommender
from Base.BaseTempFolder import BaseTempFolder
from Base.DataIO import DataIO
from Base.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping

import tensorflow as tf
import numpy as np
import scipy.sparse as sps


from tqdm import tqdm
from Conferences.IJCAI.DELF_original.Model import NMF_attention_EF, NMF_attention_MLP


class _DELF_RecommenderWrapper(BaseRecommender, Incremental_Training_Early_Stopping, BaseTempFolder):

    RECOMMENDER_NAME = "DELF_RecommenderWrapper"

    def __init__(self, URM_train, model_type='MLP'):
        super(_DELF_RecommenderWrapper, self).__init__(URM_train)

        self.train = sps.dok_matrix(self.URM_train, dtype=np.float32)

        self.num_users = int(self.train.shape[0])
        self.num_items = int(self.train.shape[1])

        self._item_indices = np.arange(0, self.n_items, dtype=np.int32)


        if model_type not in ('MLP', 'EF'):
            print('{}: Model type not correct, possible are MLP and EF'.format(self.RECOMMENDER_NAME))
            print('{}: Select default model type MLP'.format(self.RECOMMENDER_NAME))
            model_type='MLP'

        self.model_type = model_type


    def _compute_item_score(self, user_id_array, items_to_compute=None):

        if items_to_compute is None:
            item_indices = self._item_indices
        else:
            item_indices = items_to_compute

        n_items = len(item_indices)
        n_users_batch = len(user_id_array)
        item_scores = - np.ones((n_users_batch, self.num_items), dtype=np.float32)*np.inf

        for user_index in range(n_users_batch):
            u_i = user_id_array[user_index]

            users = np.full(n_items, u_i, dtype='int32')


            predictions = self.sess.run(self.model.predict, feed_dict={self.input_user: np.expand_dims(users, axis=1),
                                                                       self.input_item: np.expand_dims(item_indices, axis=1),
                                                                       self.rating_matrix: self.train_arr})
            # predictions = np.array(predictions).reshape((-1))
            # scores = np.ones(self.n_items) * (-np.inf)
            # scores[items_to_compute] = predictions
            item_scores[user_index, items_to_compute] = np.array(predictions).ravel()

        return item_scores

    def fit(self,
            epochs=1,
            learning_rate=0.001,
            batch_size=256,
            num_negatives=4,
            layers=(256, 128, 64),
            regularization_layers=(0, 0, 0),
            learner='adam',
            verbose=True,
            temp_file_folder=None,
            **earlystopping_kwargs):

        self.layers = layers
        self.reg_layers = regularization_layers
        self.num_negatives = num_negatives
        self.learner = learner
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.predictive_factors = self.layers[-1]


        self.temp_file_folder = self._get_unique_temp_folder(input_temp_file_folder=temp_file_folder)

        print("{}: Init model DELF-{}...".format(self.RECOMMENDER_NAME, self.model_type))

        NMFs = {"EF": NMF_attention_EF, 'MLP': NMF_attention_MLP}
        NMF = NMFs[self.model_type]

        tf.reset_default_graph()
        self.train_arr = self.train.toarray()
        self.input_user = tf.placeholder(tf.int32, [None, 1])
        self.input_item = tf.placeholder(tf.int32, [None, 1])
        self.output = tf.placeholder(tf.float32, [None, 1])
        self.rating_matrix = tf.placeholder(tf.float32, shape=(self.num_users, self.num_items))

        user_input, item_input, labels = get_train_instances(self.train, self.num_negatives)

        self.batch_len = len(user_input) // self.batch_size

        self.model = NMF.Model(self.input_user, self.input_item, self.output, self.num_users, self.num_items, self.rating_matrix, self.layers, self.batch_len)
        tf.summary.histogram("input_user", self.input_user)

        self.merged_summary = tf.summary.merge_all()

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter(self.DEFAULT_TEMP_FILE_FOLDER + "NMF_fmn")
        writer.add_graph(self.sess.graph)

        self.loss = np.inf

        print("{}: Init model... done!".format(self.RECOMMENDER_NAME))

        print("{}: Training...".format(self.RECOMMENDER_NAME))

        self._update_best_model()

        self._train_with_early_stopping(epochs_max=self.epochs, algorithm_name=self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        # close session tensorflow
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

        if self.verbose: print("Generate training instances epoch: {}".format(currentEpoch))
        # Generate training instances
        user_input, item_input, labels = get_train_instances(self.train, self.num_negatives)
        user_input, item_input, labels = unison_shuffled_copies(np.asarray(user_input), np.asarray(item_input), np.asarray(labels))

        if self.verbose: print("Begin training epoch: {}".format(currentEpoch))
        batch_len = len(user_input) // self.batch_size

        batch_size_range = range(self.batch_size)

        if self.verbose:
            batch_size_range = tqdm(batch_size_range)

        for batch in batch_size_range:
            _, self.loss = self.sess.run(self.model.optimize,
                               feed_dict={self.input_user: np.expand_dims(user_input, axis=1)[
                                                      batch * batch_len:(batch + 1) * batch_len],
                                          self.input_item: np.expand_dims(item_input, axis=1)[
                                                      batch * batch_len:(batch + 1) * batch_len],
                                          self.output: np.expand_dims(labels, axis=1)[
                                                  batch * batch_len:(batch + 1) * batch_len],
                                          self.rating_matrix: self.train_arr})

        if self.verbose: print("Epoch: {}, loss: {}".format(currentEpoch, self.loss))



    def save_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {'layers':self.layers,
                              'reg_layers':self.reg_layers,
                              'num_negatives':self.num_negatives,
                              'learner':self.learner,
                              'learning_rate':self.learning_rate,
                              'batch_size':self.batch_size,
                              'predictive_factors':self.predictive_factors,
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

        self.train_arr = self.train.toarray()
        self.input_user = tf.placeholder(tf.int32, [None, 1])
        self.input_item = tf.placeholder(tf.int32, [None, 1])
        self.output = tf.placeholder(tf.float32, [None, 1])
        self.rating_matrix = tf.placeholder(tf.float32, shape=(self.num_users, self.num_items))

        user_input, item_input, labels = get_train_instances(self.train, self.num_negatives)

        self.batch_len = len(user_input) // self.batch_size
        NMFs = {"EF": NMF_attention_EF, 'MLP': NMF_attention_MLP}
        NMF = NMFs[self.model_type]

        self.model = NMF.Model(self.input_user, self.input_item, self.output, self.num_users, self.num_items,
                               self.rating_matrix, self.layers, self.batch_len)



        gpu_options = tf.GPUOptions(allow_growth=True)

        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(self.sess, folder_path + file_name + "_session")

        self._print("Loading complete")


class DELF_MLP_RecommenderWrapper(_DELF_RecommenderWrapper):

    RECOMMENDER_NAME = "DELF_MLP_RecommenderWrapper"

    def __init__(self, URM_train):
        super(DELF_MLP_RecommenderWrapper, self).__init__(URM_train, model_type='MLP')


class DELF_EF_RecommenderWrapper(_DELF_RecommenderWrapper):

    RECOMMENDER_NAME = "DELF_EF_RecommenderWrapper"

    def __init__(self, URM_train):
        super(DELF_EF_RecommenderWrapper, self).__init__(URM_train, model_type='EF')





def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [], [], []
    num_users = train.shape[0]
    num_items = train.shape[1]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train: # python3
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels


def unison_shuffled_copies(a, b, c):
    assert len(a) == len(b)
    assert len(a) == len(c)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]
