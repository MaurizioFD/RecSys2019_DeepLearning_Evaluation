#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 17/03/19

@author: Simone Boglio
"""


from Base.BaseRecommender import BaseRecommender
from Base.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping

import time
from Base.DataIO import DataIO

from keras.optimizers import Adam
from keras import utils
import numpy as np

from CNN_on_embeddings.IJCAI.CoupledCF_our_interface import mainTafengUserCnn, mainMovieUserCnn
from CNN_on_embeddings.IJCAI.CoupledCF_our_interface.mainMovieUserCnn import get_train_instances
from Base.BaseTempFolder import BaseTempFolder

class CoupledCF_RecommenderWrapper(BaseRecommender, Incremental_Training_Early_Stopping, BaseTempFolder):

    RECOMMENDER_NAME = "CoupledCF_RecommenderWrapper"

    __AVAILABLE_MAP_MODES = ["all_map", "main_diagonal", "off_diagonal"]

    def __init__(self, URM_train, UCM, ICM):
        super(CoupledCF_RecommenderWrapper, self).__init__(URM_train)

        self.UCM = UCM
        self.ICM = ICM

        self.users_attr_mat = UCM.todok().toarray()
        self.items_attr_mat = ICM.todok().toarray()
        self.ratings = URM_train.todok()

        self._item_indices = np.arange(0, self.n_items, dtype=np.int32)



    def _compute_item_score(self, user_id_array, items_to_compute=None):

        if items_to_compute is None:
            item_indices = self._item_indices
        else:
            item_indices = items_to_compute

        item_scores = - np.ones((len(user_id_array), self.n_items)) * np.inf


        for user_index in range(len(user_id_array)):

            items = item_indices
            user_id = user_id_array[user_index]

            user_attr_input, user_id_input, item_id_input, item_attr_input = [], [], [], []

            for i in range(len(items)):
                item_id = items[i]
                user_attr_input.append(self.users_attr_mat[user_id])
                user_id_input.append([user_id])
                item_id_input.append([item_id])
                item_attr_input.append(self.items_attr_mat[item_id])

            user_attr_input_mat = np.array(user_attr_input)
            user_id_input_mat = np.array(user_id_input)
            item_id_input_mat = np.array(item_id_input)
            item_attr_input_mat = np.array(item_attr_input)

            if self.dataset_name != 'Tafeng':   # all model with one icm
                item_score_user = self.model.predict(
                    {
                        'user_attr_input': user_attr_input_mat,
                        'item_attr_input': item_attr_input_mat,
                        'user_id_input': user_id_input_mat,
                        'item_id_input': item_id_input_mat,
                        'permutation_matrix': self.get_permutation_batch(len(user_id_input_mat))
                    }, batch_size=100, verbose=0)
            else:   #tafeng model has different structure
                item_sub_class = item_attr_input_mat[:, 0]
                item_asset_price = item_attr_input_mat[:, 1:]
                item_score_user = self.model.predict(
                    {
                        'user_attr_input': user_attr_input_mat,
                        'item_sub_class_input': item_sub_class,
                        'item_asset_price_input': item_asset_price,
                        'user_id_input': user_id_input_mat,
                        'item_id_input': item_id_input_mat,
                        'permutation_matrix': self.get_permutation_batch(len(user_id_input_mat))
                    }, batch_size=100, verbose=0)

            item_score_user = np.array(item_score_user).ravel()

            if items_to_compute is not None:
                item_scores[user_index, item_indices] = item_score_user.ravel()
            else:
                item_scores[user_index, :] = item_score_user.ravel()

        return item_scores

    def _get_model(self, n_users, n_items, dataset_name, number_model, map_mode):
        if dataset_name == 'Movielens1M' and number_model == 0:
            model = mainMovieUserCnn.get_model_0(n_users, n_items)
        elif dataset_name == 'Movielens1M' and number_model == 1:
            model = mainMovieUserCnn.get_model_1(n_users, n_items)
        elif dataset_name == 'Movielens1M' and number_model == 2:
            model = mainMovieUserCnn.get_model_2(n_users, n_items, map_mode)
        elif dataset_name == 'Tafeng' and number_model == 0:
            model = mainTafengUserCnn.get_model_0(n_users, n_items)
        elif dataset_name == 'Tafeng' and number_model == 1:
            model = mainTafengUserCnn.get_model_1(n_users, n_items)
        elif dataset_name == 'Tafeng' and number_model == 2:
            model = mainTafengUserCnn.get_model_2(n_users, n_items, map_mode)
        else:
            print("Model number {} not defined for dataset {}\n\tReturn default model for Movielens1M...".format(number_model, dataset_name))
            model = mainMovieUserCnn.get_model_0(n_users, n_items)
        return model


    def fit(self, learning_rate=0.001,
            epochs=30,
            n_negative_sample=4,
            dataset_name='Movielens1M',
            number_model=2,
            verbose=1,
            plot_model=False,
            map_mode='all_map',
            permutation=None,
            temp_file_folder=None,
            **earlystopping_kwargs
            ):

        assert map_mode in self.__AVAILABLE_MAP_MODES, "{}: map mode not recognized, available values are {}".format(self.RECOMMENDER_NAME, self.__AVAILABLE_MAP_MODES)

        self.learning_rate = learning_rate
        self.num_epochs = epochs
        self.num_negatives = n_negative_sample
        self.dataset_name = dataset_name
        self.number_model = number_model
        self.plot_model = plot_model
        self.verbose = verbose
        self.current_epoch = 0
        self.map_mode = map_mode
        self.permutation = permutation

        self.temp_file_folder = self._get_unique_temp_folder(input_temp_file_folder=temp_file_folder)

        self.init_model()

        print("{}: Training...".format(self.RECOMMENDER_NAME))

        self._update_best_model()

        self._train_with_early_stopping(self.num_epochs,
                                        algorithm_name = self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        self.load_model(self.temp_file_folder, file_name="_best_model")

        print("{}: Tranining complete".format(self.RECOMMENDER_NAME))
        self._clean_temp_folder(temp_file_folder=self.temp_file_folder)

    def setup_permutation_matrix(self):

        embedding_size = 8      # for both movielens and tafeng
        embedding_range = np.arange(embedding_size)

        if self.permutation is None:
            self.permutation = embedding_range
        else:
            # check if the permutation array is correct
            assert len(self.permutation) == embedding_size,\
                'Invalid permutation array length, it must be equal to the hidden factor size ({}), {} passed instead'.format(
                    embedding_size, len(self.permutation))

            setdiff = np.setdiff1d(embedding_range, self.permutation)

            assert setdiff.shape[0] == 0,\
                'Invalid permutation array, not all indices are present, {} are missing'.format(str(setdiff))

        self._perm_mx = np.zeros((embedding_size, embedding_size), dtype=np.uint8)
        self._perm_mx[self.permutation, np.arange(self._perm_mx.shape[0])] = 1

    def get_permutation_batch(self, batch_size):
        return np.stack([self._perm_mx for _ in range(batch_size)])

    def init_model(self):

        self.setup_permutation_matrix()

        if self.verbose:
            print("{}: Init model for {} ...".format(self.RECOMMENDER_NAME, self.dataset_name))

        # load model
        self.model = self._get_model(self.n_users, self.n_items, self.dataset_name, self.number_model, self.map_mode)

        # compile model
        self.model.compile(optimizer=Adam(lr=self.learning_rate),
                           loss='binary_crossentropy',
                           metrics=['accuracy', 'mae'])

        if self.plot_model:
            utils.plot_model(self.model,
                             show_shapes=True,
                             to_file='CoupledCF_{}_model{}.png'.format(self.dataset_name, self.number_model))

        if self.verbose:
            self.model.summary()
            print("{}: Init model... done!".format(self.RECOMMENDER_NAME))




    def _run_epoch(self, num_epoch):
        self.current_epoch = num_epoch

        t_start = time.time()

        # generate training instances
        user_attr_input, user_id_input, item_attr_input, item_id_input, labels = get_train_instances(self.users_attr_mat, self.ratings, self.items_attr_mat, self.num_negatives)

        # train model
        if self.dataset_name != 'Tafeng':
            hist = self.model.fit(
                {
                    'user_attr_input': user_attr_input,
                    'item_attr_input': item_attr_input,
                    'user_id_input': user_id_input,
                    'item_id_input': item_id_input,
                    'permutation_matrix': self.get_permutation_batch(len(user_id_input))
                },
                labels, epochs=1, batch_size=256, verbose=self.verbose, shuffle=True)
        else:   #tafeng model has different structure
            item_sub_class = item_attr_input[:, 0]
            item_asset_price = item_attr_input[:, 1:]

            hist = self.model.fit(
                {
                    'user_attr_input': user_attr_input,
                    'item_sub_class_input': item_sub_class,
                    'item_asset_price_input': item_asset_price,
                    'user_id_input': user_id_input,
                    'item_id_input': item_id_input,
                    'permutation_matrix': self.get_permutation_batch(len(user_id_input))
                },
                labels, epochs=1, batch_size=256, verbose=self.verbose, shuffle=True)

        loss = hist.history['loss'][0]

        t_exe = time.time() - t_start

        print("{}: Epoch {}, loss {:.2E}, time: {:.0E}s".format(self.RECOMMENDER_NAME, self.current_epoch, loss, t_exe))


    def _prepare_model_for_validation(self):
        pass

    def _update_best_model(self):
        self.save_model(self.temp_file_folder, file_name="_best_model")


    def save_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {
                              'learning_rate':self.learning_rate,
                              'num_epochs':self.num_epochs,
                              'num_negatives':self.num_negatives,
                              'dataset_name':self.dataset_name,
                              'number_model':self.number_model,
                              'plot_model':self.plot_model,
                              'current_epoch':self.current_epoch,
                              'map_mode': self.map_mode,
                              'permutation': self.permutation,
                              'verbose':self.verbose,
                              }

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)
        #
        # self.model.save(folder_path + file_name + "_keras_model.h5")
        self.model.save_weights(folder_path + file_name + "_keras_model.h5", overwrite=True)

        self._print("Saving complete")


    def load_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Loading model from file '{}'".format(folder_path + file_name))

        dataIO = DataIO(folder_path=folder_path)
        data_dict = dataIO.load_data(file_name=file_name)

        for attrib_name in data_dict.keys():
             self.__setattr__(attrib_name, data_dict[attrib_name])

        print("{}: Loading keras model".format(self.RECOMMENDER_NAME))

        self.init_model()

        # self.model = models.load_model(folder_path + file_name + "_keras_model.h5")
        self.model.load_weights(folder_path + file_name + "_keras_model.h5", by_name=True)

        self._print("Loading complete")












