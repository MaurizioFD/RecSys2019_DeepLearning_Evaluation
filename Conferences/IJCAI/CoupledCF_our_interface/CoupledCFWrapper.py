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
from keras import utils, models
import numpy as np

from Conferences.IJCAI.CoupledCF_our_interface import mainMovieUserCnn
from Conferences.IJCAI.CoupledCF_our_interface import mainTafengUserCnn
from Conferences.IJCAI.CoupledCF_our_interface.mainMovieUserCnn import get_train_instances
from Base.BaseTempFolder import BaseTempFolder

class CoupledCF_RecommenderWrapper(BaseRecommender, Incremental_Training_Early_Stopping, BaseTempFolder):


    RECOMMENDER_NAME = "CoupledCF_RecommenderWrapper"


    def __init__(self, URM_train, UCM, ICM):

        super(CoupledCF_RecommenderWrapper, self).__init__(URM_train)

        self.UCM = UCM

        self.ICM = ICM

        #load data from original article file
        #self.path = "Conferences/IJCAI/CoupledCF_original/ml-1m/"
        #_, self.users_attr_mat = load_user_attributes(self.path)
        #_, self.items_attr_mat = load_itemGenres_as_matrix(self.path)
        #self.ratings = load_rating_train_as_matrix(self.path)


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

            items = items_to_compute
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

            if self.dataset_name != 'Tafeng': # all model with one icm
                item_score_user = self.model.predict([user_attr_input_mat, item_attr_input_mat, user_id_input_mat, item_id_input_mat], batch_size=100, verbose=0)
            else: #tafeng model has different structure
                item_sub_class = item_attr_input_mat[:, 0]
                item_asset_price = item_attr_input_mat[:, 1:]
                item_score_user = self.model.predict([user_attr_input_mat, item_sub_class, item_asset_price, user_id_input_mat, item_id_input_mat],batch_size=100, verbose=0)

            item_score_user = np.array(item_score_user).ravel()

            if items_to_compute is not None:
                item_scores[user_index, item_indices] = item_score_user.ravel()
            else:
                item_scores[user_index, :] = item_score_user.ravel()

        return item_scores


    def _get_model(self, n_users, n_items, dataset_name, number_model):
        if dataset_name == 'Movielens1M' and number_model == 0:
            model = mainMovieUserCnn.get_model_0(n_users, n_items)
        elif dataset_name == 'Movielens1M' and number_model == 1:
            model = mainMovieUserCnn.get_model_1(n_users, n_items)
        elif dataset_name == 'Movielens1M' and number_model == 2:
            model = mainMovieUserCnn.get_model_2(n_users, n_items)
        elif dataset_name == 'Tafeng' and number_model == 0:
            model = mainTafengUserCnn.get_model_0(n_users, n_items)
        elif dataset_name == 'Tafeng' and number_model == 1:
            model = mainTafengUserCnn.get_model_1(n_users, n_items)
        elif dataset_name == 'Tafeng' and number_model == 2:
            model = mainTafengUserCnn.get_model_2(n_users, n_items)
        else:
            print("Model number {} not defined for dataset {}\n\tReturn default model for Movielens1M...".format(number_model, dataset_name))
            model = mainMovieUserCnn.get_model_0(n_users, n_items)

        return model


    def save_weights(self):
        self.model.save_weights('Pretrain/CoupledCF/{}_model{}_epoch{}.h5'.format(self.dataset_name, self.number_model, self.current_epoch), overwrite=True)


    def fit(self, learning_rate=0.001,
            epochs=30,
            n_negative_sample=4,
            dataset_name='Movielens1M',
            number_model=0,
            verbose=1,
            plot_model=True,
            temp_file_folder=None,
            **earlystopping_kwargs
            ):


        self.learning_rate = learning_rate
        self.num_epochs = epochs
        self.num_negatives = n_negative_sample
        self.dataset_name = dataset_name
        self.number_model = number_model
        self.plot_model = plot_model
        self.verbose = verbose
        self.current_epoch = 0

        self.temp_file_folder = self._get_unique_temp_folder(input_temp_file_folder=temp_file_folder)

        print("{}: Init model for {} ...".format(self.RECOMMENDER_NAME, self.dataset_name))

        # load model
        self.model = self._get_model(self.n_users, self.n_items, self.dataset_name, self.number_model)

        # compile model
        self.model.compile(optimizer=Adam(lr=self.learning_rate),
                           loss='binary_crossentropy',
                           metrics=['accuracy', 'mae'])

        if self.plot_model:
            utils.plot_model(self.model,
                             show_shapes=True,
                             to_file='CoupledCF_{}_model{}.png'.format(self.dataset_name, self.number_model))

        if self.verbose > 1:
            self.model.summary()

        print("{}: Init model... done!".format(self.RECOMMENDER_NAME))

        print("{}: Training...".format(self.RECOMMENDER_NAME))

        self._update_best_model()

        self._train_with_early_stopping(self.num_epochs,
                                        algorithm_name = self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        self.load_model(self.temp_file_folder, file_name="_best_model")

        self._print("Training complete")
        self._clean_temp_folder(temp_file_folder=self.temp_file_folder)


    def _run_epoch(self, num_epoch):
        self.current_epoch = num_epoch

        t_start = time.time()

        # generate training instances
        user_attr_input, user_id_input, item_attr_input, item_id_input, labels = get_train_instances(self.users_attr_mat, self.ratings, self.items_attr_mat, self.num_negatives)

        # train model
        if self.dataset_name != 'Tafeng':
            hist = self.model.fit([user_attr_input, item_attr_input, user_id_input, item_id_input],
                                  labels, epochs=1, batch_size=256, verbose=self.verbose, shuffle=True)
        else: #tafeng model has different structure
            item_sub_class = item_attr_input[:, 0]
            item_asset_price = item_attr_input[:, 1:]

            hist = self.model.fit([user_attr_input, item_sub_class, item_asset_price, user_id_input, item_id_input],
                                  labels, epochs=1, batch_size=256, verbose=self.verbose, shuffle=True)

        loss = hist.history['loss'][0]

        t_exe = time.time() -t_start

        print("{}: Epoch {}, loss {:.2E}, time: {:.0E}s".format(self.RECOMMENDER_NAME, self.current_epoch, loss, t_exe))

        #original evaluation
        #testRatings = load_rating_file_as_list(path=self.path)
        #testNegatives = load_negative_file(path=self.path)
        #(hits, ndcgs) = evaluate_model(self.model, testRatings, testNegatives,self.users_attr_mat, self.items_attr_mat, 10, 1)
        #hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
        #print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f'% (num_epoch, t_exe, hr, ndcg, loss))


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
                              'verbose':self.verbose,
                              }

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)

        self.model.save(folder_path + file_name + "_keras_model.h5")

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

        self.model = models.load_model(folder_path + file_name + "_keras_model.h5")

        self._print("Loading complete")












