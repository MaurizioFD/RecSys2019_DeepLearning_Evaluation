#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19/02/19

@author: Simone Boglio
"""


from Base.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from Base.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping

import tensorflow as tf
import os

from Conferences.IJCAI.NeuRec_our_interface.UNeuRec import UNeuRec


class UNeuRec_RecommenderWrapper(BaseMatrixFactorizationRecommender, Incremental_Training_Early_Stopping):


    RECOMMENDER_NAME = "UNeuRec_RecommenderWrapper"

    def __init__(self, URM_train):

        super(UNeuRec_RecommenderWrapper, self).__init__(URM_train)

        # tf
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True



    def fit(self,
            num_neurons=300,
            num_factors=50,
            dropout_percentage=0.03,
            learning_rate=1e-4,
            regularization_rate=0.1,
            epochs=1000,
            batch_size=1024,
            display_epoch=None,
            display_step=None,
            verbose=True,
            temp_file_folder = None,
            **earlystopping_kwargs):


        self.num_neurons = num_neurons
        self.num_factors = num_factors
        self.dropout_percentage = dropout_percentage
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.display_epoch = display_epoch
        self.display_step = display_step
        self.regularization_rate = regularization_rate
        self.verbose = verbose


        print("{}: Init model...".format(self.RECOMMENDER_NAME))
        tf.reset_default_graph()

        # open session tensorflow
        self.sess = tf.Session(config=self.config)

        self.model = UNeuRec(tf_session=self.sess,
                             num_neurons=self.num_neurons,
                             num_factors=self.num_factors,
                             dropout_percentage=self.dropout_percentage,
                             learning_rate=self.learning_rate,
                             regularization_rate=self.regularization_rate,
                             max_epochs=self.epochs,
                             batch_size=self.batch_size,
                             display_epoch=self.display_epoch,
                             display_step=self.display_step,
                             verbose=self.verbose)

        self.model.fit(self.URM_train)


        print("{}: Init model... done!".format(self.RECOMMENDER_NAME))

        print("{}: Training...".format(self.RECOMMENDER_NAME))

        self._prepare_model_for_validation()
        self._update_best_model()

        self._train_with_early_stopping(self.epochs,
                                        algorithm_name = self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        # close session tensorflow
        self.sess.close()
        self.sess = tf.Session()

        self.USER_factors = self.USER_factors_best
        self.ITEM_factors = self.ITEM_factors_best

        self._print("Training complete")



    def _prepare_model_for_validation(self):
        self.USER_factors, self.ITEM_factors = self.model.get_factors()


    def _update_best_model(self):
        self.USER_factors_best = self.USER_factors.copy()
        self.ITEM_factors_best = self.ITEM_factors.copy()


    def _run_epoch(self, currentEpoch):
        loss= self.model.run_epoch()
        print("{}: Epoch {}, loss {:.2E}".format(self.RECOMMENDER_NAME,currentEpoch+1, loss))

    #
    # def save_model(self, folder_path, file_name = None):
    #
    #     import pickle
    #
    #     if file_name is None: file_name = self.RECOMMENDER_NAME
    #     self._print("Saving model in file '{}'".format(folder_path + file_name))
    #
    #     dictionary_to_save = {
    #                           'num_neurons':self.num_neurons,
    #                           'num_factors':self.num_factors,
    #                           'dropout_percentage':self.dropout_percentage,
    #                           'learning_rate':self.learning_rate,
    #                           'regularization_rate':self.regularization_rate,
    #                           'epochs':self.epochs,
    #                           'batch_size':self.batch_size,
    #                           'display_epoch':self.display_epoch,
    #                           'display_step':self.display_step,
    #                           'verbose':self.verbose,
    #                           }
    #
    #     pickle.dump(dictionary_to_save,
    #                 open(folder_path + file_name, "wb"),
    #                 protocol=pickle.HIGHEST_PROTOCOL)
    #
    #     saver = tf.train.Saver()
    #     saver.save(self.sess, folder_path + file_name + "_session")
    #     self._print("Saving complete")
    #
    #
    # def load_model(self, folder_path, file_name = None):
    #
    #     import pickle
    #
    #     if file_name is None: file_name = self.RECOMMENDER_NAME
    #     self._print("Loading model from file '{}'".format(folder_path + file_name))
    #
    #     data_dict = pickle.load(open(folder_path + file_name, "rb"))
    #
    #     for attrib_name in data_dict.keys():
    #          self.__setattr__(attrib_name, data_dict[attrib_name])
    #
    #     tf.reset_default_graph()
    #
    #     self.sess = tf.Session(config=self.config)
    #
    #     self.model = UNeuRec(tf_session=self.sess,
    #                          num_neurons=self.num_neurons,
    #                          num_factors=self.num_factors,
    #                          dropout_percentage=self.dropout_percentage,
    #                          learning_rate=self.learning_rate,
    #                          regularization_rate=self.regularization_rate,
    #                          max_epochs=self.epochs,
    #                          batch_size=self.batch_size,
    #                          display_epoch=self.display_epoch,
    #                          display_step=self.display_step,
    #                          verbose=self.verbose)
    #
    #
    #     self.model.fit(urm=self.URM_train)
    #
    #     saver = tf.train.Saver()
    #
    #     self.sess = tf.Session(config=self.config)
    #
    #     saver.restore(self.sess, folder_path + file_name + "_session")
    #
    #     self.model.sess = self.sess
    #
    #     self._print("Loading complete")
