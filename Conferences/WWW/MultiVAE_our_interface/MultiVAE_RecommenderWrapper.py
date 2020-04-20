#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30/10/18

@author: Maurizio Ferrari Dacrema
"""



import os
import shutil
import sys

import numpy as np
from scipy import sparse


import seaborn as sn
sn.set()


import tensorflow as tf


from Conferences.WWW.MultiVAE_our_interface.MultiVae_Dae import MultiDAE, MultiVAE


from Base.BaseRecommender import BaseRecommender
from Base.BaseTempFolder import BaseTempFolder
from Base.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Base.DataIO import DataIO


class Mult_VAE_RecommenderWrapper(BaseRecommender, Incremental_Training_Early_Stopping, BaseTempFolder):

    RECOMMENDER_NAME = "Mult_VAE_RecommenderWrapper"



    def __init__(self, URM_train):
        super(Mult_VAE_RecommenderWrapper, self).__init__(URM_train)


    def _compute_item_score(self, user_id_array, items_to_compute = None):

        URM_train_user_slice = self.URM_train[user_id_array]

        if sparse.isspmatrix(URM_train_user_slice):
            URM_train_user_slice = URM_train_user_slice.toarray()

        URM_train_user_slice = URM_train_user_slice.astype('float32')

        item_scores_to_compute = self.sess.run(self.logits_var, feed_dict={self.vae.input_ph: URM_train_user_slice})

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.n_items)) * np.inf
            item_scores[:, items_to_compute] = item_scores_to_compute[:, items_to_compute]
        else:
            item_scores = item_scores_to_compute

        return item_scores



    def fit(self,
            epochs = 100,
            batch_size = 500,
            total_anneal_steps = 200000,
            anneal_cap = 0.2,
            p_dims = None,
            temp_file_folder=None,
            **earlystopping_kwargs):


        self.temp_file_folder = self._get_unique_temp_folder(input_temp_file_folder=temp_file_folder)


        self.n_users, self.n_items = self.URM_train.shape

        self.batch_size = batch_size
        self.total_anneal_steps = total_anneal_steps
        self.anneal_cap = anneal_cap
        self.batches_per_epoch = int(np.ceil(float(self.n_users) / batch_size))

        self.update_count = 0.0

        if p_dims is None:
            self.p_dims = [200, 600, self.n_items]
        else:
            self.p_dims = p_dims



        tf.reset_default_graph()


        self.vae = MultiVAE(self.p_dims, lam=0.0, random_seed=98765)
        self.saver, self.logits_var, self.loss_var, self.train_op_var, self.merged_var = self.vae.build_graph()



        arch_str = "I-%s-I" % ('-'.join([str(d) for d in self.vae.dims[1:-1]]))

        self.log_dir = self.temp_file_folder + 'log/VAE_anneal{}K_cap{:1.1E}/{}'.format(
            total_anneal_steps/1000, anneal_cap, arch_str)

        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)

        print("Mult_VAE_RecommenderWrapper: log directory: %s" % self.log_dir)

        self.summary_writer = tf.summary.FileWriter(self.log_dir, graph=tf.get_default_graph())

        self.chkpt_dir = self.temp_file_folder + 'chkpt/VAE_anneal{}K_cap{:1.1E}/{}'.format(
            total_anneal_steps/1000, anneal_cap, arch_str)

        if not os.path.isdir(self.chkpt_dir):
            os.makedirs(self.chkpt_dir)

        print("Mult_VAE_RecommenderWrapper: checkpoint directory: %s" % self.chkpt_dir)


        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self._update_best_model()

        self._train_with_early_stopping(epochs,
                                        algorithm_name = self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)


        self.sess.close()

        self.load_model(self.temp_file_folder, file_name="_best_model")


        self._clean_temp_folder(temp_file_folder=self.temp_file_folder)




    def _prepare_model_for_validation(self):
        pass


    def _update_best_model(self):
        self.save_model(self.temp_file_folder, file_name="_best_model")


    def _run_epoch(self, num_epoch):

        user_index_list_train = list(range(self.n_users))

        np.random.shuffle(user_index_list_train)

        # train for one epoch
        for bnum, st_idx in enumerate(range(0, self.n_users, self.batch_size)):

            end_idx = min(st_idx + self.batch_size, self.n_users)
            X = self.URM_train[user_index_list_train[st_idx:end_idx]]

            if sparse.isspmatrix(X):
                X = X.toarray()
            X = X.astype('float32')

            if self.total_anneal_steps > 0:
                anneal = min(self.anneal_cap, 1. * self.update_count / self.total_anneal_steps)
            else:
                anneal = self.anneal_cap

            feed_dict = {self.vae.input_ph: X,
                         self.vae.keep_prob_ph: 0.5,
                         self.vae.anneal_ph: anneal,
                         self.vae.is_training_ph: 1}
            self.sess.run(self.train_op_var, feed_dict=feed_dict)

            if bnum % 100 == 0:
                summary_train = self.sess.run(self.merged_var, feed_dict=feed_dict)
                self.summary_writer.add_summary(summary_train,
                                           global_step=num_epoch * self.batches_per_epoch + bnum)

            self.update_count += 1








    def save_model(self, folder_path, file_name = None):

        #https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        saver = tf.train.Saver()
        saver.save(self.sess, folder_path + file_name + "_session")


        data_dict_to_save = {"n_users": self.n_users,
                              "n_items": self.n_items,
                              "batch_size": self.batch_size,
                              "total_anneal_steps": self.total_anneal_steps,
                              "anneal_cap": self.anneal_cap,
                              "update_count": self.update_count,
                              "p_dims": self.p_dims,
                              "batches_per_epoch": self.batches_per_epoch,
                              "log_dir": self.log_dir,
                              "chkpt_dir": self.chkpt_dir,

                              }

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)


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
        self.vae = MultiVAE(self.p_dims, lam=0.0)
        self.saver, self.logits_var, self.loss_var, self.train_op_var, self.merged_var = self.vae.build_graph()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.saver.restore(self.sess, folder_path + file_name + "_session")

        self.summary_writer = tf.summary.FileWriter(self.log_dir, graph=tf.get_default_graph())

        self._print("Loading complete")







