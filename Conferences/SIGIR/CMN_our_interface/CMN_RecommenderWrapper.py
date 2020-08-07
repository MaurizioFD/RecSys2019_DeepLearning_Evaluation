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
from collections import defaultdict
from tqdm import tqdm
import tensorflow as tf
import os, copy, shutil

from Conferences.SIGIR.CMN_github.util.cmn import CollaborativeMemoryNetwork
from Conferences.SIGIR.CMN_github.util.gmf import PairwiseGMF
from Conferences.SIGIR.CMN_github.util.helper import BaseConfig



class CMN_Config(object):

    def __init__(self, logdir,
                 embed_size,
                 batch_size,
                 hops,
                 l2,
                 user_count,
                 item_count,
                 optimizer,
                 neg_count,
                 optimizer_params,
                 learning_rate,
                 pretrain,
                 max_neighbors = -1
                 ):
        super(CMN_Config, self).__init__()

        self.logdir = logdir
        self.save_directory = self.logdir
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.hops = hops
        self.l2 = l2
        self.user_count = user_count
        self.item_count = item_count
        self.optimizer = optimizer
        self.tol = 1e-5
        self.neg_count = neg_count
        self.optimizer_params = optimizer_params
        self.grad_clip = 5.0
        self.decay_rate = 0.9
        self.learning_rate = learning_rate
        self.pretrain = pretrain
        self.max_neighbors = max_neighbors


    def get_deepcopy(self):

        return CMN_Config(**self.get_dict())


    def get_dict(self):

        attribute_list = [ "logdir",
                 "embed_size",
                 "batch_size",
                 "hops",
                 "l2",
                 "user_count",
                 "item_count",
                 "optimizer",
                 "neg_count",
                 "optimizer_params",
                 "learning_rate",
                 "pretrain",
                 "max_neighbors"]

        dictionary = {}

        for attribute in attribute_list:
            dictionary[attribute] = copy.deepcopy(self.__getattribute__(attribute))

        return dictionary



class EvaluatorModelLoss(object):

    def __init__(self):
        pass

    def evaluateRecommender(self, recommender_object, parameter_dictionary = None):

        average_loss = recommender_object.model._average_loss_evaluation

        result_dict = {
            "all": {"_average_loss_evaluation": - average_loss}
        }

        return result_dict, str("Average_loss: {}".format(average_loss))





class CMN_RecommenderWrapper(BaseRecommender, Incremental_Training_Early_Stopping, BaseTempFolder):


    RECOMMENDER_NAME = "CMN_RecommenderWrapper"

    def __init__(self, URM_train):
        super(CMN_RecommenderWrapper, self).__init__(URM_train)

        self.n_users, self.n_items = self.URM_train.shape

        self._item_indices = np.arange(0, self.n_items, dtype=np.int32)
        self._user_ones_vector = np.ones_like(self._item_indices)

        # Neighborhoods
        self.user_items = defaultdict(set)
        self.item_users = defaultdict(set)
        self.item_users_list = defaultdict(list)

        self.URM_train = sps.coo_matrix(self.URM_train)

        for interaction_index in range(self.URM_train.nnz):

            user_idx = self.URM_train.row[interaction_index]
            item_idx = self.URM_train.col[interaction_index]

            self.user_items[user_idx].add(item_idx)
            self.item_users[item_idx].add(user_idx)
            # Get a list version so we do not need to perform type casting
            self.item_users_list[item_idx].append(user_idx)

        self.URM_train = sps.csr_matrix(self.URM_train)




    def _compute_item_score(self, user_id_array, items_to_compute=None):

        if items_to_compute is None:
            item_indices = self._item_indices
        else:
            item_indices = items_to_compute

        item_scores = - np.ones((len(user_id_array), self.n_items)) * np.inf

        input_user_handle = self.model.input_users
        input_item_handle = self.model.input_items
        neighborhood = self.item_users_list
        input_neighborhood_handle = self.model.input_neighborhoods
        input_neighborhood_length_handle = self.model.input_neighborhood_lengths
        score_op = self.model.score


        for user_index in range(len(user_id_array)):

            user_id = user_id_array[user_index]

            # The prediction requires a list of two arrays user_id, item_id of equal length
            # To compute the recommendations for a single user, we must provide its intex as many times as the
            # number of items

            feed = {
                #input_user_handle: self._user_ones_vector*user_id,
                input_user_handle: np.ones(len(item_indices)) * user_id,
                input_item_handle: item_indices,
            }

            if neighborhood is not None:
                # neighborhoods, neighborhood_length = np.zeros((self.n_items, self.cmn_config.max_neighbors),
                #                                               dtype=np.int32), np.ones(self.n_items, dtype=np.int32)
                neighborhoods, neighborhood_length = np.zeros((len(item_indices), self.cmn_config.max_neighbors),
                                                              dtype=np.int32), np.ones(len(item_indices), dtype=np.int32)

                for _idx, item in enumerate(item_indices):
                    _len = min(len(neighborhood[item]), self.cmn_config.max_neighbors)
                    if _len > 0:
                        neighborhoods[_idx, :_len] = neighborhood[item][:_len]
                        neighborhood_length[_idx] = _len
                    else:
                        neighborhoods[_idx, :1] = user_id
                feed.update({
                    input_neighborhood_handle: neighborhoods,
                    input_neighborhood_length_handle: neighborhood_length
                })


            item_score_user = self.sess.run(score_op, feed)

            if items_to_compute is not None:
                item_scores[user_index, item_indices] = item_score_user.ravel()
            else:
                item_scores[user_index, :] = item_score_user.ravel()

        return item_scores


    def get_early_stopping_final_epochs_dict(self):
        """
        This function returns a dictionary to be used as optimal parameters in the .fit() function
        It provides the flexibility to deal with multiple early-stopping in a single algorithm
        e.g. in NeuMF there are three model components each with its own optimal number of epochs
        the return dict would be {"epochs": epochs_best_neumf, "epochs_gmf": epochs_best_gmf, "epochs_mlp": epochs_best_mlp}
        :return:
        """

        return {"epochs": self.epochs_best, "epochs_gmf": self.epochs_best_gmf}




    def fit(self,
            epochs = 100,
            epochs_gmf = 100,
            batch_size = 128,
            embed_size = 50,
            hops = 2,
            neg_samples = 4,
            reg_l2_cmn = 1e-1,
            reg_l2_gmf=1e-4,
            pretrain = True,
            learning_rate = 1e-3,
            verbose = False,
            temp_file_folder = None,
            **earlystopping_kwargs):



        #assert learner in ["adagrad", "adam", "rmsprop", "sgd"]

        self.verbose = verbose

        self.temp_file_folder = self._get_unique_temp_folder(input_temp_file_folder=temp_file_folder)


        # it is the max number of interaction an item has received
        self.URM_train = sps.csc_matrix(self.URM_train)
        max_neighbors = max(np.ediff1d(self.URM_train.indptr))

        self.URM_train = sps.csr_matrix(self.URM_train)
        self.URM_train_coo = sps.coo_matrix(self.URM_train)



        self.cmn_config = CMN_Config(logdir = self.temp_file_folder,
                                     embed_size= embed_size,
                                     batch_size = batch_size,
                                     hops = hops,
                                     l2 = reg_l2_cmn,
                                     optimizer = "rmsprop",
                                     neg_count= neg_samples,
                                     item_count = self.n_items,
                                     user_count = self.n_users,
                                     optimizer_params = {"decay": 0.9, "momentum": 0.9},
                                     learning_rate = learning_rate,
                                     pretrain = self.temp_file_folder,
                                     max_neighbors = max_neighbors)

        self.cmn_config_clone = self.cmn_config.get_deepcopy()

        self.pairwiseGMF_config = CMN_Config(logdir = self.temp_file_folder,
                                             embed_size= embed_size,
                                             batch_size = batch_size,
                                             hops = hops,
                                             l2 = reg_l2_gmf,
                                             optimizer = "adam",
                                             neg_count= neg_samples,
                                             item_count = self.n_items,
                                             user_count = self.n_users,
                                             optimizer_params = {"decay": 0.9, "momentum": 0.9},
                                             learning_rate = learning_rate,
                                             pretrain = self.temp_file_folder)





        # Build model
        tf.reset_default_graph()

        self.model = PairwiseGMF(self.pairwiseGMF_config)
        self.sv = tf.train.Supervisor(logdir=None, save_model_secs=0, save_summaries_secs=0)
        self.sess = self.sv.prepare_or_wait_for_session(
                    config=tf.ConfigProto(gpu_options=tf.GPUOptions(
                        per_process_gpu_memory_fraction=0.1,
                        allow_growth=True)))


        self._run_epoch = self._run_epoch_GMF
        self._update_best_model = self._update_best_model_GMF

        print("CMN_RecommenderWrapper: Pretraining GMF...")

        self._update_best_model()
        validation_metric_loss_GMF = "_average_loss_evaluation"
        evaluator_object_loss_GMF = EvaluatorModelLoss()

        # Check if earlystopping has to be applied for GMF
        earlystopping_kwargs_gmf = earlystopping_kwargs.copy()
        if "evaluator_object" in earlystopping_kwargs and earlystopping_kwargs_gmf["evaluator_object"] is not None:
            earlystopping_kwargs_gmf["validation_metric"] = validation_metric_loss_GMF
            earlystopping_kwargs_gmf["evaluator_object"] = evaluator_object_loss_GMF

        self._train_with_early_stopping(epochs_gmf,
                                        algorithm_name = self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs_gmf)

        self.epochs_best_gmf = self.epochs_best

        print("CMN_RecommenderWrapper: Pretraining complete")









        # Build model
        tf.reset_default_graph()

        self.model = CollaborativeMemoryNetwork(self.cmn_config)

        self.sv = tf.train.Supervisor(logdir=None, save_model_secs=60 * 10,
                                        save_summaries_secs=0)

        self.sess = self.sv.prepare_or_wait_for_session(config=tf.ConfigProto(
                    gpu_options=tf.GPUOptions(allow_growth=True)))


        self.sess.graph._unsafe_unfinalize()

        self.sess.run([
            self.model.user_memory.embeddings.assign(self._GMF_user_embed*0.5),
            self.model.item_memory.embeddings.assign(self._GMF_item_embed*0.5)])






        self._run_epoch = self._run_epoch_CMN
        self._update_best_model = self._update_best_model_CMN

        print("CMN_RecommenderWrapper: Training CMN...")

        self._update_best_model()

        self._train_with_early_stopping(epochs,
                                        algorithm_name = self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)


        self._print("Training complete")


        self.sess.close()

        self.sess = self.sv.prepare_or_wait_for_session(config=tf.ConfigProto(
                    gpu_options=tf.GPUOptions(allow_growth=True)))


        self.load_model(self.temp_file_folder, file_name="best_model")

        self._clean_temp_folder(temp_file_folder=self.temp_file_folder)



    def _prepare_model_for_validation(self):
        pass


    def _update_best_model_CMN(self):
        self.save_model(self.temp_file_folder, file_name="best_model")


    def _update_best_model_GMF(self):

        self._GMF_user_embed, self._GMF_item_embed, self._GMF_v = self.sess.run(
            [self.model.user_memory.embeddings, self.model.item_memory.embeddings, self.model.v.w])

        self._GMF_user_embed = self._GMF_user_embed.copy()
        self._GMF_item_embed = self._GMF_item_embed.copy()
        self._GMF_v = self._GMF_v.copy()


    def _run_epoch_CMN(self, currentEpoch):

        train_data_iterator = self.get_train_data_iterator(neighborhood = True)

        progress = enumerate(train_data_iterator)

        if self.verbose:
            progress = tqdm(progress,
                            dynamic_ncols=True,
                            total=(self.URM_train.nnz * self.cmn_config.neg_count) // self.cmn_config.batch_size)

        loss = []
        for k, example in progress:
            ratings, pos_neighborhoods, pos_neighborhood_length, \
            neg_neighborhoods, neg_neighborhood_length = example
            feed = {
                self.model.input_users: ratings[:, 0],
                self.model.input_items: ratings[:, 1],
                self.model.input_items_negative: ratings[:, 2],
                self.model.input_neighborhoods: pos_neighborhoods,
                self.model.input_neighborhood_lengths: pos_neighborhood_length,
                self.model.input_neighborhoods_negative: neg_neighborhoods,
                self.model.input_neighborhood_lengths_negative: neg_neighborhood_length
            }
            batch_loss, _ = self.sess.run([self.model.loss, self.model.train], feed)
            loss.append(batch_loss)

            if self.verbose:
                progress.set_description(u"[{}] Loss: {:,.4f} » » » » ".format(currentEpoch, batch_loss))

        print("CMN_RecommenderWrapper: Epoch {}: Avg Loss/Batch {:<20,.6f}".format(currentEpoch, np.mean(loss)))




    def _run_epoch_GMF(self, currentEpoch):

        train_data_iterator = self.get_train_data_iterator(neighborhood=False)

        progress = enumerate(train_data_iterator)

        if self.verbose:
            progress = tqdm(progress,
                            dynamic_ncols=True,
                            total=(self.URM_train.nnz * self.cmn_config.neg_count) // self.cmn_config.batch_size)
        loss = []

        for k, example in progress:
            feed = {
                self.model.input_users: example[:, 0],
                self.model.input_items: example[:, 1],
                self.model.input_items_negative: example[:, 2],
            }
            batch_loss, _ = self.sess.run([self.model.loss, self.model.train], feed)
            loss.append(batch_loss)

            if self.verbose:
                progress.set_description(u"[{}] Loss: {:,.4f} » » » » ".format(currentEpoch, batch_loss))

        self.model._average_loss_evaluation = np.mean(loss)

        print("CMN_RecommenderWrapper: Epoch {}: Avg Loss/Batch {:<20,.6f}".format(currentEpoch, self.model._average_loss_evaluation))









    def get_train_data_iterator(self, neighborhood = True):
        # Allocate inputs
        batch = np.zeros((self.cmn_config.batch_size, 3), dtype=np.uint32)
        pos_neighbor = np.zeros((self.cmn_config.batch_size, self.cmn_config.max_neighbors), dtype=np.int32)
        pos_length = np.zeros(self.cmn_config.batch_size, dtype=np.int32)
        neg_neighbor = np.zeros((self.cmn_config.batch_size, self.cmn_config.max_neighbors), dtype=np.int32)
        neg_length = np.zeros(self.cmn_config.batch_size, dtype=np.int32)

        # Shuffle index
        self._train_index = np.arange(self.URM_train.nnz, dtype=np.uint)
        np.random.shuffle(self._train_index)

        idx = 0
        for interaction_index in self._train_index:

            user_idx = self.URM_train_coo.row[interaction_index]
            item_idx = self.URM_train_coo.col[interaction_index]

            # TODO: set positive values outside of for loop
            for _ in range(self.cmn_config.neg_count):
                neg_item_idx = self._sample_negative_item(user_idx)
                batch[idx, :] = [user_idx, item_idx, neg_item_idx]

                # Get neighborhood information
                if neighborhood:
                    if len(self.item_users[item_idx]) > 0:
                        pos_length[idx] = len(self.item_users[item_idx])
                        pos_neighbor[idx, :pos_length[idx]] = self.item_users_list[item_idx]
                    else:
                        # Length defaults to 1
                        pos_length[idx] = 1
                        pos_neighbor[idx, 0] = item_idx

                    if len(self.item_users[neg_item_idx]) > 0:
                        neg_length[idx] = len(self.item_users[neg_item_idx])
                        neg_neighbor[idx, :neg_length[idx]] = self.item_users_list[neg_item_idx]
                    else:
                        # Length defaults to 1
                        neg_length[idx] = 1
                        neg_neighbor[idx, 0] = neg_item_idx

                idx += 1
                # Yield batch if we filled queue
                if idx == self.cmn_config.batch_size:
                    if neighborhood:
                        max_length = max(neg_length.max(), pos_length.max())
                        yield batch, pos_neighbor[:, :max_length], pos_length, \
                              neg_neighbor[:, :max_length], neg_length
                        pos_length[:] = 1
                        neg_length[:] = 1
                    else:
                        yield batch
                    # Reset
                    idx = 0

        # Provide remainder
        if idx > 0:
            if neighborhood:
                max_length = max(neg_length[:idx].max(), pos_length[:idx].max())
                yield batch[:idx], pos_neighbor[:idx, :max_length], pos_length[:idx], \
                      neg_neighbor[:idx, :max_length], neg_length[:idx]
            else:
                yield batch[:idx]




    def _sample_negative_item(self, user_id):
        """
        Uniformly sample a negative item
        """
        if user_id > self.n_users:
            raise ValueError("Trying to sample user id: {} > user count: {}".format(
                user_id, self.n_users))

        n = self._sample_item()
        positive_items = self.user_items[user_id]

        if len(positive_items) >= self.n_items:
            raise ValueError("The User has rated more items than possible %s / %s" % (
                len(positive_items), self.n_items))
        while n in positive_items or n not in self.item_users:
            n = self._sample_item()
        return n


    def _sample_item(self):
        """
        Draw an item uniformly
        """
        return np.random.randint(0, self.n_items)








    def save_model(self, folder_path, file_name = None):

        #https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {"n_users": self.n_users,
                              "n_items": self.n_items,
                              "cmn_config_dict": self.cmn_config_clone.get_dict()
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

            if attrib_name == "cmn_config_dict":
                self.cmn_config = CMN_Config(**data_dict[attrib_name])
                self.cmn_config_clone = self.cmn_config.get_deepcopy()

            self.__setattr__(attrib_name, data_dict[attrib_name])


        tf.reset_default_graph()

        self.model = CollaborativeMemoryNetwork(self.cmn_config)

        self.sv = tf.train.Supervisor(logdir=None, save_model_secs=60 * 10,
                                        save_summaries_secs=0)

        self.sess = self.sv.prepare_or_wait_for_session(config=tf.ConfigProto(
                    gpu_options=tf.GPUOptions(allow_growth=True)))

        self.sess.graph._unsafe_unfinalize()
        saver = tf.train.Saver()

        saver.restore(self.sess, folder_path + file_name + "_session")


        self._print("Loading complete")





