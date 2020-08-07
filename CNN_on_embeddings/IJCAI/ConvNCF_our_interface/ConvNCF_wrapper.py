#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 25/02/19

@author: Simone Boglio
"""

import numpy as np
import CNN_on_embeddings.IJCAI.ConvNCF_our_interface.ConvNCF as ConvNCF
from Conferences.IJCAI.ConvNCF_our_interface.MFBPR_Wrapper import MFBPR_Wrapper
from Base.BaseTempFolder import BaseTempFolder
from Base.DataIO import DataIO
import os, platform

from Base.BaseRecommender import BaseRecommender
from Base.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping

class DatasetInterface:

    def __init__(self, URM_train):

        self.num_users, self.num_items = URM_train.shape

        self.trainMatrix = URM_train.todok()

        m_train = URM_train.tocsr()
        train_list = []
        for u in range(m_train.shape[0]):
            items = m_train.indices[m_train.indptr[u]:m_train.indptr[u + 1]].tolist()
            train_list.append(items)
        self.trainList = train_list

        # m_test = URM_test.tocsr()
        # test_list = []
        # for u in range(m_test.shape[0]):
        #     items = m_test.indices[m_test.indptr[u]:m_test.indptr[u + 1]].tolist()
        #     assert len(items) <= 1, 'atm is supported only leave one out'
        #     if len(items) == 0:  # user no test item,set to -1 to avoid error in bad sampling
        #         test_list.append([u, -1])
        #
        #
        # self.testRatings = test_list
        self.testRatings = None

        # # Ensure in no way test data gets used in the train. Create empty negative matrix
        # URM_negative = sps.csr_matrix(URM_train.shape)
        #
        # m_negative = URM_negative.tocsr()
        # negative_list = []
        # for u in range(m_negative.shape[0]):
        #     items = m_negative.indices[m_negative.indptr[u]:m_negative.indptr[u + 1]].tolist()
        #     negative_list.append(items)
        # self.testNegatives = negative_list

        self.testNegatives = None

        # assert len(self.testRatings) == len(self.testNegatives) and len(self.testNegatives) == self.num_users

        # assert len(self.testRatings) == self.num_users
        # print('assert ok')


class ArgsInterface:

    def __init__(self):
        pass



class ConvNCF_RecommenderWrapper(BaseRecommender, Incremental_Training_Early_Stopping, BaseTempFolder):

    RECOMMENDER_NAME = "ConvNCF_Wrapper"
    
    __AVAILABLE_MAP_MODES = ["all_map", "main_diagonal", "off_diagonal"]


    def __init__(self, URM_train):
        super(ConvNCF_RecommenderWrapper, self).__init__(URM_train)

        assert platform.system() != "Windows", "Current recommender is implemented using global variables and will not work under Windows"

        self._item_indices = np.arange(0, self.n_items, dtype=np.int32)


    def get_early_stopping_final_epochs_dict(self):
        """
        This function returns a dictionary to be used as optimal parameters in the .fit() function
        It provides the flexibility to deal with multiple early-stopping in a single algorithm
        e.g. in NeuMF there are three model componets each with its own optimal number of epochs
        the return dict would be {"epochs": epochs_best_neumf, "epochs_gmf": epochs_best_gmf, "epochs_mlp": epochs_best_mlp}
        :return:
        """

        return {"epochs": self.epochs_best, "epochs_MFBPR": self.epochs_best_MFBPR}



    def _compute_item_score(self, user_id_array, items_to_compute=None):

        if items_to_compute is None:
            item_indices = self._item_indices
        else:
            item_indices = items_to_compute

        item_scores = - np.ones((len(user_id_array), self.n_items)) * np.inf

        item_input = np.array(item_indices)
        item_input = item_input.reshape((item_input.shape[0], 1))

        for user_index in range(len(user_id_array)):

            user_id = user_id_array[user_index]

            # predictions = np.array(ConvNCF.compute_score_single_user(user_id, items_to_compute))

            user_input = np.full(item_input.shape[0], user_id)
            # item_input = item_input.reshape((item_input.shape[0], 1))
            user_input = user_input.reshape((item_input.shape[0], 1))

            feed_dict = {ConvNCF._model.user_input: user_input,
                         ConvNCF._model.item_input_pos: item_input,
                         ConvNCF._model.keep_prob: ConvNCF.TEST_KEEP_PROB}

            item_score_user = ConvNCF._sess.run(ConvNCF._model.output, feed_dict).flatten()

            # scores = np.ones(self.dataset.num_items) * (-np.inf)
            # scores[items_to_compute] = predictions
            # item_scores[user_index, :] = scores.ravel()

            if items_to_compute is not None:
                item_scores[user_index, item_indices] = item_score_user.ravel()
            else:
                item_scores[user_index, :] = item_score_user.ravel()


        return item_scores



    def fit(self,
            batch_size=512,
            epochs=1500,
            epochs_MFBPR = 500,
            load_pretrained_MFBPR_if_available = False,
            MF_latent_factors_folder = None,
            embedding_size=64,
            hidden_size=128,
            negative_sample_per_positive=1,
            negative_instances_per_positive=4,
            regularization_users_items=0.01,
            regularization_weights=10,
            regularization_filter_weights=1,
            learning_rate_embeddings=0.05,
            learning_rate_CNN=0.05,
            channel_size=[32,32,32,32,32,32],
            dropout=0.0,
            epoch_verbose=25,
            map_mode = "full_map",
            temp_file_folder = None,
            **earlystopping_kwargs
            ):
        
        
        assert map_mode in self.__AVAILABLE_MAP_MODES, "{}: map mode not recognized, available values are {}".format(self.RECOMMENDER_NAME, self.__AVAILABLE_MAP_MODES)
        

        self.temp_file_folder = self._get_unique_temp_folder(input_temp_file_folder=temp_file_folder)

        # initialize models
        print("{}: Init model...".format(self.RECOMMENDER_NAME))

        self.dataset = DatasetInterface(URM_train = self.URM_train)
        self.epochs_best_MFBPR = None

        self.map_mode = map_mode

        self.args = ArgsInterface()
        self.args.dataset = 'no_dataset_name'
        self.args.model = self.RECOMMENDER_NAME
        self.args.verbose = epoch_verbose
        self.args.batch_size = batch_size
        self.args.embed_size = embedding_size
        self.args.hidden_size = hidden_size
        self.args.dns = negative_sample_per_positive
        self.args.regs = [regularization_users_items, regularization_weights, regularization_filter_weights]
        self.args.task = 'no_task_name'
        self.args.num_neg = negative_instances_per_positive
        self.args.lr_embed = learning_rate_embeddings
        self.args.lr_net = learning_rate_CNN
        self.args.net_channel = channel_size
        self.args.pretrain = 1
        self.args.ckpt = 0
        self.args.train_auc = 0
        self.args.keep = 1-dropout
        self.args.path_logging = self.temp_file_folder


        # Pre train the weights for ConvNCF net
        if load_pretrained_MFBPR_if_available and os.path.isfile(MF_latent_factors_folder + "best_model_latent_factors.npy"):
            self.args.path_partial_results = MF_latent_factors_folder
            print("{}: MF_BPR_model found in '{}', skipping training!".format(self.RECOMMENDER_NAME, self.args.path_partial_results))

        else:
            print("{}: MF_BPR_model not found in '{}', training!".format(self.RECOMMENDER_NAME, self.args.path_partial_results))

            MF_BPR_model = MFBPR_Wrapper(self.URM_train)
            MF_BPR_model.fit(
                   batch_size=512,
                   epochs= epochs_MFBPR,
                   embed_size=embedding_size,
                   negative_sample_per_positive=negative_sample_per_positive,
                   regularization_users=0.01,
                   regularization_items=0.0,
                   learning_rate=0.05,
                   epoch_verbose=25,
                   train_auc_verbose=0,
                   path_partial_results= self.args.path_partial_results,
                    **earlystopping_kwargs,
                   )

            self.epochs_best_MFBPR = MF_BPR_model.epochs_best_MFBPR

            MF_BPR_model._dealloc_global_variables()
            self.args.path_partial_results = MF_latent_factors_folder



        ConvNCF.init_logging(self.args)
        ConvNCF.TRAIN_KEEP_PROB = self.args.keep
        ConvNCF.tf.reset_default_graph()

        self.model = ConvNCF.ConvNCF(self.dataset.num_users, self.dataset.num_items, self.args, map_mode = self.map_mode)
        self.model.build_graph()
        ConvNCF.initialize(self.model, self.dataset, self.args)
        self.sess = ConvNCF.get_session()

        print("{}: Init model... done!".format(self.RECOMMENDER_NAME))

        self._update_best_model()

        self._train_with_early_stopping(epochs_max=epochs,
                                        algorithm_name=self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        # close session tensorflow
        ConvNCF.close_session()


        self.sess = ConvNCF.tf.Session()

        self.load_model(self.temp_file_folder, file_name="_best_model")

        print("{}: Training complete".format(self.RECOMMENDER_NAME))

        self._clean_temp_folder(temp_file_folder=self.temp_file_folder)



    def _prepare_model_for_validation(self):
        pass


    def _update_best_model(self):
        self.save_model(self.temp_file_folder, file_name="_best_model")


    def _run_epoch(self, currentEpoch):

        batch_time, train_time = ConvNCF.run_epoch(model=self.model,
                                                   epoch_count=currentEpoch,
                                                   args=self.args,
                                                   dataset=self.dataset,
                                                   verbose=False,
                                                   original_evaluation=False)

        print("{}: Epoch: {} batch cost: {} train cost: {}".format(self.RECOMMENDER_NAME,currentEpoch, batch_time, train_time))



    def save_model(self, folder_path, file_name = None):


        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {"map_mode": self.map_mode,
                             **{'_args_{}'.format(key):value for (key,value) in vars(self.args).items()}}

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)

        saver = ConvNCF.tf.train.Saver()
        saver.save(ConvNCF._sess, folder_path + file_name + "_session")

        self._print("Saving complete")



    def load_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Loading model from file '{}'".format(folder_path + file_name))

        dataIO = DataIO(folder_path=folder_path)
        data_dict = dataIO.load_data(file_name=file_name)

        self.args = ArgsInterface()


        for attrib_name in data_dict.keys():

            if attrib_name.startswith("_args_"):
                data_dict_key = attrib_name
                attrib_name = attrib_name[len("_args_"):]
                setattr(self.args, attrib_name, data_dict[data_dict_key])

            else:
                self.__setattr__(attrib_name, data_dict[attrib_name])


        self.dataset = DatasetInterface(URM_train = self.URM_train)

        ConvNCF.tf.reset_default_graph()

        self.sess = ConvNCF.tf.Session()

        ConvNCF.TRAIN_KEEP_PROB = self.args.keep
        self.model = ConvNCF.ConvNCF(self.dataset.num_users, self.dataset.num_items, self.args, map_mode = self.map_mode)
        self.model.build_graph()
        ConvNCF.initialize(self.model, self.dataset, self.args)
        ConvNCF._model = self.model


        saver = ConvNCF.tf.train.Saver()
        self.sess = ConvNCF.tf.Session()
        saver.restore(self.sess, folder_path + file_name + "_session")

        ConvNCF._sess = self.sess

        self._print("Loading complete")




