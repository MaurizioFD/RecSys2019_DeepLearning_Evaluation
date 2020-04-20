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
from Base.BaseRecommender import BaseRecommender
from Base.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Base.DataIO import DataIO

class MFBPR_Wrapper(BaseRecommender, Incremental_Training_Early_Stopping):

    RECOMMENDER_NAME = "MF_BPR_Wrapper"


    def __init__(self, URM_train):
        super(MFBPR_Wrapper, self).__init__(URM_train)

        assert platform.system() != "Windows", "Current recommender is implemented using global variables and will not work under Windows"

        self._item_indices = np.arange(0, self.n_items, dtype=np.int32)

    def _print(self, string):
        print("{}: {}".format(self.RECOMMENDER_NAME, string))




    def _compute_item_score(self, user_id_array, items_to_compute = None):

        if items_to_compute is None:
            item_indices = self._item_indices
        else:
            item_indices = items_to_compute

        item_scores = - np.ones((len(user_id_array), self.n_items)) * np.inf


        for user_index in range(len(user_id_array)):

            user_id = user_id_array[user_index]

            user_input = np.full(len(item_indices), user_id, dtype='int32')[:, None]
            item_input = np.array(item_indices)[:, None]

            feed_dict = {MF_BPR._model.user_input: user_input,
                         MF_BPR._model.item_input_pos: item_input}

            item_score_user = self.sess.run(MF_BPR._model.output, feed_dict)


            if items_to_compute is not None:
                item_scores[user_index, item_indices] = item_score_user.ravel()
            else:
                item_scores[user_index, :] = item_score_user.ravel()

        return item_scores






    def fit(self,
            batch_size=512,
            epochs=500,
            embed_size=64,
            negative_sample_per_positive=1,
            regularization_users=0.01,
            regularization_items=0.0,
            learning_rate=0.05,
            epoch_evaluation=25,
            train_auc_verbose=0,
            path_partial_results = None,
            **earlystopping_kwargs
            ):

        assert path_partial_results is not None
        self.path_partial_results = path_partial_results

        self.args = ArgsInterface()
        self.args.dataset = 'no_dataset_name'
        self.args.model = self.RECOMMENDER_NAME
        self.args.verbose = epoch_evaluation
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

        self._print("Start training...")

        MF_BPR.tf.reset_default_graph()

        MF_BPR.init_logging(self.args)

        # initialize dataset
        self.dataset = DatasetInterface(URM_train=self.URM_train)#, URM_test=URM_test, URM_negative=URM_negative)

        # initialize models
        self.model_GMF = MF_BPR.GMF(self.dataset.num_users, self.dataset.num_items, self.args)
        self.model_GMF.build_graph()

        # MF_BPR.training(model=self.model_GMF, dataset=self.dataset, args=self.args)

        self.sess = MF_BPR.tf.Session()
        self.sess.run(MF_BPR.tf.global_variables_initializer())

        # sample the data
        self.samples = MF_BPR.sampling(self.dataset)


        self._update_best_model()

        self._train_with_early_stopping(epochs,
                                        algorithm_name = self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        self.epochs_best_MFBPR = self.epochs_best

        self._print("Tranining complete")

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
        self.save_model(self.path_partial_results, file_name="best_model")


    def _run_epoch(self, num_epoch):

        # initialize for training batches
        batches = MF_BPR.shuffle(self.samples, self.args.batch_size, self.dataset, self.model_GMF)  # , args.exclude_gtItem)

        # compute the accuracy before training
        # prev_batch = batches[0], batches[1], batches[3]
        # _, prev_acc =  MF_BPR.training_loss_acc(self.model_GMF, self.sess, prev_batch)

        # training the model
        _ = MF_BPR.training_batch(self.model_GMF, self.sess, batches)









    def save_model(self, folder_path, file_name = None):


        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {
                            # 'global_var': (#MF_BPR._user_input,
                            #                 #MF_BPR._item_input_pos,
                            #                 #MF_BPR._batch_size,
                            #                 #MF_BPR._index,
                            #                 #MF_BPR._model,
                            #                 #MF_BPR._dataset,
                            #                 #MF_BPR._K,
                            #                 #MF_BPR._feed_dict,
                            #                 #MF_BPR._output,
                            #                 # MF_BPR._exclude_gtItem,
                            #                 MF_BPR._user_exclude_validation
                            #                 ),
                             **{'_args_{}'.format(key):value for (key,value) in vars(self.args).items()}
                              }




        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)

        self.model_GMF.saveParams(self.sess, file_name + "_latent_factors", self.args)

        # saver = MF_BPR.tf.train.Saver()
        # saver.save(MF_BPR._sess, folder_path + file_name + "_session")

        self._print("Saving complete")



    def load_model(self, folder_path, file_name = None):


        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Loading model from file '{}'".format(folder_path + file_name))

        dataIO = DataIO(folder_path=folder_path)
        data_dict = dataIO.load_data(file_name=file_name)

        self.args = ArgsInterface()


        for attrib_name in data_dict.keys():
            #
            # if attrib_name == "global_var":
            #     MF_BPR._K, MF_BPR._feed_dict,\
            #     MF_BPR._output, MF_BPR._user_exclude_validation = data_dict[attrib_name]

            if attrib_name.startswith("_args_"):
                data_dict_key = attrib_name
                attrib_name = attrib_name[len("_args_"):]
                setattr(self.args, attrib_name, data_dict[data_dict_key])

            else:
                self.__setattr__(attrib_name, data_dict[attrib_name])


        self.dataset = DatasetInterface(URM_train = self.URM_train)

        MF_BPR.tf.reset_default_graph()

        # saver = MF_BPR.tf.train.Saver()
        self.sess = MF_BPR.tf.Session()
        # saver.restore(self.sess, folder_path + file_name + "_session")

        MF_BPR._sess = self.sess

        # initialize models
        self.model_GMF = MF_BPR.GMF(self.dataset.num_users, self.dataset.num_items, self.args)
        self.model_GMF.build_graph()

        self.model_GMF.load_parameter_MF(self.sess, folder_path + file_name + "_latent_factors.npy")

        self._print("Loading complete")



