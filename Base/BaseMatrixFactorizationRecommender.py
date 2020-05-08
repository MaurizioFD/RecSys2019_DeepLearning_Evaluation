#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 16/09/2017

@author: Maurizio Ferrari Dacrema
"""

from Base.BaseRecommender import BaseRecommender
from Base.DataIO import DataIO
import numpy as np



class BaseMatrixFactorizationRecommender(BaseRecommender):
    """
    This class refers to a BaseRecommender KNN which uses matrix factorization,
    it provides functions to compute item's score as well as a function to save the W_matrix

    The prediction for cold users will always be -inf for ALL items
    """

    def __init__(self, URM_train, verbose=True):
        super(BaseMatrixFactorizationRecommender, self).__init__(URM_train, verbose=verbose)

        self.use_bias = False




    #########################################################################################################
    ##########                                                                                     ##########
    ##########                               COMPUTE ITEM SCORES                                   ##########
    ##########                                                                                     ##########
    #########################################################################################################


    def _compute_item_score(self, user_id_array, items_to_compute = None):
        """
        USER_factors is n_users x n_factors
        ITEM_factors is n_items x n_factors

        The prediction for cold users will always be -inf for ALL items

        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        assert self.USER_factors.shape[1] == self.ITEM_factors.shape[1], \
            "{}: User and Item factors have inconsistent shape".format(self.RECOMMENDER_NAME)

        assert self.USER_factors.shape[0] > np.max(user_id_array),\
                "{}: Cold users not allowed. Users in trained model are {}, requested prediction for users up to {}".format(
                self.RECOMMENDER_NAME, self.USER_factors.shape[0], np.max(user_id_array))

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.ITEM_factors.shape[0]), dtype=np.float32)*np.inf
            item_scores[:, items_to_compute] = np.dot(self.USER_factors[user_id_array], self.ITEM_factors[items_to_compute,:].T)

        else:
            item_scores = np.dot(self.USER_factors[user_id_array], self.ITEM_factors.T)


        # No need to select only the specific negative items or warm users because the -inf score will not change
        if self.use_bias:
            item_scores += self.ITEM_bias + self.GLOBAL_bias
            item_scores = (item_scores.T + self.USER_bias[user_id_array]).T

        return item_scores


    #########################################################################################################
    ##########                                                                                     ##########
    ##########                                LOAD AND SAVE                                        ##########
    ##########                                                                                     ##########
    #########################################################################################################



    def save_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {"USER_factors": self.USER_factors,
                              "ITEM_factors": self.ITEM_factors,
                              "use_bias": self.use_bias,
                              "_cold_user_mask": self._cold_user_mask,
                            }

        if self.use_bias:
            data_dict_to_save["ITEM_bias"] = self.ITEM_bias
            data_dict_to_save["USER_bias"] = self.USER_bias
            data_dict_to_save["GLOBAL_bias"] = self.GLOBAL_bias

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)


        self._print("Saving complete")
