#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 16/09/2017

@author: Maurizio Ferrari Dacrema
"""

from Base.BaseRecommender import BaseRecommender
from KNN.ItemKNNCustomSimilarityRecommender import ItemKNNCustomSimilarityRecommender
from Base.Recommender_utils import check_matrix
from Base.DataIO import DataIO
import numpy as np
import scipy.sparse as sps


def compute_W_sparse_from_item_latent_factors(ITEM_factors, topK = 100):


    n_items, n_factors = ITEM_factors.shape

    block_size = 100

    start_item = 0
    end_item = 0

    values = []
    rows = []
    cols = []


    # Compute all similarities for each item using vectorization
    while start_item < n_items:

        end_item = min(n_items, start_item + block_size)

        this_block_weight = np.dot(ITEM_factors[start_item:end_item, :], ITEM_factors.T)


        for col_index_in_block in range(this_block_weight.shape[0]):

            this_column_weights = this_block_weight[col_index_in_block, :]
            item_original_index = start_item + col_index_in_block

            # Sort indices and select TopK
            # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
            # - Partition the data to extract the set of relevant items
            # - Sort only the relevant items
            # - Get the original item index
            relevant_items_partition = (-this_column_weights).argpartition(topK-1)[0:topK]
            relevant_items_partition_sorting = np.argsort(-this_column_weights[relevant_items_partition])
            top_k_idx = relevant_items_partition[relevant_items_partition_sorting]

            # Incrementally build sparse matrix, do not add zeros
            notZerosMask = this_column_weights[top_k_idx] != 0.0
            numNotZeros = np.sum(notZerosMask)

            values.extend(this_column_weights[top_k_idx][notZerosMask])
            rows.extend(top_k_idx[notZerosMask])
            cols.extend(np.ones(numNotZeros) * item_original_index)



        start_item += block_size

    W_sparse = sps.csr_matrix((values, (rows, cols)),
                              shape=(n_items, n_items),
                              dtype=np.float32)

    return W_sparse





class BaseMatrixFactorizationRecommender(BaseRecommender):
    """
    This class refers to a BaseRecommender KNN which uses matrix factorization,
    it provides functions to compute item's score as well as a function to save the W_matrix

    The prediction for cold users will always be -inf for ALL items
    """

    def __init__(self, URM_train, verbose=True):
        super(BaseMatrixFactorizationRecommender, self).__init__(URM_train, verbose=verbose)

        self.use_bias = False

        self._cold_user_KNN_model_flag = False
        self._cold_user_KNN_estimated_factors_flag = False
        self._warm_user_KNN_mask = np.zeros(len(self._get_cold_user_mask()), dtype=np.bool)



    def set_URM_train(self, URM_train_new, estimate_model_for_cold_users = False, topK = 100, **kwargs):
        """

        :param URM_train_new:
        :param estimate_item_similarity_for_cold_users: Set to TRUE if you want to estimate the item-item similarity for cold users to be used as in a KNN algorithm
        :param topK: 100
        :param kwargs:
        :return:
        """

        assert self.URM_train.shape == URM_train_new.shape, "{}: set_URM_train old and new URM train have different shapes".format(self.RECOMMENDER_NAME)

        if len(kwargs)>0:
            self._print("set_URM_train keyword arguments not supported for this recommender class. Received: {}".format(kwargs))

        URM_train_new = check_matrix(URM_train_new, 'csr', dtype=np.float32)
        profile_length_new = np.ediff1d(URM_train_new.indptr)

        if estimate_model_for_cold_users == "itemKNN":

            self._print("Estimating ItemKNN model from ITEM latent factors...")

            W_sparse = compute_W_sparse_from_item_latent_factors(self.ITEM_factors, topK=topK)

            self._ItemKNNRecommender = ItemKNNCustomSimilarityRecommender(URM_train_new)
            self._ItemKNNRecommender.fit(W_sparse, topK=topK)
            self._ItemKNNRecommender_topK = topK

            self._cold_user_KNN_model_flag = True
            self._warm_user_KNN_mask = profile_length_new > 0

            self._print("Estimating ItemKNN model from ITEM latent factors... done!")



        elif estimate_model_for_cold_users == "mean_item_factors":

            self._print("Estimating USER latent factors from ITEM latent factors...")

            cold_user_mask_previous = self._get_cold_user_mask()
            profile_length_sqrt = np.sqrt(profile_length_new)

            self.USER_factors[cold_user_mask_previous,:] = URM_train_new.dot(self.ITEM_factors)[cold_user_mask_previous,:]
            self._cold_user_KNN_estimated_factors_flag = True

            #Divide every row for the sqrt of the profile length
            for user_index in range(self.n_users):
                if cold_user_mask_previous[user_index] and profile_length_sqrt[user_index] > 0:

                    self.USER_factors[user_index, :] /= profile_length_sqrt[user_index]

            self._print("Estimating USER latent factors from ITEM latent factors... done!")


        self.URM_train = check_matrix(URM_train_new.copy(), 'csr', dtype=np.float32)
        self.URM_train.eliminate_zeros()


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
                              "_cold_user_KNN_model_flag": self._cold_user_KNN_model_flag,
                              "_cold_user_KNN_estimated_factors_flag": self._cold_user_KNN_estimated_factors_flag}

        if self.use_bias:
            data_dict_to_save["ITEM_bias"] = self.ITEM_bias
            data_dict_to_save["USER_bias"] = self.USER_bias
            data_dict_to_save["GLOBAL_bias"] = self.GLOBAL_bias

        if self._cold_user_KNN_model_flag:
            data_dict_to_save["_ItemKNNRecommender_W_sparse"] = self._ItemKNNRecommender.W_sparse
            data_dict_to_save["_ItemKNNRecommender_topK"] = self._ItemKNNRecommender_topK


        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)


        self._print("Saving complete")




    def load_model(self, folder_path, file_name = None):
        super(BaseMatrixFactorizationRecommender, self).load_model(folder_path, file_name = file_name)

        if self._cold_user_KNN_model_flag:
            self._ItemKNNRecommender = ItemKNNCustomSimilarityRecommender(self.URM_train)
            self._ItemKNNRecommender.fit(self._ItemKNNRecommender_W_sparse, topK=self._ItemKNNRecommender_topK)

            del self._ItemKNNRecommender_W_sparse
            del self._ItemKNNRecommender_topK