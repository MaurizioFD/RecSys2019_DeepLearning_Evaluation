#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 16/09/2017

@author: Maurizio Ferrari Dacrema
"""

from Base.BaseRecommender import BaseRecommender
from KNN.ItemKNNCustomSimilarityRecommender import ItemKNNCustomSimilarityRecommender
from Base.Recommender_utils import check_matrix
import pickle
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

    def __init__(self, URM_train):
        super(BaseMatrixFactorizationRecommender, self).__init__(URM_train)

        self._cold_user_mask = np.ediff1d(self.URM_train.indptr) == 0
        self._cold_user_KNN_model_available = False
        self._warm_user_KNN_mask = np.zeros(len(self._cold_user_mask), dtype=np.bool)

        if self._cold_user_mask.any():
            print("{}: Detected {} ({:.2f} %) cold users.".format(
                self.RECOMMENDER_NAME, self._cold_user_mask.sum(), self._cold_user_mask.sum()/len(self._cold_user_mask)*100))


    def _get_cold_user_mask(self):
        return self._cold_user_mask


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
            print("{}: set_URM_train keyword arguments not supported for this recommender class. Received: {}".format(self.RECOMMENDER_NAME, kwargs))

        self.URM_train = check_matrix(URM_train_new.copy(), 'csr', dtype=np.float32)
        self.URM_train.eliminate_zeros()

        if estimate_model_for_cold_users == "itemKNN":

            print("{}: Estimating ItemKNN model from ITEM latent factors...".format(self.RECOMMENDER_NAME))

            W_sparse = compute_W_sparse_from_item_latent_factors(self.ITEM_factors, topK=topK)

            self._ItemKNNRecommender = ItemKNNCustomSimilarityRecommender(self.URM_train)
            self._ItemKNNRecommender.fit(W_sparse, topK=topK)

            self._cold_user_KNN_model_available = True
            self._warm_user_KNN_mask = np.ediff1d(self.URM_train.indptr) > 0

            print("{}: Estimating ItemKNN model from ITEM latent factors... done!".format(self.RECOMMENDER_NAME))



        elif estimate_model_for_cold_users == "mean_item_factors":

            print("{}: Estimating USER latent factors from ITEM latent factors...".format(self.RECOMMENDER_NAME))

            self._cold_user_mask = np.ediff1d(self.URM_train.indptr) == 0

            profile_length = np.ediff1d(self.URM_train.indptr)
            profile_length_sqrt = np.sqrt(profile_length)

            self.USER_factors = self.URM_train.dot(self.ITEM_factors)

            #Divide every row for the sqrt of the profile length
            for user_index in range(self.n_users):

                if profile_length_sqrt[user_index] > 0:

                    self.USER_factors[user_index, :] /= profile_length_sqrt[user_index]

            print("{}: Estimating USER latent factors from ITEM latent factors... done!".format(self.RECOMMENDER_NAME))



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

        assert self.USER_factors.shape[0] > user_id_array.max(),\
                "{}: Cold users not allowed. Users in trained model are {}, requested prediction for users up to {}".format(
                self.RECOMMENDER_NAME, self.USER_factors.shape[0], user_id_array.max())

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.ITEM_factors.shape[0]), dtype=np.float32)*np.inf
            item_scores[:, items_to_compute] = np.dot(self.USER_factors[user_id_array], self.ITEM_factors[items_to_compute,:].T)
        else:
            item_scores = np.dot(self.USER_factors[user_id_array], self.ITEM_factors.T)

        cold_users_MF_mask = self._get_cold_user_mask()[user_id_array]

        if cold_users_MF_mask.any():

            if self._cold_user_KNN_model_available:
                # Add KNN scores for users cold for MF but warm in KNN model
                cold_users_in_MF_warm_in_KNN_mask = np.logical_and(cold_users_MF_mask, self._warm_user_KNN_mask[user_id_array])

                item_scores[cold_users_in_MF_warm_in_KNN_mask, :] = self._ItemKNNRecommender._compute_item_score(user_id_array[cold_users_in_MF_warm_in_KNN_mask], items_to_compute=items_to_compute)

                # Set cold users as those neither in MF nor in KNN
                cold_users_MF_mask = np.logical_and(cold_users_MF_mask, np.logical_not(cold_users_in_MF_warm_in_KNN_mask))

            # Set as -inf all remaining cold user scores
            item_scores[cold_users_MF_mask, :] = - np.ones_like(item_scores[cold_users_MF_mask, :]) * np.inf

        return item_scores





    def saveModel(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        print("{}: Saving model in file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))

        dictionary_to_save = {"USER_factors": self.USER_factors,
                              "ITEM_factors": self.ITEM_factors,
                              "_cold_user_mask": self._cold_user_mask}


        pickle.dump(dictionary_to_save,
                    open(folder_path + file_name, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)


        print("{}: Saving complete".format(self.RECOMMENDER_NAME, folder_path + file_name))
