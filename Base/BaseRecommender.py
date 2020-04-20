#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
from Base.DataIO import DataIO
import os
from Base.Recommender_utils import check_matrix


class BaseRecommender(object):
    """Abstract BaseRecommender"""

    RECOMMENDER_NAME = "Recommender_Base_Class"

    def __init__(self, URM_train, verbose=True):

        super(BaseRecommender, self).__init__()

        self.URM_train = check_matrix(URM_train.copy(), 'csr', dtype=np.float32)
        self.URM_train.eliminate_zeros()

        self.n_users, self.n_items = self.URM_train.shape
        self.verbose = verbose

        self.filterTopPop = False
        self.filterTopPop_ItemsID = np.array([], dtype=np.int)

        self.items_to_ignore_flag = False
        self.items_to_ignore_ID = np.array([], dtype=np.int)

        self._cold_user_mask = np.ediff1d(self.URM_train.indptr) == 0

        if self._cold_user_mask.any():
            self._print("URM Detected {} ({:.2f} %) cold users.".format(
                self._cold_user_mask.sum(), self._cold_user_mask.sum()/self.n_users*100))


        self._cold_item_mask = np.ediff1d(self.URM_train.tocsc().indptr) == 0

        if self._cold_item_mask.any():
            self._print("URM Detected {} ({:.2f} %) cold items.".format(
                self._cold_item_mask.sum(), self._cold_item_mask.sum()/self.n_items*100))


    def _get_cold_user_mask(self):
        return self._cold_user_mask

    def _get_cold_item_mask(self):
        return self._cold_item_mask


    def _print(self, string):
        if self.verbose:
            print("{}: {}".format(self.RECOMMENDER_NAME, string))

    def fit(self):
        pass

    def get_URM_train(self):
        return self.URM_train.copy()

    def set_URM_train(self, URM_train_new, **kwargs):

        assert self.URM_train.shape == URM_train_new.shape, "{}: set_URM_train old and new URM train have different shapes".format(self.RECOMMENDER_NAME)

        if len(kwargs)>0:
            self._print("set_URM_train keyword arguments not supported for this recommender class. Received: {}".format(kwargs))

        self.URM_train = check_matrix(URM_train_new.copy(), 'csr', dtype=np.float32)
        self.URM_train.eliminate_zeros()

        self._cold_user_mask = np.ediff1d(self.URM_train.indptr) == 0

        if self._cold_user_mask.any():
            self._print("Detected {} ({:.2f} %) cold users.".format(
                self._cold_user_mask.sum(), self._cold_user_mask.sum()/len(self._cold_user_mask)*100))



    def set_items_to_ignore(self, items_to_ignore):
        self.items_to_ignore_flag = True
        self.items_to_ignore_ID = np.array(items_to_ignore, dtype=np.int)

    def reset_items_to_ignore(self):
        self.items_to_ignore_flag = False
        self.items_to_ignore_ID = np.array([], dtype=np.int)


    #########################################################################################################
    ##########                                                                                     ##########
    ##########                     COMPUTE AND FILTER RECOMMENDATION LIST                          ##########
    ##########                                                                                     ##########
    #########################################################################################################


    def _remove_TopPop_on_scores(self, scores_batch):
        scores_batch[:, self.filterTopPop_ItemsID] = -np.inf
        return scores_batch


    def _remove_custom_items_on_scores(self, scores_batch):
        scores_batch[:, self.items_to_ignore_ID] = -np.inf
        return scores_batch


    def _remove_seen_on_scores(self, user_id, scores):

        assert self.URM_train.getformat() == "csr", "Recommender_Base_Class: URM_train is not CSR, this will cause errors in filtering seen items"

        seen = self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]

        scores[seen] = -np.inf
        return scores


    def _compute_item_score(self, user_id_array, items_to_compute = None):
        """

        :param user_id_array:       array containing the user indices whose recommendations need to be computed
        :param items_to_compute:    array containing the items whose scores are to be computed.
                                        If None, all items are computed, otherwise discarded items will have as score -np.inf
        :return:                    array (len(user_id_array), n_items) with the score.
        """
        raise NotImplementedError("BaseRecommender: compute_item_score not assigned for current recommender, unable to compute prediction scores")


    def recommend(self, user_id_array, cutoff = None, remove_seen_flag=True, items_to_compute = None,
                  remove_top_pop_flag = False, remove_custom_items_flag = False, return_scores = False):

        # If is a scalar transform it in a 1-cell array
        if np.isscalar(user_id_array):
            user_id_array = np.atleast_1d(user_id_array)
            single_user = True
        else:
            single_user = False

        if cutoff is None:
            cutoff = self.URM_train.shape[1] - 1

        # Compute the scores using the model-specific function
        # Vectorize over all users in user_id_array
        scores_batch = self._compute_item_score(user_id_array, items_to_compute=items_to_compute)


        # if self.normalize:
        #     # normalization will keep the scores in the same range
        #     # of value of the ratings in dataset
        #     user_profile = self.URM_train[user_id]
        #
        #     rated = user_profile.copy()
        #     rated.data = np.ones_like(rated.data)
        #     if self.sparse_weights:
        #         den = rated.dot(self.W_sparse).toarray().ravel()
        #     else:
        #         den = rated.dot(self.W).ravel()
        #     den[np.abs(den) < 1e-6] = 1.0  # to avoid NaNs
        #     scores /= den


        for user_index in range(len(user_id_array)):

            user_id = user_id_array[user_index]

            if remove_seen_flag:
                scores_batch[user_index,:] = self._remove_seen_on_scores(user_id, scores_batch[user_index, :])

            # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
            # - Partition the data to extract the set of relevant items
            # - Sort only the relevant items
            # - Get the original item index
            # relevant_items_partition = (-scores_user).argpartition(cutoff)[0:cutoff]
            # relevant_items_partition_sorting = np.argsort(-scores_user[relevant_items_partition])
            # ranking = relevant_items_partition[relevant_items_partition_sorting]
            #
            # ranking_list.append(ranking)


        if remove_top_pop_flag:
            scores_batch = self._remove_TopPop_on_scores(scores_batch)

        if remove_custom_items_flag:
            scores_batch = self._remove_custom_items_on_scores(scores_batch)

        # relevant_items_partition is block_size x cutoff
        relevant_items_partition = (-scores_batch).argpartition(cutoff, axis=1)[:,0:cutoff]

        # Get original value and sort it
        # [:, None] adds 1 dimension to the array, from (block_size,) to (block_size,1)
        # This is done to correctly get scores_batch value as [row, relevant_items_partition[row,:]]
        relevant_items_partition_original_value = scores_batch[np.arange(scores_batch.shape[0])[:, None], relevant_items_partition]
        relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value, axis=1)
        ranking = relevant_items_partition[np.arange(relevant_items_partition.shape[0])[:, None], relevant_items_partition_sorting]

        ranking_list = [None] * ranking.shape[0]

        # Remove from the recommendation list any item that has a -inf score
        # Since -inf is a flag to indicate an item to remove
        for user_index in range(len(user_id_array)):
            user_recommendation_list = ranking[user_index]
            user_item_scores = scores_batch[user_index, user_recommendation_list]

            not_inf_scores_mask = np.logical_not(np.isinf(user_item_scores))

            user_recommendation_list = user_recommendation_list[not_inf_scores_mask]
            ranking_list[user_index] = user_recommendation_list.tolist()



        # Return single list for one user, instead of list of lists
        if single_user:
            ranking_list = ranking_list[0]


        if return_scores:
            return ranking_list, scores_batch

        else:
            return ranking_list



    #########################################################################################################
    ##########                                                                                     ##########
    ##########                                LOAD AND SAVE                                        ##########
    ##########                                                                                     ##########
    #########################################################################################################



    def save_model(self, folder_path, file_name = None):
        raise NotImplementedError("BaseRecommender: save_model not implemented")




    def load_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Loading model from file '{}'".format(folder_path + file_name))

        dataIO = DataIO(folder_path=folder_path)
        data_dict = dataIO.load_data(file_name=file_name)

        for attrib_name in data_dict.keys():
             self.__setattr__(attrib_name, data_dict[attrib_name])

        self._print("Loading complete")
