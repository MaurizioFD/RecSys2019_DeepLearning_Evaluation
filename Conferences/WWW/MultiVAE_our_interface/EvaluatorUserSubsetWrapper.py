#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/11/18

@author: Maurizio Ferrari Dacrema
"""


from MatrixFactorization.PureSVDRecommender import compute_W_sparse_from_item_latent_factors
from Base.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from KNN.ItemKNNCustomSimilarityRecommender import ItemKNNCustomSimilarityRecommender
import numpy as np
import scipy.sparse as sps
from Base.DataIO import DataIO
from Base.Recommender_utils import check_matrix



class EvaluatorUserSubsetWrapper(object):

    def __init__(self, evaluator_object, URM_validation_train):

        self.evaluator_object = evaluator_object
        self.URM_validation_train = URM_validation_train.copy()


    def evaluateRecommender(self, recommender_object):

        URM_train_original = recommender_object.URM_train.copy()

        # Set the new URM with held-out users
        recommender_object.set_URM_train(self.URM_validation_train)

        results_dict, n_users_evaluated = self.evaluator_object.evaluateRecommender(recommender_object)

        # Restore the train URM
        recommender_object.set_URM_train(URM_train_original)

        return results_dict, n_users_evaluated









class MF_cold_user_wrapper(BaseMatrixFactorizationRecommender):
    """ MF_cold_user_wrapper"""

    RECOMMENDER_NAME = "MF_cold_user_wrapper"

    def __init__(self, MF_recommender_class:BaseMatrixFactorizationRecommender, *posargs, **kwargs):
        """
        Creates an instance of the MF algorithm with the given hyperparameters and data
        :param MF_recommender_class:
        :param posargs:
        :param kwargs:
        """
        super(MF_cold_user_wrapper, self).__init__(*posargs, **kwargs)
        self.mf_recommender = MF_recommender_class(*posargs, **kwargs)
        self.RECOMMENDER_NAME = self.mf_recommender.RECOMMENDER_NAME# + "_cold_user_Wrapper"
        self.estimate_model_for_cold_users = False


        self._cold_user_KNN_model_flag = False
        self._cold_user_KNN_estimated_factors_flag = False
        self._warm_user_KNN_mask = np.zeros(len(self._get_cold_user_mask()), dtype=np.bool)


    def _compute_item_score(self, user_id_array, items_to_compute = None):
        """
        Compute the items scores using the native function for the MF algorithm
        :param posargs:
        :param kwargs:
        :return:
        """

        # item_scores = self.mf_recommender._compute_item_score(user_id_array, items_to_compute = items_to_compute)
        item_scores = super(MF_cold_user_wrapper, self)._compute_item_score(user_id_array, items_to_compute = items_to_compute)
        item_scores = self._compute_item_score_for_cold_users(user_id_array, item_scores, items_to_compute = items_to_compute)

        return item_scores



    def _compute_item_score_for_cold_users(self, user_id_array, item_scores, items_to_compute = None):
        """
        Compute item scores with the ItemKNN model
        :param user_id_array:
        :param item_scores:
        :return:
        """

        cold_users_batch_mask = self._get_cold_user_mask()[user_id_array]

        if cold_users_batch_mask.any() and not self._cold_user_KNN_estimated_factors_flag:

            if self._cold_user_KNN_model_flag:
                # Add KNN scores for users cold for MF but warm in KNN model
                cold_users_in_MF_warm_in_KNN_mask = np.logical_and(cold_users_batch_mask, self._warm_user_KNN_mask[user_id_array])

                item_scores[cold_users_in_MF_warm_in_KNN_mask, :] = self._ItemKNNRecommender._compute_item_score(user_id_array[cold_users_in_MF_warm_in_KNN_mask], items_to_compute=items_to_compute)

        return item_scores



    def set_URM_train(self, URM_train_new):
        """

        :param URM_train_new:
        :param estimate_item_similarity_for_cold_users: Set to TRUE if you want to estimate the item-item similarity for cold users to be used as in a KNN algorithm
        :param topK: 100
        :param kwargs:
        :return:
        """

        assert self.URM_train.shape == URM_train_new.shape, "{}: set_URM_train old and new URM train have different shapes".format(self.RECOMMENDER_NAME)

        URM_train_new = check_matrix(URM_train_new, 'csr', dtype=np.float32)
        profile_length_new = np.ediff1d(URM_train_new.indptr)

        if self.estimate_model_for_cold_users == "itemKNN":

            self._print("Generating ItemKNN model from ITEM latent factors...")

            W_sparse = compute_W_sparse_from_item_latent_factors(self.ITEM_factors,
                                                                 topK=self.estimate_model_for_cold_users_topK)

            self._ItemKNNRecommender = ItemKNNCustomSimilarityRecommender(URM_train_new)
            self._ItemKNNRecommender.fit(W_sparse, topK=None)

            self._cold_user_KNN_model_flag = True
            self._cold_user_KNN_model_flag = True
            self._warm_user_KNN_mask = profile_length_new > 0

            self._print("Generating ItemKNN model from ITEM latent factors... done!")



        elif self.estimate_model_for_cold_users == "mean_item_factors":

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


        self.URM_train = sps.csr_matrix(URM_train_new.copy())
        self.URM_train.eliminate_zeros()


    def fit(self, *posargs,
            estimate_model_for_cold_users = None,
            estimate_model_for_cold_users_topK = 100,
            **kwargs):
        """
        Fits the MF model with the given hyper-parameters and sets the kwarg "estimate_model_for_cold_users"
        :param posargs:
        :param kwargs:
        :return:
        """
        self.estimate_model_for_cold_users = estimate_model_for_cold_users
        self.estimate_model_for_cold_users_topK = estimate_model_for_cold_users_topK

        self.mf_recommender.fit(*posargs, **kwargs)

        self.USER_factors = self.mf_recommender.USER_factors.copy()
        self.ITEM_factors = self.mf_recommender.ITEM_factors.copy()




    def save_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        self.mf_recommender.save_model(folder_path, file_name = file_name + "_warm_users")

        data_dict_to_save = {"_cold_user_KNN_model_flag": self._cold_user_KNN_model_flag,
                             "_cold_user_KNN_estimated_factors_flag": self._cold_user_KNN_estimated_factors_flag}

        if self._cold_user_KNN_model_flag:
            self._ItemKNNRecommender.save_model(folder_path, file_name = file_name + "_cold_users")

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)

        self._print("Saving complete")




    def load_model(self, folder_path, file_name = None):
        super(BaseMatrixFactorizationRecommender, self).load_model(folder_path, file_name = file_name)

        self.mf_recommender.load_model(folder_path, file_name = file_name + "_warm_users")

        if self._cold_user_KNN_model_flag:
            self._ItemKNNRecommender = ItemKNNCustomSimilarityRecommender(self.URM_train)
            self._ItemKNNRecommender.load_model(folder_path, file_name = file_name + "_cold_users")
