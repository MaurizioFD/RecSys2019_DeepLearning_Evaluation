#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/11/18

@author: Maurizio Ferrari Dacrema
"""



class EvaluatorUserSubsetWrapper(object):

    def __init__(self, evaluator_object, URM_validation_train):

        self.evaluator_object = evaluator_object
        self.URM_validation_train = URM_validation_train.copy()


    def evaluateRecommender(self, recommender_object):

        # URM_train_original = recommender_object.URM_train.copy()
        #
        # recommender_object.set_URM_train(self.URM_validation_train)
        # recommender_object.URM_train = self.URM_validation_train
        #
        # results_dict, n_users_evaluated = self.evaluator_object.evaluateRecommender(recommender_object)
        #
        # recommender_object.URM_train = URM_train_original

        URM_train_original = recommender_object.URM_train.copy()

        recommender_object.set_URM_train(self.URM_validation_train)

        results_dict, n_users_evaluated = self.evaluator_object.evaluateRecommender(recommender_object)

        recommender_object.set_URM_train(URM_train_original)

        return results_dict, n_users_evaluated






from Base.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender


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
        self.RECOMMENDER_NAME = self.mf_recommender.RECOMMENDER_NAME
        self.estimate_model_for_cold_users = False

    def _compute_item_score(self, *posargs, **kwargs):
        """
        Compute the items scores using the native function for the MF algorithm
        :param posargs:
        :param kwargs:
        :return:
        """
        return self.mf_recommender._compute_item_score(*posargs, **kwargs)

    def set_URM_train(self, URM_train_new, **kwargs):
        """
        Sets the new URM_train and applies the estimation of latent factors for cold users as requested in the fit function
        :param URM_train_new:
        :param kwargs:
        :return:
        """
        self.mf_recommender.set_URM_train(URM_train_new, estimate_model_for_cold_users = self.estimate_model_for_cold_users)

    def fit(self, *posargs, **kwargs):
        """
        Fits the MF model with the given hyperparameters and sets the kwarg "estimate_model_for_cold_users"
        :param posargs:
        :param kwargs:
        :return:
        """
        self.estimate_model_for_cold_users = kwargs["estimate_model_for_cold_users"]

        del kwargs["estimate_model_for_cold_users"]

        self.mf_recommender.fit(*posargs, **kwargs)



    def save_model(self, folder_path, file_name = None):
        """
        Saves the model using the native function for the MF model
        :param folder_path:
        :param file_name:
        :return:
        """
        self.mf_recommender.save_model(folder_path, file_name = file_name)


