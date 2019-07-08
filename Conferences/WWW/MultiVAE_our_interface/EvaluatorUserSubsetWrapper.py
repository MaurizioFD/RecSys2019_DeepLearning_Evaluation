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

        recommender_object.set_URM_train(self.URM_validation_train, estimate_model_for_cold_users = "mean_item_factors", topK = 300)

        results_dict, n_users_evaluated = self.evaluator_object.evaluateRecommender(recommender_object)

        recommender_object.set_URM_train(URM_train_original)

        return results_dict, n_users_evaluated

