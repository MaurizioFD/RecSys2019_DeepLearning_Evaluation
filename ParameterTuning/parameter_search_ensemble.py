#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/03/19

@author: Anonymous authors
"""

import os

from Ensemble.Hybrid_Ensemble_Recommender import Hybrid_Ensemble_Recommender

from skopt.space import Real, Categorical

from ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderParameters
from ParameterTuning.utility import get_dict_from_parameter_search


def runParameterSearch_Ensemble(recommender_class, URM_train, EURM_list, EURM_name, n_cases = 30,
                                evaluator_validation= None, evaluator_test=None, metric_to_optimize = "PRECISION",
                                output_folder_path ="result_experiments/", return_best_parameters_dict_list = False):


    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

   ##########################################################################################################

    output_file_name_root = recommender_class.RECOMMENDER_NAME + "_{}".format(EURM_name)

    parameterSearch = SearchBayesianSkopt(recommender_class, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)

    if recommender_class is Hybrid_Ensemble_Recommender:

        hyperparamethers_range_dictionary = {}

        hyperparamethers_range_dictionary["normalization"] = Categorical(["l1", "max", "none"])

        for i in range(len(EURM_list)):
            key = 'eurm_w{}'.format(int(i))
            hyperparamethers_range_dictionary[key] = Real(low = 0, high = 10, prior = 'uniform')

        recommender_parameters = SearchInputRecommenderParameters(
            CONSTRUCTOR_POSITIONAL_ARGS = [EURM_list, URM_train],
            CONSTRUCTOR_KEYWORD_ARGS = {},
            FIT_POSITIONAL_ARGS = [],
            FIT_KEYWORD_ARGS = {}
        )

        parameterSearch.search(recommender_parameters,
                               parameter_search_space=hyperparamethers_range_dictionary,
                               n_cases=n_cases,
                               output_folder_path=output_folder_path,
                               output_file_name_root=output_file_name_root,
                               metric_to_optimize=metric_to_optimize)

        if return_best_parameters_dict_list:
            dict_best = get_dict_from_parameter_search(parameterSearch)

            return [dict_best]
