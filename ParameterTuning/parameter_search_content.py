#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/03/19

@author: Anonymous authors
"""

import os, multiprocessing
from functools import partial

from  KNN.ItemKNNCBF_FW_Recommender import ItemKNNCBF_FW_Recommender
from KNN.UserKNNCBF_FW_Recommender import UserKNNCBF_FW_Recommender

from skopt.space import Real, Integer, Categorical

from ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderParameters
from ParameterTuning.utility import get_dict_from_parameter_search


def runParameterSearch_UserContentFW(recommender_class, URM_train, UCM_list, UCM_name, n_cases = 30,
                                     evaluator_validation= None, evaluator_test=None, metric_to_optimize = "PRECISION",
                                     output_folder_path ="result_experiments/", parallelizeKNN = False, allow_weighting = True,
                                     similarity_type_list = None, return_best_parameters_dict_list = False):


    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

   ##########################################################################################################

    output_file_name_root = recommender_class.RECOMMENDER_NAME + "_{}".format(UCM_name)

    parameterSearch = SearchBayesianSkopt(recommender_class, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)


    if recommender_class is UserKNNCBF_FW_Recommender:

        if similarity_type_list is None:
            similarity_type_list = ['cosine', 'jaccard', 'asymmetric', 'dice', 'tversky']


        hyperparamethers_range_dictionary = {}

        for i in range(len(UCM_list)):
            key = 'ucm_w{}'.format(int(i))
            hyperparamethers_range_dictionary[key] = Real(low=0, high=5, prior='uniform')

        recommender_parameters = SearchInputRecommenderParameters(
            CONSTRUCTOR_POSITIONAL_ARGS = [UCM_list, URM_train],
            CONSTRUCTOR_KEYWORD_ARGS = {},
            FIT_POSITIONAL_ARGS = [],
            FIT_KEYWORD_ARGS = {}
        )


        run_KNN_CBF_FW_Recommender_on_similarity_type_partial = partial(_run_KNNRecommender_on_similarity_type,
                                                                        parameter_search_space = hyperparamethers_range_dictionary,
                                                                        recommender_parameters = recommender_parameters,
                                                                        parameterSearch = parameterSearch,
                                                                        n_cases = n_cases,
                                                                        output_folder_path = output_folder_path,
                                                                        output_file_name_root = output_file_name_root,
                                                                        metric_to_optimize = metric_to_optimize,
                                                                        allow_weighting = allow_weighting,
                                                                        return_best_parameters_dict = return_best_parameters_dict_list,
                                                                        )



        if parallelizeKNN:
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count(), maxtasksperchild=1)
            resultList = pool.map(run_KNN_CBF_FW_Recommender_on_similarity_type_partial, similarity_type_list)
            pool.close()
            pool.join()


        else:
            resultList = []
            for similarity_type in similarity_type_list:
                resultList.append(run_KNN_CBF_FW_Recommender_on_similarity_type_partial(similarity_type))

        if return_best_parameters_dict_list:
            return resultList

        return


def runParameterSearch_ItemContentFW(recommender_class, URM_train, ICM_list, ICM_name, n_cases = 30,
                                     evaluator_validation= None, evaluator_test=None, metric_to_optimize = "PRECISION",
                                     output_folder_path ="result_experiments/", parallelizeKNN = False, allow_weighting = True,
                                     similarity_type_list = None, return_best_parameters_dict_list=False):


    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

   ##########################################################################################################

    output_file_name_root = recommender_class.RECOMMENDER_NAME + "_{}".format(ICM_name)

    parameterSearch = SearchBayesianSkopt(recommender_class, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)


    if recommender_class is ItemKNNCBF_FW_Recommender:

        if similarity_type_list is None:
            similarity_type_list = ['cosine', 'jaccard', "asymmetric", "dice", "tversky"]


        hyperparamethers_range_dictionary = {}

        for i in range(len(ICM_list)):
            key = 'icm_w{}'.format(int(i))
            hyperparamethers_range_dictionary[key] = Real(low=0, high=5, prior='uniform')

        recommender_parameters = SearchInputRecommenderParameters(
            CONSTRUCTOR_POSITIONAL_ARGS = [ICM_list, URM_train],
            CONSTRUCTOR_KEYWORD_ARGS = {},
            FIT_POSITIONAL_ARGS = [],
            FIT_KEYWORD_ARGS = {}
        )


        run_KNN_CBF_FW_Recommender_on_similarity_type_partial = partial(_run_KNNRecommender_on_similarity_type,
                                                                        parameter_search_space = hyperparamethers_range_dictionary,
                                                                        recommender_parameters = recommender_parameters,
                                                                        parameterSearch = parameterSearch,
                                                                        n_cases = n_cases,
                                                                        output_folder_path = output_folder_path,
                                                                        output_file_name_root = output_file_name_root,
                                                                        metric_to_optimize = metric_to_optimize,
                                                                        allow_weighting = allow_weighting,
                                                                        return_best_parameters_dict=return_best_parameters_dict_list
                                                                        )



        if parallelizeKNN:
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count(), maxtasksperchild=1)
            resultList = pool.map(run_KNN_CBF_FW_Recommender_on_similarity_type_partial, similarity_type_list)
            pool.close()
            pool.join()


        else:
            resultList = []
            for similarity_type in similarity_type_list:
                resultList.append(run_KNN_CBF_FW_Recommender_on_similarity_type_partial(similarity_type))


        if return_best_parameters_dict_list:
            return resultList



        return


def _run_KNNRecommender_on_similarity_type(similarity_type, parameterSearch,
                                           parameter_search_space,
                                           recommender_parameters,
                                           n_cases,
                                           output_folder_path,
                                           output_file_name_root,
                                           metric_to_optimize,
                                           allow_weighting,
                                           return_best_parameters_dict = False):

    original_parameter_search_space = parameter_search_space

    hyperparamethers_range_dictionary = {}
    hyperparamethers_range_dictionary["topK"] = Integer(5, 800)
    hyperparamethers_range_dictionary["shrink"] = Integer(0, 1000)
    hyperparamethers_range_dictionary["similarity"] = Categorical([similarity_type])
    hyperparamethers_range_dictionary["normalize"] = Categorical([True, False])

    if similarity_type == "asymmetric":
        hyperparamethers_range_dictionary["asymmetric_alpha"] = Real(low = 0, high = 2, prior = 'uniform')
        hyperparamethers_range_dictionary["normalize"] = Categorical([True])

    elif similarity_type == "tversky":
        hyperparamethers_range_dictionary["tversky_alpha"] = Real(low = 0, high = 2, prior = 'uniform')
        hyperparamethers_range_dictionary["tversky_beta"] = Real(low = 0, high = 2, prior = 'uniform')
        hyperparamethers_range_dictionary["normalize"] = Categorical([True])

    elif similarity_type == "euclidean":
        hyperparamethers_range_dictionary["normalize"] = Categorical([True, False])
        hyperparamethers_range_dictionary["normalize_avg_row"] = Categorical([True, False])
        hyperparamethers_range_dictionary["similarity_from_distance_mode"] = Categorical(["lin", "log", "exp"])

    if similarity_type in ["cosine", "asymmetric", "euclidean"] and allow_weighting:
        hyperparamethers_range_dictionary["feature_weighting"] = Categorical(["none", "BM25", "TF-IDF"])


    local_parameter_search_space = {**hyperparamethers_range_dictionary, **original_parameter_search_space}

    parameterSearch.search(recommender_parameters,
                           parameter_search_space = local_parameter_search_space,
                           n_cases = n_cases,
                           output_folder_path = output_folder_path,
                           output_file_name_root = output_file_name_root + "_" + similarity_type,
                           metric_to_optimize = metric_to_optimize)

    if return_best_parameters_dict:
        dict_best = get_dict_from_parameter_search(parameterSearch)
        return dict_best
