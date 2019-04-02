#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/03/18

@author: Anonymous authors
"""

import os, multiprocessing
import traceback

from functools import partial
from skopt.space import Real, Integer, Categorical

from Base.NonPersonalizedRecommender import TopPop, Random, GlobalEffects
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from GraphBased.P3alphaRecommender import P3alphaRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython
from MatrixFactorization.PureSVDRecommender import PureSVDRecommender

from ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from ParameterTuning.SearchSingleCase import SearchSingleCase
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderParameters
from ParameterTuning.utility import get_dict_from_parameter_search


def runParameterSearch_Collaborative(recommender_class, URM_train, metric_to_optimize = "PRECISION",
                                     evaluator_validation = None, evaluator_test = None, evaluator_validation_earlystopping = None,
                                     output_folder_path ="result_experiments/", parallelizeKNN = True, n_cases = 35, allow_weighting = True,
                                     similarity_type_list = None, return_best_parameters_dict_list=False):



    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)


    try:

        output_file_name_root = recommender_class.RECOMMENDER_NAME

        parameterSearch = SearchBayesianSkopt(recommender_class, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)




        if recommender_class in [TopPop, GlobalEffects, Random]:
            """
            TopPop, GlobalEffects and Random have no parameters therefore only one evaluation is needed
            """


            parameterSearch = SearchSingleCase(recommender_class, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)

            recommender_parameters = SearchInputRecommenderParameters(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {}
            )

            parameterSearch.search(recommender_parameters,
                                   fit_parameters_values={},
                                   output_folder_path = output_folder_path,
                                   output_file_name_root = output_file_name_root
                                   )

            if return_best_parameters_dict_list:
                dict_best = get_dict_from_parameter_search(parameterSearch)

                return [dict_best]


            return



        ##########################################################################################################

        if recommender_class in [ItemKNNCFRecommender, UserKNNCFRecommender]:

            if similarity_type_list is None:
                similarity_type_list = ['cosine', 'jaccard', "asymmetric", "dice", "tversky"]

            recommender_parameters = SearchInputRecommenderParameters(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {}
            )


            run_KNNCFRecommender_on_similarity_type_partial = partial(_run_KNNRecommender_on_similarity_type,
                                                           recommender_parameters = recommender_parameters,
                                                           parameter_search_space = {},
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
                resultList = pool.map(run_KNNCFRecommender_on_similarity_type_partial, similarity_type_list)
                pool.close()
                pool.join()

            else:
                resultList = []
                for similarity_type in similarity_type_list:
                    resultList.append(run_KNNCFRecommender_on_similarity_type_partial(similarity_type))

            if return_best_parameters_dict_list:
                return resultList


            return



       ##########################################################################################################

        if recommender_class is P3alphaRecommender:

            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = Integer(5, 800)
            hyperparamethers_range_dictionary["alpha"] = Real(low = 0, high = 2, prior = 'uniform')
            hyperparamethers_range_dictionary["normalize_similarity"] = Categorical([True, False])

            recommender_parameters = SearchInputRecommenderParameters(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {}
            )


        ##########################################################################################################

        if recommender_class is RP3betaRecommender:

            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = Integer(5, 800)
            hyperparamethers_range_dictionary["alpha"] = Real(low = 0, high = 2, prior = 'uniform')
            hyperparamethers_range_dictionary["beta"] = Real(low = 0, high = 2, prior = 'uniform')
            hyperparamethers_range_dictionary["normalize_similarity"] = Categorical([True, False])

            recommender_parameters = SearchInputRecommenderParameters(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {}
            )



        ##########################################################################################################

        if recommender_class is MatrixFactorization_FunkSVD_Cython:

            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["sgd_mode"] = Categorical(["adagrad", "adam"])
            #hyperparamethers_range_dictionary["epochs"] = Integer(1, 150)
            hyperparamethers_range_dictionary["num_factors"] = Integer(1, 150)
            hyperparamethers_range_dictionary["reg"] = Real(low = 1e-12, high = 1e-3, prior = 'log-uniform')
            hyperparamethers_range_dictionary["learning_rate"] = Real(low = 1e-5, high = 1e-2, prior = 'log-uniform')

            recommender_parameters = SearchInputRecommenderParameters(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {"validation_every_n": 5,
                                    "stop_on_validation": True,
                                    "evaluator_object": evaluator_validation_earlystopping,
                                    "lower_validations_allowed": 20,
                                    "validation_metric": metric_to_optimize}
            )



        ##########################################################################################################

        if recommender_class is MatrixFactorization_BPR_Cython:

            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["sgd_mode"] = Categorical(["adagrad", "adam"])
            #hyperparamethers_range_dictionary["epochs"] = Integer(1, 150)
            hyperparamethers_range_dictionary["num_factors"] = Integer(1, 150)
            hyperparamethers_range_dictionary["batch_size"] = Categorical([1])
            hyperparamethers_range_dictionary["positive_reg"] = Real(low = 1e-12, high = 1e-3, prior = 'log-uniform')
            hyperparamethers_range_dictionary["negative_reg"] = Real(low = 1e-12, high = 1e-3, prior = 'log-uniform')
            hyperparamethers_range_dictionary["learning_rate"] = Real(low = 1e-5, high = 1e-2, prior = 'log-uniform')

            recommender_parameters = SearchInputRecommenderParameters(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {"validation_every_n": 5,
                                    "stop_on_validation": True,
                                    "evaluator_object": evaluator_validation_earlystopping,
                                    "lower_validations_allowed": 20,
                                    "validation_metric": metric_to_optimize}
            )


        ##########################################################################################################

        if recommender_class is PureSVDRecommender:

            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["num_factors"] = Integer(1, 250)

            recommender_parameters = SearchInputRecommenderParameters(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {}
            )


        #########################################################################################################

        if recommender_class is SLIM_BPR_Cython:

            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = Integer(5, 800)
            #hyperparamethers_range_dictionary["epochs"] = Integer(1, 150)
            hyperparamethers_range_dictionary["sgd_mode"] = Categorical(["adagrad", "adam"])
            hyperparamethers_range_dictionary["lambda_i"] = Real(low = 1e-12, high = 1e-3, prior = 'log-uniform')
            hyperparamethers_range_dictionary["lambda_j"] = Real(low = 1e-12, high = 1e-3, prior = 'log-uniform')

            recommender_parameters = SearchInputRecommenderParameters(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {'train_with_sparse_weights':True, 'symmetric':True, 'positive_threshold':0},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {"validation_every_n": 5,
                                    "stop_on_validation": True,
                                    "evaluator_object": evaluator_validation_earlystopping,
                                    "lower_validations_allowed": 20,
                                    "validation_metric": metric_to_optimize}
            )



        ##########################################################################################################

        if recommender_class is SLIMElasticNetRecommender:

            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = Integer(5, 800)
            hyperparamethers_range_dictionary["l1_ratio"] = Real(low = 1e-5, high = 1.0, prior = 'log-uniform')

            recommender_parameters = SearchInputRecommenderParameters(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {}
            )



       #########################################################################################################

        ## Final step, after the hyperparameter range has been defined for each type of algorithm
        parameterSearch.search(recommender_parameters,
                               parameter_search_space = hyperparamethers_range_dictionary,
                               n_cases = n_cases,
                               output_folder_path = output_folder_path,
                               output_file_name_root = output_file_name_root,
                               metric_to_optimize = metric_to_optimize)


        if return_best_parameters_dict_list:
            dict_best = get_dict_from_parameter_search(parameterSearch)

            return [dict_best]




    except Exception as e:

        print("On recommender {} Exception {}".format(recommender_class, str(e)))
        traceback.print_exc()

        error_file = open(output_folder_path + "ErrorLog.txt", "a")
        error_file.write("On recommender {} Exception {}\n".format(recommender_class, str(e)))
        error_file.close()



def _run_KNNRecommender_on_similarity_type(similarity_type, parameterSearch,
                                           parameter_search_space,
                                           recommender_parameters,
                                           n_cases,
                                           output_folder_path,
                                           output_file_name_root,
                                           metric_to_optimize,
                                           allow_weighting,
                                           return_best_parameters_dict=False):

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

