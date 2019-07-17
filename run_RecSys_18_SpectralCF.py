#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17

@author: Maurizio Ferrari Dacrema
@author: Simone Boglio

"""

from Recommender_import_list import *
from Conferences.RecSys.SpectralCF_our_interface.SpectralCF_RecommenderWrapper import SpectralCF_RecommenderWrapper


from ParameterTuning.run_parameter_search import runParameterSearch_Collaborative


from Utils.print_results_latex_table import print_time_statistics_latex_table, print_results_latex_table, print_parameters_latex_table
from Utils.assertions_on_data_for_experiments import assert_implicit_data, assert_disjoint_matrices
from Utils.plot_popularity import plot_popularity_bias, save_popularity_statistics

from functools import partial
import numpy as np
import os, traceback, multiprocessing

from Conferences.RecSys.SpectralCF_our_interface.Movielens1M.Movielens1MReader import Movielens1MReader
from Conferences.RecSys.SpectralCF_our_interface.MovielensHetrec2011.MovielensHetrec2011Reader import MovielensHetrec2011Reader
from Conferences.RecSys.SpectralCF_our_interface.AmazonInstantVideo.AmazonInstantVideoReader import AmazonInstantVideoReader



######################################################################
from skopt.space import Real, Integer, Categorical
import traceback
from Utils.PoolWithSubprocess import PoolWithSubprocess


from ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from ParameterTuning.SearchSingleCase import SearchSingleCase
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderParameters


def runParameterSearch_SpectralCF(recommender_class, URM_train, earlystopping_parameters, output_file_name_root, n_cases = 35,
                             evaluator_validation= None, evaluator_test=None, metric_to_optimize = "RECALL",
                             output_folder_path ="result_experiments/"):



    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)


    parameterSearch = SearchBayesianSkopt(recommender_class, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)


    ##########################################################################################################

    if recommender_class is SpectralCF_RecommenderWrapper:

        hyperparameters_range_dictionary = {}
        hyperparameters_range_dictionary["batch_size"] = Categorical([1024])
        hyperparameters_range_dictionary["embedding_size"] =  Categorical([4, 8, 16, 32])
        hyperparameters_range_dictionary["decay"] = Real(low = 1e-5, high = 1e-1, prior = 'log-uniform')
        hyperparameters_range_dictionary["learning_rate"] = Real(low = 1e-5, high = 1e-2, prior = 'log-uniform')
        hyperparameters_range_dictionary["k"] = Integer(low = 1, high = 6)

        recommender_parameters = SearchInputRecommenderParameters(
            CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
            CONSTRUCTOR_KEYWORD_ARGS = {},
            FIT_POSITIONAL_ARGS = [],
            FIT_KEYWORD_ARGS = earlystopping_parameters
        )


    #########################################################################################################

    parameterSearch.search(recommender_parameters,
                           parameter_search_space = hyperparameters_range_dictionary,
                           n_cases = n_cases,
                           output_folder_path = output_folder_path,
                           output_file_name_root = output_file_name_root,
                           metric_to_optimize = metric_to_optimize)





def read_data_split_and_search_SpectralCF(dataset_name, cold_start=False,
                                          cold_items=None, isKNN_multiprocess=True, isKNN_tune=True,
                                          isSpectralCF_train_default=True, isSpectralCF_tune=True, print_results=True):


    if dataset_name == "movielens1m_original":
        assert(cold_start is not True)
        dataset = Movielens1MReader(type="original")

    elif dataset_name == "movielens1m_ours":
        dataset = Movielens1MReader(type="ours", cold_start=cold_start, cold_items=cold_items)

    elif dataset_name == "hetrec":
        assert (cold_start is not True)
        dataset = MovielensHetrec2011Reader()

    elif dataset_name == "amazon_instant_video":
        assert (cold_start is not True)
        dataset = AmazonInstantVideoReader()


    if not cold_start:
        output_folder_path = "result_experiments/{}/{}_{}/".format(CONFERENCE_NAME, ALGORITHM_NAME, dataset_name)
    else:
        output_folder_path = "result_experiments/{}/{}_cold_{}_{}/".format(CONFERENCE_NAME, ALGORITHM_NAME, cold_items, dataset_name)


    URM_train = dataset.URM_train.copy()
    URM_validation = dataset.URM_validation.copy()
    URM_test = dataset.URM_test.copy()

    # Ensure IMPLICIT data and DISJOINT sets
    assert_implicit_data([URM_train, URM_validation, URM_test])
    assert_disjoint_matrices([URM_train, URM_validation, URM_test])


    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    algorithm_dataset_string = "{}_{}_".format(ALGORITHM_NAME, dataset_name)

    plot_popularity_bias([URM_train + URM_validation, URM_test],
                         ["URM train", "URM test"],
                         output_folder_path + algorithm_dataset_string + "popularity_plot")

    save_popularity_statistics([URM_train + URM_validation, URM_test],
                               ["URM train", "URM test"],
                               output_folder_path + algorithm_dataset_string + "popularity_statistics")


    metric_to_optimize = "RECALL"

    from Base.Evaluation.Evaluator import EvaluatorHoldout

    if not cold_start:
        cutoff_list_validation = [50]
        cutoff_list_test = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    else:
        cutoff_list_validation = [20]
        cutoff_list_test = [20]

    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=cutoff_list_validation)
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list_test)

    ################################################################################################
    ###### KNN CF

    if isKNN_tune:
        collaborative_algorithm_list = [
            Random,
            TopPop,
            UserKNNCFRecommender,
            ItemKNNCFRecommender,
            P3alphaRecommender,
            RP3betaRecommender,
        ]

        runParameterSearch_Collaborative_partial = partial(runParameterSearch_Collaborative,
                                                           URM_train = URM_train,
                                                           metric_to_optimize = metric_to_optimize,
                                                           evaluator_validation_earlystopping = evaluator_validation,
                                                           evaluator_validation = evaluator_validation,
                                                           evaluator_test = evaluator_test,
                                                           output_folder_path = output_folder_path,
                                                           parallelizeKNN = False,
                                                           allow_weighting = True,
                                                           n_cases = 35)
        if isKNN_multiprocess:
            pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
            resultList = pool.map(runParameterSearch_Collaborative_partial, collaborative_algorithm_list)

            pool.close()
            pool.join()

        else:
            for recommender_class in collaborative_algorithm_list:
                try:
                    runParameterSearch_Collaborative_partial(recommender_class)
                except Exception as e:
                    print("On recommender {} Exception {}".format(recommender_class, str(e)))
                    traceback.print_exc()



    ################################################################################################
    ###### SpectralCF

    if isSpectralCF_train_default:
        try:

            spectralCF_article_parameters = {
                "epochs": 1000,
                "batch_size": 1024,
                "embedding_size": 16,
                "decay": 0.001,
                "k": 3,
                "learning_rate": 1e-3,
            }

            spectralCF_earlystopping_parameters = {
                "validation_every_n": 5,
                "stop_on_validation": True,
                "lower_validations_allowed": 20,
                "evaluator_object": evaluator_validation,
                "validation_metric": metric_to_optimize,
                "epochs_min": 400,
            }

            parameterSearch = SearchSingleCase(SpectralCF_RecommenderWrapper,
                                               evaluator_validation=evaluator_validation,
                                               evaluator_test=evaluator_test)

            recommender_parameters = SearchInputRecommenderParameters(
                                                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                                                FIT_KEYWORD_ARGS = spectralCF_earlystopping_parameters)

            parameterSearch.search(recommender_parameters,
                                   fit_parameters_values = spectralCF_article_parameters,
                                   output_folder_path = output_folder_path,
                                   output_file_name_root = SpectralCF_RecommenderWrapper.RECOMMENDER_NAME + "_article_default")

        except Exception as e:

            print("On recommender {} Exception {}".format(SpectralCF_RecommenderWrapper, str(e)))
            traceback.print_exc()


    elif isSpectralCF_tune:

        try:

            spectralCF_earlystopping_parameters = {
                "validation_every_n": 5,
                "stop_on_validation": True,
                "lower_validations_allowed": 20,
                "evaluator_object": evaluator_validation,
                "validation_metric": metric_to_optimize,
                "epochs_min": 400,
                "epochs": 2000
            }

            runParameterSearch_SpectralCF(SpectralCF_RecommenderWrapper,
                                             URM_train = URM_train,
                                             earlystopping_parameters = spectralCF_earlystopping_parameters,
                                             metric_to_optimize = metric_to_optimize,
                                             evaluator_validation = evaluator_validation,
                                             evaluator_test = evaluator_test,
                                             output_folder_path = output_folder_path,
                                             n_cases = 35,
                                             output_file_name_root = SpectralCF_RecommenderWrapper.RECOMMENDER_NAME)



        except Exception as e:

            print("On recommender {} Exception {}".format(SpectralCF_RecommenderWrapper, str(e)))
            traceback.print_exc()

    ################################################################################################
    ###### print results

    if print_results:

        n_validation_users = np.sum(np.ediff1d(URM_validation.indptr)>=1)
        n_test_users = np.sum(np.ediff1d(URM_test.indptr)>=1)

        if not cold_start:
            results_file_root_name = ALGORITHM_NAME
        else:
            results_file_root_name = "{}_cold_{}".format(ALGORITHM_NAME, cold_items)



        print_time_statistics_latex_table(result_folder_path = output_folder_path,
                                          dataset_name = dataset_name,
                                          results_file_prefix_name = results_file_root_name,
                                          other_algorithm_list = [SpectralCF_RecommenderWrapper],
                                          n_validation_users = n_validation_users,
                                          n_test_users = n_test_users,
                                          n_decimals = 2)

        if cold_start:
            cutoffs_to_report_list = [20]
        else:
            cutoffs_to_report_list = [20, 40, 60, 80, 100]

        print_results_latex_table(result_folder_path = output_folder_path,
                                  results_file_prefix_name = results_file_root_name,
                                  dataset_name = dataset_name,
                                  metrics_to_report_list = ["RECALL", "MAP"],
                                  cutoffs_to_report_list = cutoffs_to_report_list,
                                  other_algorithm_list = [SpectralCF_RecommenderWrapper])



if __name__ == '__main__':

    ALGORITHM_NAME = "SpectralCF"
    CONFERENCE_NAME = "RecSys"

    # isKNN_multiprocess = True: knn parameter search in parallel among multiple process (fast), False: search in sequential way (slow, but no worry for deadlock)
    isKNN_multiprocess = False
    # isKNN_tune = True: knn parameter search, False: avoid this step
    isKNN_tune = True
    # isSpectralCF_train = True: train the SpectralCF model, False: avoid this step
    isSpectralCF_train_default = False
    # isSpectralCF_tune = True: parameter search for SpectralCF, False: avoid this step
    isSpectralCF_tune = True
    # print_results = True: print the results read from the output folder, False: avoid this step
    print_results = True

    cold_start = False

    dataset_list = ["movielens1m_ours", "movielens1m_original", "hetrec", "amazon_instant_video"]
    dataset_cold_start_list = ["movielens1m_ours"]
    cold_start_items_list = [1, 2, 3, 4, 5]

    if cold_start:
        for dataset_name in dataset_cold_start_list:
           for cold_start_items in cold_start_items_list:
               read_data_split_and_search_SpectralCF(dataset_name, cold_start=cold_start, cold_items=cold_start_items,
                                                     isKNN_multiprocess=isKNN_multiprocess,
                                                     isKNN_tune=isKNN_tune,
                                                     isSpectralCF_train_default=isSpectralCF_train_default,
                                                     print_results=print_results
                                                     )


    else:
        for dataset_name in dataset_list:
            read_data_split_and_search_SpectralCF(dataset_name, cold_start=cold_start,
                                                  isKNN_multiprocess=isKNN_multiprocess,
                                                  isKNN_tune=isKNN_tune,
                                                  isSpectralCF_train_default=isSpectralCF_train_default,
                                                  print_results=print_results
                                                  )


    # mantain compatibility with latex parameteres function
    if cold_start and print_results:
        for n_cold_item in cold_start_items_list:
            print_parameters_latex_table(result_folder_path = "result_experiments/{}/".format(CONFERENCE_NAME),
                                              results_file_prefix_name = "{}_cold_{}".format(ALGORITHM_NAME, n_cold_item),
                                              experiment_subfolder_list = dataset_cold_start_list,
                                              other_algorithm_list = [SpectralCF_RecommenderWrapper])
    elif not cold_start and print_results:
        print_parameters_latex_table(result_folder_path = "result_experiments/{}/".format(CONFERENCE_NAME),
                                       results_file_prefix_name = ALGORITHM_NAME,
                                       experiment_subfolder_list = dataset_list,
                                       other_algorithm_list = [SpectralCF_RecommenderWrapper])
