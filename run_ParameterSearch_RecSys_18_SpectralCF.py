#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17

@author: Anonymous authors

"""


from Base.NonPersonalizedRecommender import TopPop, Random
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from GraphBased.P3alphaRecommender import P3alphaRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython

from ParameterTuning.run_parameter_search import runParameterSearch_Collaborative, runParameterSearch_Content, runParameterSearch_Hybrid
from ParameterTuning.SearchSingleCase import SearchSingleCase
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderParameters

from Conferences.RecSys.SpectralCF_our_interface.SpectralCF_RecommenderWrapper import SpectralCF_RecommenderWrapper

from Utils.print_results_latex_table import print_time_statistics_latex_table, print_results_latex_table, print_parameters_latex_table


from functools import partial
import numpy as np
import os, traceback, multiprocessing

from Conferences.RecSys.SpectralCF_our_interface.Movielens1M.Movielens1MReader import Movielens1MReader
from Conferences.RecSys.SpectralCF_our_interface.MovielensHetrec2011.MovielensHetrec2011Reader import MovielensHetrec2011Reader
from Conferences.RecSys.SpectralCF_our_interface.AmazonInstantVideo.AmazonInstantVideoReader import AmazonInstantVideoReader



def read_data_split_and_search_SpectralCF(dataset_name, cold_start=False, cold_items=None, isKNN_multiprocess=True, isKNN_tune=True, isSpectralCF_train=True, print_results=True):


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
        output_folder_path = "result_experiments/RecSys/SpectralCF_{}/".format(dataset_name)
    else:
        output_folder_path = "result_experiments/RecSys/SpectralCF_cold_{}_{}/".format(cold_items, dataset_name)


    URM_train = dataset.URM_train.copy()
    URM_validation = dataset.URM_validation.copy()
    URM_test = dataset.URM_test.copy()

    # Ensure IMPLICIT data and DISJOINT sets
    from Utils.assertions_on_data_for_experiments import assert_implicit_data, assert_disjoint_matrices

    assert_implicit_data([URM_train, URM_validation, URM_test])
    if not cold_start: assert_disjoint_matrices([URM_train, URM_validation, URM_test])


    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)


    from Utils.plot_popularity import plot_popularity_bias, save_popularity_statistics

    plot_popularity_bias([URM_train + URM_validation, URM_test],
                         ["URM_train", "URM_test"],
                         output_folder_path + "_plot_popularity_bias")

    save_popularity_statistics([URM_train + URM_validation, URM_test],
                               ["URM_train", "URM_test"],
                               output_folder_path + "_latex_popularity_statistics")

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
            PureSVDRecommender,
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

    if isSpectralCF_train:
        try:


            temp_file_folder = output_folder_path + "SpectralCF_log/"


            spectralCF_article_parameters = {
                "epochs": 1000,
                "batch_size": 1024,
                "embedding_size": 16,
                "decay": 0.001,
                "k": 3,
                "learning_rate": 1e-3,
            }

            spectralCF_earlystopping_parameters = {
                "validation_every_n": 10,
                "stop_on_validation": True,
                "lower_validations_allowed": 20,
                "evaluator_object": evaluator_validation,
                "validation_metric": metric_to_optimize
            }

            parameterSearch = SearchSingleCase(SpectralCF_RecommenderWrapper,
                                               evaluator_validation=evaluator_validation,
                                               evaluator_test=evaluator_test)

            recommender_parameters = SearchInputRecommenderParameters(
                                                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                                                FIT_KEYWORD_ARGS = spectralCF_earlystopping_parameters)

            parameterSearch.search(recommender_parameters,
                                   fit_parameters_values=spectralCF_article_parameters,
                                   output_folder_path = output_folder_path,
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
            results_file_root_name = "SpectralCF"
        else:
            results_file_root_name = "SpectralCF_cold_{}".format(cold_items)



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


        print_results_latex_table(result_folder_path = output_folder_path,
                                  results_file_prefix_name = results_file_root_name + "_all_metrics",
                                  dataset_name = dataset_name,
                                  metrics_to_report_list = ["PRECISION", "RECALL", "MAP", "MRR", "NDCG", "F1", "HIT_RATE", "ARHR", "NOVELTY", "DIVERSITY_MEAN_INTER_LIST", "DIVERSITY_HERFINDAHL", "COVERAGE_ITEM", "DIVERSITY_GINI", "SHANNON_ENTROPY"],
                                  cutoffs_to_report_list = [50],
                                  other_algorithm_list = [SpectralCF_RecommenderWrapper])




if __name__ == '__main__':

    isKNN_multiprocess = True
    isKNN_tune = True
    isSpectralCF_train = True
    print_results = True
    print_popularity_bias_movielens1m = False

    cold_start = False

    dataset_list = ["movielens1m_ours", "movielens1m_original", "hetrec", "amazon_instant_video"]
    dataset_cold_start_list = ["movielens1m_ours"]
    cold_start_items_list = [1, 2, 3, 4, 5]

    if cold_start:
        for dataset_name in dataset_cold_start_list:
           for cold_start_items in cold_start_items_list:
               read_data_split_and_search_SpectralCF(dataset_name, cold_start=cold_start, cold_items=cold_start_items,
                                                     isKNN_multiprocess=isKNN_multiprocess, isKNN_tune=isKNN_tune, isSpectralCF_train=isSpectralCF_train, print_results=print_results
                                                     )


    else:
        for dataset_name in dataset_list:
            read_data_split_and_search_SpectralCF(dataset_name, cold_start=cold_start,
                                                  isKNN_multiprocess=isKNN_multiprocess, isKNN_tune=isKNN_tune, isSpectralCF_train=isSpectralCF_train, print_results=print_results
                                                  )


    # mantain compatibility with latex parameteres function
    if cold_start:
        for n_cold_item in cold_start_items_list:
            print_parameters_latex_table(result_folder_path = "result_experiments/RecSys/",
                                              results_file_prefix_name = "SpectralCF_cold_{}".format(n_cold_item),
                                              experiment_subfolder_list = dataset_cold_start_list,
                                              other_algorithm_list = [SpectralCF_RecommenderWrapper])
    else:
        print_parameters_latex_table(result_folder_path = "result_experiments/RecSys/",
                                       results_file_prefix_name = "SpectralCF",
                                       experiment_subfolder_list = dataset_list,
                                       other_algorithm_list = [SpectralCF_RecommenderWrapper])




    if print_popularity_bias_movielens1m:
        from Utils.plot_popularity import plot_popularity_bias

        dataset = Movielens1MReader(type="original")

        URM_train_original = dataset.URM_train.copy()
        URM_validation_original = dataset.URM_validation.copy()
        URM_test_original = dataset.URM_test.copy()

        dataset = Movielens1MReader(type="ours")
        URM_train_ours = dataset.URM_train.copy()
        URM_validation_ours = dataset.URM_validation.copy()
        URM_test_ours = dataset.URM_test.copy()


        plot_popularity_bias([URM_train_original, URM_train_ours],
                             ["URM_train_original", "URM_train_ours"],
                             "result_experiments/RecSys/SpectralCF_popularity_bias_train")

        plot_popularity_bias([URM_validation_original, URM_validation_ours],
                             ["URM_validation_original", "URM_validation_ours"],
                             "result_experiments/RecSys/SpectralCF_popularity_bias_validation")

        plot_popularity_bias([URM_test_original, URM_test_ours],
                             ["URM_test_original", "URM_test_ours"],
                             "result_experiments/RecSys/SpectralCF_popularity_bias_test")