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


from ParameterTuning.run_parameter_search import runParameterSearch_Collaborative
from ParameterTuning.SearchSingleCase import SearchSingleCase
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderParameters

import traceback, multiprocessing, os
from functools import partial
import numpy as np

from Utils.print_results_latex_table import print_time_statistics_latex_table, print_results_latex_table, print_parameters_latex_table


from Conferences.WWW.NeuCF_our_interface.NeuCF_RecommenderWrapper import NeuCF_RecommenderWrapper





def read_data_split_and_search_NeuCF(dataset_name):


    from Conferences.WWW.NeuCF_our_interface.Movielens1M.Movielens1MReader import Movielens1MReader
    from Conferences.WWW.NeuCF_our_interface.Pinterest.PinterestICCVReader import PinterestICCVReader

    if dataset_name == "movielens1m":
        dataset = Movielens1MReader()

    elif dataset_name == "pinterest":
        dataset = PinterestICCVReader()

    output_folder_path = "result_experiments/{}/{}_{}/".format(CONFERENCE_NAME, ALGORITHM_NAME, dataset_name)


    URM_train = dataset.URM_train.copy()
    URM_validation = dataset.URM_validation.copy()
    URM_test = dataset.URM_test.copy()
    URM_test_negative = dataset.URM_test_negative.copy()


    # Ensure IMPLICIT data and DISJOINT sets
    from Utils.assertions_on_data_for_experiments import assert_implicit_data, assert_disjoint_matrices

    assert_implicit_data([URM_train, URM_validation, URM_test, URM_test_negative])


    assert_disjoint_matrices([URM_train, URM_validation, URM_test])
    assert_disjoint_matrices([URM_train, URM_validation, URM_test_negative])


    from Utils.plot_popularity import plot_popularity_bias, save_popularity_statistics

    plot_popularity_bias([URM_train + URM_validation, URM_test],
                         ["URM_train", "URM_test"],
                         output_folder_path + "_plot_popularity_bias")

    save_popularity_statistics([URM_train + URM_validation, URM_test],
                               ["URM_train", "URM_test"],
                               output_folder_path + "_latex_popularity_statistics")


    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)




    collaborative_algorithm_list = [
        Random,
        TopPop,
        UserKNNCFRecommender,
        ItemKNNCFRecommender,
        P3alphaRecommender,
        RP3betaRecommender,
        # SLIM_BPR_Cython,
        # SLIMElasticNetRecommender,
        # MatrixFactorization_BPR_Cython,
        # MatrixFactorization_FunkSVD_Cython,
        PureSVDRecommender,
    ]

    metric_to_optimize = "HIT_RATE"


    from Base.Evaluation.Evaluator import EvaluatorNegativeItemSample

    evaluator_validation = EvaluatorNegativeItemSample(URM_validation, URM_test_negative, cutoff_list=[10])
    evaluator_test = EvaluatorNegativeItemSample(URM_test, URM_test_negative, cutoff_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


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





    # pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
    # resultList = pool.map(runParameterSearch_Collaborative_partial, collaborative_algorithm_list)
    #
    # pool.close()
    # pool.join()
    #
    #
    # for recommender_class in collaborative_algorithm_list:
    #
    #     try:
    #
    #         runParameterSearch_Collaborative_partial(recommender_class)
    #
    #     except Exception as e:
    #
    #         print("On recommender {} Exception {}".format(recommender_class, str(e)))
    #         traceback.print_exc()
    #


    ################################################################################################
    ###### NeuCF



    try:

        neuCF_article_parameters = {
            "epochs": 100,
            "epochs_gmf": 100,
            "epochs_mlp": 100,
            "batch_size": 256,
            "num_factors": 8,
            "layers": [64,32,16,8],
            "reg_mf": 0.0,
            "reg_layers": [0,0,0,0],
            "num_negatives": 4,
            "learning_rate": 1e-3,
            "learner": "adam",
            "pretrain": True
        }

        neuCF_earlystopping_parameters = {
            "validation_every_n": 5,
            "stop_on_validation": True,
            "evaluator_object": evaluator_validation,
            "lower_validations_allowed": 5,
            "validation_metric": metric_to_optimize
        }



        parameterSearch = SearchSingleCase(NeuCF_RecommenderWrapper,
                                           evaluator_validation=evaluator_validation,
                                           evaluator_test=evaluator_test)

        recommender_parameters = SearchInputRecommenderParameters(
                                            CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                                            FIT_KEYWORD_ARGS = neuCF_earlystopping_parameters)

        # parameterSearch.search(recommender_parameters,
        #                        fit_parameters_values=neuCF_article_parameters,
        #                        output_folder_path = output_folder_path,
        #                        output_file_name_root = NeuCF_RecommenderWrapper.RECOMMENDER_NAME)



    except Exception as e:

        print("On recommender {} Exception {}".format(NeuCF_RecommenderWrapper, str(e)))
        traceback.print_exc()










    n_validation_users = np.sum(np.ediff1d(URM_validation.indptr)>=1)
    n_test_users = np.sum(np.ediff1d(URM_test.indptr)>=1)


    print_time_statistics_latex_table(result_folder_path = output_folder_path,
                                      dataset_name = dataset_name,
                                      results_file_prefix_name = ALGORITHM_NAME,
                                      other_algorithm_list = [NeuCF_RecommenderWrapper],
                                      n_validation_users = n_validation_users,
                                      n_test_users = n_test_users,
                                      n_decimals = 2)


    print_results_latex_table(result_folder_path = output_folder_path,
                              results_file_prefix_name = ALGORITHM_NAME,
                              dataset_name = dataset_name,
                              metrics_to_report_list = ["HIT_RATE", "NDCG"],
                              cutoffs_to_report_list = [1, 5, 10],
                              other_algorithm_list = [NeuCF_RecommenderWrapper])



    print_results_latex_table(result_folder_path = output_folder_path,
                              results_file_prefix_name = ALGORITHM_NAME + "_all_metrics",
                              dataset_name = dataset_name,
                              metrics_to_report_list = ["PRECISION", "RECALL", "MAP", "MRR", "NDCG", "F1", "HIT_RATE", "ARHR", "NOVELTY", "DIVERSITY_MEAN_INTER_LIST", "DIVERSITY_HERFINDAHL", "COVERAGE_ITEM", "DIVERSITY_GINI", "SHANNON_ENTROPY"],
                              cutoffs_to_report_list = [10],
                              other_algorithm_list = [NeuCF_RecommenderWrapper])



if __name__ == '__main__':

    ALGORITHM_NAME = "NeuCF"
    CONFERENCE_NAME = "WWW"

    dataset_list = ["movielens1m", "pinterest"]

    for dataset in dataset_list:

        read_data_split_and_search_NeuCF(dataset)




    print_parameters_latex_table(result_folder_path = "result_experiments/{}/".format(CONFERENCE_NAME),
                                  results_file_prefix_name = ALGORITHM_NAME,
                                  experiment_subfolder_list = dataset_list,
                                  other_algorithm_list = [NeuCF_RecommenderWrapper])
