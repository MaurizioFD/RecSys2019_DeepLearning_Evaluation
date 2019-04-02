#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17

@author: Anonymous authors
"""


from Base.NonPersonalizedRecommender import TopPop, Random
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from SLIM_ElasticNet.Cython.SLIM_Structure_Cython import SLIM_Structure_Cython
from GraphBased.P3alphaRecommender import P3alphaRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender


from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender

from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython
from MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from MatrixFactorization.WRMFRecommender import WRMFRecommender

from ParameterTuning.SearchSingleCase import SearchSingleCase
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderParameters

from ParameterTuning.run_parameter_search import runParameterSearch_Collaborative
# from Utils.print_results_latex_table import print_time_statistics_latex_table, print_results_latex_table, print_parameters_latex_table


from functools import partial
import traceback, os, multiprocessing
import numpy as np

from Conferences.WWW.MultiVAE_our_interface.EvaluatorUserSubsetWrapper import EvaluatorUserSubsetWrapper
# from Conferences.WWW.MultiVAE_our_interface.MultiVAE_RecommenderWrapper import MultiVAE_RecommenderWrapper





def read_data_split_and_search_MultiVAE(dataset_name):


    from Conferences.WWW.MultiVAE_our_interface.Movielens20M.Movielens20MReader import Movielens20MReader
    from Conferences.WWW.MultiVAE_our_interface.NetflixPrize.NetflixPrizeReader import NetflixPrizeReader

    split_type = "cold_user"

    if dataset_name == "movielens20m":
        dataset = Movielens20MReader(split_type = split_type)

    elif dataset_name == "netflixPrize":
        dataset = NetflixPrizeReader()


    output_folder_path = "result_experiments/{}/{}_{}_{}/".format(CONFERENCE_NAME, ALGORITHM_NAME, dataset_name, split_type)


    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    metric_to_optimize = "NDCG"


    if split_type == "cold_user":


        collaborative_algorithm_list = [
            # Random,
            # TopPop,
            # # Non applicabile, new users - UserKNNCFRecommender,
            # ItemKNNCFRecommender,
            # P3alphaRecommender,
            # RP3betaRecommender,
            SLIM_BPR_Cython,
            SLIMElasticNetRecommender,
            # Non applicabile, new users - MatrixFactorization_BPR_Cython,
            # Non applicabile, new users - MatrixFactorization_FunkSVD_Cython,
            # Non applicabile, new users - PureSVDRecommender,
            MatrixFactorization_AsySVD_Cython,
            MatrixFactorization_BPR_Cython,
            MatrixFactorization_FunkSVD_Cython,
            PureSVDRecommender,
            WRMFRecommender
        ]

        URM_train = dataset.URM_train.copy()
        URM_train_all = dataset.URM_train_all.copy()
        URM_validation = dataset.URM_validation.copy()
        URM_test = dataset.URM_test.copy()



        # Ensure IMPLICIT data and DISJOINT sets
        from Utils.assertions_on_data_for_experiments import assert_implicit_data, assert_disjoint_matrices

        assert_implicit_data([URM_train, URM_train_all, URM_validation, URM_test])
        assert_disjoint_matrices([URM_train, URM_validation, URM_test])
        assert_disjoint_matrices([URM_train_all, URM_validation, URM_test])


        from Base.Evaluation.Evaluator import EvaluatorHoldout

        evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[100])
        evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[20, 50, 100])

        evaluator_validation = EvaluatorUserSubsetWrapper(evaluator_validation, URM_train_all)
        evaluator_test = EvaluatorUserSubsetWrapper(evaluator_test, URM_train_all)




    elif split_type == "warm_user":



        collaborative_algorithm_list = [
            #Random,
            TopPop,
            #UserKNNCFRecommender,
            #ItemKNNCFRecommender,
            #P3alphaRecommender,
            #RP3betaRecommender,
            #SLIM_BPR_Cython,
            # SLIMElasticNetRecommender,
            # MatrixFactorization_BPR_Cython,
            # MatrixFactorization_FunkSVD_Cython,
            #PureSVDRecommender,
        ]


        URM_train = dataset.URM_train.copy()
        URM_validation = dataset.URM_validation.copy()
        URM_test = dataset.URM_test.copy()

        # Ensure IMPLICIT data
        from Utils.assertions_on_data_for_experiments import assert_implicit_data

        assert_implicit_data([URM_train, URM_validation, URM_test])


        from Base.Evaluation.Evaluator import EvaluatorHoldout

        evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[100])
        evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[20, 50, 100])



    else:

        assert False, "split_type not recognized"






    # from Utils.plot_popularity import plot_popularity_bias, save_popularity_statistics
    #
    # plot_popularity_bias([URM_train + URM_validation, URM_test],
    #                      ["URM_train", "URM_test"],
    #                      output_folder_path + "_plot_popularity_bias")
    #
    # save_popularity_statistics([URM_train + URM_validation, URM_test],
    #                            ["URM_train", "URM_test"],
    #                            output_folder_path + "_latex_popularity_statistics")




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





    pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
    pool.map(runParameterSearch_Collaborative_partial, collaborative_algorithm_list)

    pool.close()
    pool.join()

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



    ################################################################################################
    ###### MultiVAE



    try:

        output_root_path_MultiVAE = output_folder_path + "{}_log/".format(ALGORITHM_NAME)

        if dataset_name == "movielens20m":
            epochs = 100

        elif dataset_name == "netflixPrize":
            epochs = 200


        multiVAE_article_parameters = {
            "epochs": epochs,
            "batch_size": 500,
            "total_anneal_steps": 200000,
            "p_dims": None,
        }

        multiVAE_earlystopping_parameters = {
            "validation_every_n": 5,
            "stop_on_validation": True,
            "evaluator_object": evaluator_validation,
            "lower_validations_allowed": 5,
            "validation_metric": metric_to_optimize,
            "temp_file_folder": output_root_path_MultiVAE
        }


        parameterSearch = SearchSingleCase(MultiVAE_RecommenderWrapper,
                                           evaluator_validation=evaluator_validation,
                                           evaluator_test=evaluator_test)

        recommender_parameters = SearchInputRecommenderParameters(
                                            CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                                            FIT_KEYWORD_ARGS = multiVAE_earlystopping_parameters)

        # parameterSearch.search(recommender_parameters,
        #                        fit_parameters_values=multiVAE_article_parameters,
        #                        output_folder_path = output_folder_path,
        #                        output_file_name_root = MultiVAE_RecommenderWrapper.RECOMMENDER_NAME)



    except Exception as e:

        print("On recommender {} Exception {}".format(MultiVAE_RecommenderWrapper, str(e)))
        traceback.print_exc()



    n_validation_users = np.sum(np.ediff1d(URM_validation.indptr)>=1)
    n_test_users = np.sum(np.ediff1d(URM_test.indptr)>=1)


    print_time_statistics_latex_table(result_folder_path = output_folder_path,
                                      dataset_name = dataset_name,
                                      results_file_prefix_name = ALGORITHM_NAME,
                                      other_algorithm_list = [MultiVAE_RecommenderWrapper],
                                      n_validation_users = n_validation_users,
                                      n_test_users = n_test_users,
                                      n_decimals = 2)


    print_results_latex_table(result_folder_path = output_folder_path,
                              results_file_prefix_name = ALGORITHM_NAME,
                              dataset_name = dataset_name,
                              metrics_to_report_list = ["RECALL", "NDCG"],
                              cutoffs_to_report_list = [20, 50, 100],
                              other_algorithm_list = [MultiVAE_RecommenderWrapper])


    print_results_latex_table(result_folder_path = output_folder_path,
                              results_file_prefix_name = ALGORITHM_NAME + "_all_metrics",
                              dataset_name = dataset_name,
                              metrics_to_report_list = ["PRECISION", "RECALL", "MAP", "MRR", "NDCG", "F1", "HIT_RATE", "ARHR", "NOVELTY", "DIVERSITY_MEAN_INTER_LIST", "DIVERSITY_HERFINDAHL", "COVERAGE_ITEM", "DIVERSITY_GINI", "SHANNON_ENTROPY"],
                              cutoffs_to_report_list = [50],
                              other_algorithm_list = [MultiVAE_RecommenderWrapper])





from functools import partial






if __name__ == '__main__':

    ALGORITHM_NAME = "MultiVAE"
    CONFERENCE_NAME = "WWW"


    dataset_list = ["movielens20m", "netflixPrize"]
    dataset_list = ["netflixPrize"]


    for dataset in dataset_list:

        read_data_split_and_search_MultiVAE(dataset)


    print_parameters_latex_table(result_folder_path = "result_experiments/{}/".format(CONFERENCE_NAME),
                                  results_file_prefix_name = ALGORITHM_NAME,
                                  experiment_subfolder_list = ["{}_cold_user".format(dataset) for dataset in dataset_list],
                                  other_algorithm_list = [MultiVAE_RecommenderWrapper])
