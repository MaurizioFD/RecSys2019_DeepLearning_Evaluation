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
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender


from KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender

from MatrixFactorization.PureSVDRecommender import PureSVDRecommender


from ParameterTuning.run_parameter_search import runParameterSearch_Collaborative, runParameterSearch_Content, runParameterSearch_Hybrid

from Conferences.KDD.CollaborativeVAE_our_interface.CollaborativeVAE_RecommenderWrapper import CollaborativeVAE_RecommenderWrapper

from ParameterTuning.SearchSingleCase import SearchSingleCase
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderParameters

from functools import partial
import os, traceback
import numpy as np



from Utils.print_results_latex_table import print_time_statistics_latex_table, print_results_latex_table, print_parameters_latex_table








def read_data_split_and_search_CollaborativeVAE(dataset_variant, train_interactions):



    from Conferences.KDD.CollaborativeVAE_our_interface.Citeulike.CiteulikeReader import CiteulikeReader

    dataset = CiteulikeReader(dataset_variant = dataset_variant, train_interactions = train_interactions)



    output_folder_path = "result_experiments/{}/{}_citeulike_{}_{}/".format(CONFERENCE_NAME, ALGORITHM_NAME, dataset_variant, train_interactions)


    URM_train = dataset.URM_train.copy()
    URM_validation = dataset.URM_validation.copy()
    URM_test = dataset.URM_test.copy()


    # Ensure IMPLICIT data
    from Utils.assertions_on_data_for_experiments import assert_implicit_data

    assert_implicit_data([URM_train, URM_validation, URM_test])




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

    metric_to_optimize = "RECALL"


    from Base.Evaluation.Evaluator import EvaluatorHoldout

    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[150])
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[50, 100, 150, 200, 250, 300])


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
    ###### Content Baselines

    ICM_title_abstract = dataset.ICM_title_abstract.copy()

    #
    # try:
    #
    #     runParameterSearch_Content(ItemKNNCBFRecommender,
    #                                URM_train = URM_train,
    #                                metric_to_optimize = metric_to_optimize,
    #                                evaluator_validation = evaluator_validation,
    #                                evaluator_test = evaluator_test,
    #                                output_folder_path = output_folder_path,
    #                                parallelizeKNN = False,
    #                                ICM_name = "ICM_title_abstract",
    #                                ICM_object = ICM_title_abstract,
    #                                allow_weighting = False,
    #                                n_cases = 35)
    #
    # except Exception as e:
    #
    #     print("On recommender {} Exception {}".format(ItemKNNCBFRecommender, str(e)))
    #     traceback.print_exc()


    ################################################################################################
    ###### Hybrid

    # try:
    #
    #     runParameterSearch_Hybrid(ItemKNN_CFCBF_Hybrid_Recommender,
    #                                URM_train = URM_train,
    #                                metric_to_optimize = metric_to_optimize,
    #                                evaluator_validation = evaluator_validation,
    #                                evaluator_test = evaluator_test,
    #                                output_folder_path = output_folder_path,
    #                                parallelizeKNN = False,
    #                                ICM_name = "ICM_title_abstract",
    #                                ICM_object = ICM_title_abstract,
    #                                allow_weighting = True,
    #                                n_cases = 35)
    #
    #
    # except Exception as e:
    #
    #     print("On recommender {} Exception {}".format(ItemKNN_CFCBF_Hybrid_Recommender, str(e)))
    #     traceback.print_exc()




    ################################################################################################
    ###### CollaborativeVAE

    try:

        temp_file_folder = output_folder_path + "{}_log/".format(ALGORITHM_NAME)

        cvae_recommender_article_parameters = {
            "epochs": 200,
            "learning_rate_vae": 1e-2,
            "learning_rate_cvae": 1e-3,
            "num_factors": 50,
            "dimensions_vae": [200, 100],
            "epochs_vae": [50, 50],
            "batch_size": 128,
            "lambda_u": 0.1,
            "lambda_v": 10,
            "lambda_r": 1,
            "a": 1,
            "b": 0.01,
            "M": 300,
            "temp_file_folder": temp_file_folder
        }



        cvae_earlystopping_parameters = {
            "validation_every_n": 5,
            "stop_on_validation": True,
            "evaluator_object": evaluator_validation,
            "lower_validations_allowed": 5,
            "validation_metric": metric_to_optimize
        }

        parameterSearch = SearchSingleCase(CollaborativeVAE_RecommenderWrapper,
                                           evaluator_validation=evaluator_validation,
                                           evaluator_test=evaluator_test)

        recommender_parameters = SearchInputRecommenderParameters(
                                            CONSTRUCTOR_POSITIONAL_ARGS = [URM_train, ICM_title_abstract],
                                            FIT_KEYWORD_ARGS = cvae_earlystopping_parameters)

        # parameterSearch.search(recommender_parameters,
        #                        fit_parameters_values=cvae_recommender_article_parameters,
        #                        output_folder_path = output_folder_path,
        #                        output_file_name_root = CollaborativeVAE_RecommenderWrapper.RECOMMENDER_NAME)




    except Exception as e:

        print("On recommender {} Exception {}".format(CollaborativeVAE_RecommenderWrapper, str(e)))
        traceback.print_exc()






    n_validation_users = np.sum(np.ediff1d(URM_validation.indptr)>=1)
    n_test_users = np.sum(np.ediff1d(URM_test.indptr)>=1)

    ICM_names_to_report_list = ["ICM_title_abstract"]
    dataset_name = "{}_{}".format(dataset_variant, train_interactions)


    print_time_statistics_latex_table(result_folder_path = output_folder_path,
                                      dataset_name = dataset_name,
                                      results_file_prefix_name = ALGORITHM_NAME,
                                      other_algorithm_list = [CollaborativeVAE_RecommenderWrapper],
                                      ICM_names_to_report_list = ICM_names_to_report_list,
                                      n_validation_users = n_validation_users,
                                      n_test_users = n_test_users,
                                      n_decimals = 2)


    print_results_latex_table(result_folder_path = output_folder_path,
                              results_file_prefix_name = ALGORITHM_NAME,
                              dataset_name = dataset_name,
                              metrics_to_report_list = ["RECALL"],
                              cutoffs_to_report_list = [50, 100, 150, 200, 250, 300],
                              ICM_names_to_report_list = ICM_names_to_report_list,
                              other_algorithm_list = [CollaborativeVAE_RecommenderWrapper])



    print_results_latex_table(result_folder_path = output_folder_path,
                              results_file_prefix_name = ALGORITHM_NAME + "_all_metrics",
                              dataset_name = dataset_name,
                              metrics_to_report_list = ["PRECISION", "RECALL", "MAP", "MRR", "NDCG", "F1", "HIT_RATE", "ARHR", "NOVELTY", "DIVERSITY_MEAN_INTER_LIST", "DIVERSITY_HERFINDAHL", "COVERAGE_ITEM", "DIVERSITY_GINI", "SHANNON_ENTROPY"],
                              cutoffs_to_report_list = [150],
                              ICM_names_to_report_list = ICM_names_to_report_list,
                              other_algorithm_list = [CollaborativeVAE_RecommenderWrapper])



if __name__ == '__main__':

    ALGORITHM_NAME = "CollaborativeVAE"
    CONFERENCE_NAME = "KDD"


    dataset_variant_list = ["a", "t"]
    train_interactions_list = [1, 10]

    for dataset_variant in dataset_variant_list:

        for train_interactions in train_interactions_list:

            read_data_split_and_search_CollaborativeVAE(dataset_variant, train_interactions)




    print_parameters_latex_table(result_folder_path = "result_experiments/{}/".format(CONFERENCE_NAME),
                                  results_file_prefix_name = ALGORITHM_NAME,
                                  experiment_subfolder_list = [
                                      "citeulike_{}_{}".format(dataset_variant, train_interactions) for dataset_variant in dataset_variant_list for train_interactions in train_interactions_list
                                  ],
                                  ICM_names_to_report_list = ["ICM_title_abstract"],
                                  other_algorithm_list = [CollaborativeVAE_RecommenderWrapper])
