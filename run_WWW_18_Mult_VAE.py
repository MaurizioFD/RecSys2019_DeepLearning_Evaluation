#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17

@author: Maurizio Ferrari Dacrema
"""


from Recommender_import_list import *
from Conferences.WWW.MultiVAE_our_interface.MultiVAE_RecommenderWrapper import MultiVAE_RecommenderWrapper


from ParameterTuning.SearchSingleCase import SearchSingleCase
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderParameters

from ParameterTuning.run_parameter_search import runParameterSearch_Collaborative
from Utils.print_results_latex_table import print_time_statistics_latex_table, print_results_latex_table, print_parameters_latex_table


from functools import partial
import traceback, os, multiprocessing
import numpy as np

from Conferences.WWW.MultiVAE_our_interface.EvaluatorUserSubsetWrapper import EvaluatorUserSubsetWrapper
from Utils.assertions_on_data_for_experiments import assert_implicit_data, assert_disjoint_matrices



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
           Random,
            TopPop,
            ItemKNNCFRecommender,
            P3alphaRecommender,
            RP3betaRecommender,
        ]

        URM_train = dataset.URM_train.copy()
        URM_train_all = dataset.URM_train_all.copy()
        URM_validation = dataset.URM_validation.copy()
        URM_test = dataset.URM_test.copy()



        # Ensure IMPLICIT data and DISJOINT sets
        assert_implicit_data([URM_train, URM_train_all, URM_validation, URM_test])
        assert_disjoint_matrices([URM_train, URM_validation, URM_test])
        assert_disjoint_matrices([URM_train_all, URM_validation, URM_test])


        from Base.Evaluation.Evaluator import EvaluatorHoldout

        evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[100])
        evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[20, 50, 100])

        evaluator_validation = EvaluatorUserSubsetWrapper(evaluator_validation, URM_train_all)
        evaluator_test = EvaluatorUserSubsetWrapper(evaluator_test, URM_train_all)




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
    # pool.map(runParameterSearch_Collaborative_partial, collaborative_algorithm_list)
    #
    # pool.close()
    # pool.join()


    for recommender_class in collaborative_algorithm_list:

        try:

            runParameterSearch_Collaborative_partial(recommender_class)

        except Exception as e:

            print("On recommender {} Exception {}".format(recommender_class, str(e)))
            traceback.print_exc()



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

        parameterSearch.search(recommender_parameters,
                               fit_parameters_values=multiVAE_article_parameters,
                               output_folder_path = output_folder_path,
                               output_file_name_root = MultiVAE_RecommenderWrapper.RECOMMENDER_NAME)



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



from functools import partial






if __name__ == '__main__':

    ALGORITHM_NAME = "Mult_VAE"
    CONFERENCE_NAME = "WWW"


    dataset_list = ["movielens20m", "netflixPrize"]


    for dataset in dataset_list:

        read_data_split_and_search_MultiVAE(dataset)


    print_parameters_latex_table(result_folder_path = "result_experiments/{}/".format(CONFERENCE_NAME),
                                  results_file_prefix_name = ALGORITHM_NAME,
                                  experiment_subfolder_list = ["{}_cold_user".format(dataset) for dataset in dataset_list],
                                  other_algorithm_list = [MultiVAE_RecommenderWrapper])
