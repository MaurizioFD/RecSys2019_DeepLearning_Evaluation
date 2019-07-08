#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17

@author: Maurizio Ferrari Dacrema
"""

from Recommender_import_list import *
from Conferences.SIGIR.CMN_our_interface.CMN_RecommenderWrapper import CMN_RecommenderWrapper


from ParameterTuning.run_parameter_search import runParameterSearch_Collaborative

import traceback, os, multiprocessing, pickle
from functools import partial
import numpy as np


from ParameterTuning.SearchSingleCase import SearchSingleCase
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderParameters

from Utils.print_results_latex_table import print_time_statistics_latex_table, print_results_latex_table, print_parameters_latex_table
from Utils.assertions_on_data_for_experiments import assert_implicit_data, assert_disjoint_matrices
from Utils.plot_popularity import plot_popularity_bias, save_popularity_statistics


def read_data_split_and_search_CMN(dataset_name):

    from Conferences.SIGIR.CMN_our_interface.CiteULike.CiteULikeReader import CiteULikeReader
    from Conferences.SIGIR.CMN_our_interface.Pinterest.PinterestICCVReader import PinterestICCVReader
    from Conferences.SIGIR.CMN_our_interface.Epinions.EpinionsReader import EpinionsReader

    if dataset_name == "citeulike":
        dataset = CiteULikeReader()

    elif dataset_name == "epinions":
        dataset = EpinionsReader()

    elif dataset_name == "pinterest":
        dataset = PinterestICCVReader()

    output_folder_path = "result_experiments/{}/{}_{}/".format(CONFERENCE_NAME, ALGORITHM_NAME, dataset_name)


    URM_train = dataset.URM_train.copy()
    URM_validation = dataset.URM_validation.copy()
    URM_test = dataset.URM_test.copy()
    URM_test_negative = dataset.URM_test_negative.copy()



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
    ]

    metric_to_optimize = "HIT_RATE"


    # Ensure IMPLICIT data and DISJOINT sets
    assert_implicit_data([URM_train, URM_validation, URM_test, URM_test_negative])


    if dataset_name == "citeulike":
        assert_disjoint_matrices([URM_train, URM_validation, URM_test])
        assert_disjoint_matrices([URM_test, URM_test_negative])

    elif dataset_name == "pinterest":
        assert_disjoint_matrices([URM_train, URM_validation, URM_test])
        assert_disjoint_matrices([URM_train, URM_validation, URM_test_negative])

    else:
        assert_disjoint_matrices([URM_train, URM_validation, URM_test, URM_test_negative])



    algorithm_dataset_string = "{}_{}_".format(ALGORITHM_NAME, dataset_name)

    plot_popularity_bias([URM_train + URM_validation, URM_test],
                         ["URM train", "URM test"],
                         output_folder_path + algorithm_dataset_string + "popularity_plot")

    save_popularity_statistics([URM_train + URM_validation, URM_test],
                               ["URM train", "URM test"],
                               output_folder_path + algorithm_dataset_string + "popularity_statistics")



    from Base.Evaluation.Evaluator import EvaluatorNegativeItemSample

    evaluator_validation = EvaluatorNegativeItemSample(URM_validation, URM_test_negative, cutoff_list=[5])
    evaluator_test = EvaluatorNegativeItemSample(URM_test, URM_test_negative, cutoff_list=[5, 10])


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


    for recommender_class in collaborative_algorithm_list:

        try:

            runParameterSearch_Collaborative_partial(recommender_class)

        except Exception as e:

            print("On recommender {} Exception {}".format(recommender_class, str(e)))
            traceback.print_exc()



    ################################################################################################
    ###### CMN




    try:


        temp_file_folder = output_folder_path + "{}_log/".format(ALGORITHM_NAME)

        CMN_article_parameters = {
            "epochs": 100,
            "epochs_gmf": 100,
            "hops": 3,
            "neg_samples": 4,
            "reg_l2_cmn": 1e-1,
            "reg_l2_gmf": 1e-4,
            "pretrain": True,
            "learning_rate": 1e-3,
            "verbose": False,
            "temp_file_folder": temp_file_folder
        }

        if dataset_name == "citeulike":
            CMN_article_parameters["batch_size"] = 128
            CMN_article_parameters["embed_size"] = 50

        elif dataset_name == "epinions":
            CMN_article_parameters["batch_size"] = 128
            CMN_article_parameters["embed_size"] = 40

        elif dataset_name == "pinterest":
            CMN_article_parameters["batch_size"] = 256
            CMN_article_parameters["embed_size"] = 50



        CMN_earlystopping_parameters = {
            "validation_every_n": 5,
            "stop_on_validation": True,
            "evaluator_object": evaluator_validation,
            "lower_validations_allowed": 5,
            "validation_metric": metric_to_optimize
        }


        parameterSearch = SearchSingleCase(CMN_RecommenderWrapper,
                                           evaluator_validation=evaluator_validation,
                                           evaluator_test=evaluator_test)

        recommender_parameters = SearchInputRecommenderParameters(
                                            CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                                            FIT_KEYWORD_ARGS = CMN_earlystopping_parameters)

        parameterSearch.search(recommender_parameters,
                               fit_parameters_values=CMN_article_parameters,
                               output_folder_path = output_folder_path,
                               output_file_name_root = CMN_RecommenderWrapper.RECOMMENDER_NAME)




    except Exception as e:

        print("On recommender {} Exception {}".format(CMN_RecommenderWrapper, str(e)))
        traceback.print_exc()





    n_validation_users = np.sum(np.ediff1d(URM_validation.indptr)>=1)
    n_test_users = np.sum(np.ediff1d(URM_test.indptr)>=1)


    print_time_statistics_latex_table(result_folder_path = output_folder_path,
                                      dataset_name = dataset_name,
                                      results_file_prefix_name = ALGORITHM_NAME,
                                      other_algorithm_list = [CMN_RecommenderWrapper],
                                      ICM_names_to_report_list = [],
                                      n_validation_users = n_validation_users,
                                      n_test_users = n_test_users,
                                      n_decimals = 2)


    print_results_latex_table(result_folder_path = output_folder_path,
                              results_file_prefix_name = ALGORITHM_NAME,
                              dataset_name = dataset_name,
                              metrics_to_report_list = ["HIT_RATE", "NDCG"],
                              cutoffs_to_report_list = [5, 10],
                              ICM_names_to_report_list = [],
                              other_algorithm_list = [CMN_RecommenderWrapper])



if __name__ == '__main__':

    ALGORITHM_NAME = "CMN"
    CONFERENCE_NAME = "SIGIR"

    dataset_list = ["citeulike", "pinterest", "epinions"]

    for dataset in dataset_list:

        read_data_split_and_search_CMN(dataset)




    print_parameters_latex_table(result_folder_path = "result_experiments/{}/".format(CONFERENCE_NAME),
                                  results_file_prefix_name = ALGORITHM_NAME,
                                  experiment_subfolder_list = dataset_list,
                                  ICM_names_to_report_list = [],
                                  other_algorithm_list = [CMN_RecommenderWrapper])
