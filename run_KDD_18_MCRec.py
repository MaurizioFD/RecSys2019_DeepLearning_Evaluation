#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17

@author: Maurizio Ferrari Dacrema
"""

from Recommender_import_list import *
from Conferences.KDD.MCRec_our_interface.MCRecRecommenderWrapper import MCRecML100k_RecommenderWrapper

from ParameterTuning.SearchSingleCase import SearchSingleCase
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderParameters

from ParameterTuning.run_parameter_search import runParameterSearch_Collaborative, runParameterSearch_Content, runParameterSearch_Hybrid

import os, traceback
import numpy as np


from Utils.print_results_latex_table import print_time_statistics_latex_table, print_results_latex_table, print_parameters_latex_table
from Utils.assertions_on_data_for_experiments import assert_implicit_data, assert_disjoint_matrices
from Utils.plot_popularity import plot_popularity_bias, save_popularity_statistics


def read_data_split_and_search_MCRec(dataset_name):

    from Conferences.KDD.MCRec_our_interface.Movielens100K.Movielens100KReader import Movielens100KReader


    if dataset_name == "movielens100k":
        dataset = Movielens100KReader()



    output_folder_path = "result_experiments/{}/{}_{}/".format(CONFERENCE_NAME, ALGORITHM_NAME, dataset_name)


    URM_train = dataset.URM_train.copy()
    URM_validation = dataset.URM_validation.copy()
    URM_test = dataset.URM_test.copy()
    URM_test_negative = dataset.URM_test_negative.copy()


    # Ensure IMPLICIT data
    assert_implicit_data([URM_train, URM_validation, URM_test, URM_test_negative])
    assert_disjoint_matrices([URM_train, URM_validation, URM_test, URM_test_negative])


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



    from Base.Evaluation.Evaluator import EvaluatorNegativeItemSample

    if dataset_name == "movielens100k":
        URM_train += URM_validation
        evaluator_validation = EvaluatorNegativeItemSample(URM_validation, URM_test_negative, cutoff_list=[10], exclude_seen=False)
    else:
        evaluator_validation = EvaluatorNegativeItemSample(URM_validation, URM_test_negative, cutoff_list=[10])


    evaluator_test = EvaluatorNegativeItemSample(URM_test, URM_test_negative, cutoff_list=[10])



    collaborative_algorithm_list = [
        Random,
        TopPop,
        UserKNNCFRecommender,
        ItemKNNCFRecommender,
        P3alphaRecommender,
        RP3betaRecommender,
        PureSVDRecommender
    ]

    metric_to_optimize = "PRECISION"



    runParameterSearch_Collaborative_partial = partial(runParameterSearch_Collaborative,
                                                       URM_train = URM_train,
                                                       metric_to_optimize = metric_to_optimize,
                                                       evaluator_validation_earlystopping = evaluator_validation,
                                                       evaluator_validation = evaluator_validation,
                                                       evaluator_test = evaluator_test,
                                                       output_folder_path = output_folder_path,
                                                       parallelizeKNN = False,
                                                       n_cases = 35)





    # pool = PoolWithSubprocess(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
    # resultList = pool.map(runParameterSearch_Collaborative_partial, collaborative_algorithm_list)

    # pool.close()
    # pool.join()




    for recommender_class in collaborative_algorithm_list:

        try:

            runParameterSearch_Collaborative_partial(recommender_class)

        except Exception as e:

            print("On recommender {} Exception {}".format(recommender_class, str(e)))
            traceback.print_exc()


    ################################################################################################
    ###### Content Baselines

    ICM_dictionary = dataset.ICM_dict

    ICM_name_list = ICM_dictionary.keys()


    for ICM_name in ICM_name_list:

        try:

            ICM_object = ICM_dictionary[ICM_name]

            runParameterSearch_Content(ItemKNNCBFRecommender,
                                       URM_train = URM_train,
                                       metric_to_optimize = metric_to_optimize,
                                       evaluator_validation = evaluator_validation,
                                       evaluator_test = evaluator_test,
                                       output_folder_path = output_folder_path,
                                       parallelizeKNN = False,
                                       ICM_name = ICM_name,
                                       ICM_object = ICM_object.copy(),
                                       n_cases = 35)

        except Exception as e:

            print("On CBF recommender for ICM {} Exception {}".format(ICM_name, str(e)))
            traceback.print_exc()


    ################################################################################################
    ###### Hybrid

    for ICM_name in ICM_name_list:

        try:

            ICM_object = ICM_dictionary[ICM_name]

            runParameterSearch_Hybrid(ItemKNN_CFCBF_Hybrid_Recommender,
                                       URM_train = URM_train,
                                       metric_to_optimize = metric_to_optimize,
                                       evaluator_validation = evaluator_validation,
                                       evaluator_test = evaluator_test,
                                       output_folder_path = output_folder_path,
                                       parallelizeKNN = False,
                                       ICM_name = ICM_name,
                                       ICM_object = ICM_object,
                                       allow_weighting = True,
                                       n_cases = 35)


        except Exception as e:

            print("On recommender {} Exception {}".format(ItemKNN_CFCBF_Hybrid_Recommender, str(e)))
            traceback.print_exc()

    ################################################################################################
    ###### MCRec

    if dataset_name == "movielens100k":

        # Since I am using the original Data reader, the content of URM_validation are seen items, therefore I have to set another
        # evaluator which does not exclude them
        # evaluator_validation = EvaluatorNegativeItemSample(URM_validation, URM_test_negative, cutoff_list=[10], exclude_seen=False)

        MCRec_article_parameters = {
            "epochs": 100,
            "latent_dim": 128,
            "reg_latent": 0,
            "layers": [512, 256, 128, 64],
            "reg_layes": [0 ,0, 0, 0],
            "learning_rate": 1e-3,
            "batch_size": 256,
            "num_negatives": 4,
        }

        MCRec_earlystopping_parameters = {
            "validation_every_n": 5,
            "stop_on_validation": True,
            "evaluator_object": evaluator_validation,
            "lower_validations_allowed": 5,
            "validation_metric": metric_to_optimize
        }



        parameterSearch = SearchSingleCase(MCRecML100k_RecommenderWrapper,
                                           evaluator_validation=evaluator_validation,
                                           evaluator_test=evaluator_test)

        recommender_parameters = SearchInputRecommenderParameters(
                                            CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                                            FIT_KEYWORD_ARGS = MCRec_earlystopping_parameters)

        parameterSearch.search(recommender_parameters,
                               fit_parameters_values=MCRec_article_parameters,
                               output_folder_path = output_folder_path,
                               output_file_name_root = MCRecML100k_RecommenderWrapper.RECOMMENDER_NAME)







    n_validation_users = np.sum(np.ediff1d(URM_validation.indptr)>=1)
    n_test_users = np.sum(np.ediff1d(URM_test.indptr)>=1)

    ICM_names_to_report_list = ["ICM_genre"]


    print_time_statistics_latex_table(result_folder_path = output_folder_path,
                                      dataset_name = dataset_name,
                                      results_file_prefix_name = ALGORITHM_NAME,
                                      other_algorithm_list = [MCRecML100k_RecommenderWrapper],
                                      ICM_names_to_report_list = ICM_names_to_report_list,
                                      n_validation_users = n_validation_users,
                                      n_test_users = n_test_users,
                                      n_decimals = 2)


    print_results_latex_table(result_folder_path = output_folder_path,
                              results_file_prefix_name = ALGORITHM_NAME,
                              dataset_name = dataset_name,
                              metrics_to_report_list = ["PRECISION", "RECALL", "NDCG"],
                              cutoffs_to_report_list = [10],
                              ICM_names_to_report_list = ICM_names_to_report_list,
                              other_algorithm_list = [MCRecML100k_RecommenderWrapper])



from functools import partial

if __name__ == '__main__':

    ALGORITHM_NAME = "MCRec"
    CONFERENCE_NAME = "KDD"

    dataset_list = ["movielens100k"]

    for dataset in dataset_list:

        read_data_split_and_search_MCRec(dataset)



    print_parameters_latex_table(result_folder_path = "result_experiments/{}/".format(CONFERENCE_NAME),
                                  results_file_prefix_name = ALGORITHM_NAME,
                                  experiment_subfolder_list = dataset_list,
                                  ICM_names_to_report_list = ["ICM_genre"],
                                  other_algorithm_list = [MCRecML100k_RecommenderWrapper])
