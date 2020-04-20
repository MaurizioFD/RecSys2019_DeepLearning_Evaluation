#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17

@author: Maurizio Ferrari Dacrema
"""

from Recommender_import_list import *
from Conferences.KDD.CollaborativeDL_our_interface.CollaborativeDL_Matlab_RecommenderWrapper import CollaborativeDL_Matlab_RecommenderWrapper

from ParameterTuning.run_parameter_search import runParameterSearch_Collaborative, runParameterSearch_Content, runParameterSearch_Hybrid

from ParameterTuning.SearchSingleCase import SearchSingleCase
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs


import os, traceback, argparse
import numpy as np

from Utils.ResultFolderLoader import ResultFolderLoader, generate_latex_hyperparameters
from Utils.assertions_on_data_for_experiments import assert_implicit_data, assert_disjoint_matrices





def read_data_split_and_search(dataset_variant, train_interactions,
                                   flag_baselines_tune = False,
                                   flag_DL_article_default = False, flag_DL_tune = False,
                                   flag_print_results = False):


    # Using dataReader from CollaborativeVAE_our_interface as they use the same data in the same way
    from Conferences.KDD.CollaborativeVAE_our_interface.Citeulike.CiteulikeReader import CiteulikeReader

    result_folder_path = "result_experiments/{}/{}_citeulike_{}_{}/".format(CONFERENCE_NAME, ALGORITHM_NAME, dataset_variant, train_interactions)
    result_folder_path_CollaborativeVAE = "result_experiments/{}/{}_citeulike_{}_{}/".format(CONFERENCE_NAME, "CollaborativeVAE", dataset_variant, train_interactions)

    dataset = CiteulikeReader(result_folder_path_CollaborativeVAE, dataset_variant = dataset_variant, train_interactions = train_interactions)

    URM_train = dataset.URM_DICT["URM_train"].copy()
    URM_validation = dataset.URM_DICT["URM_validation"].copy()
    URM_test = dataset.URM_DICT["URM_test"].copy()

    # Ensure IMPLICIT data
    assert_implicit_data([URM_train, URM_validation, URM_test])

    # Due to the sparsity of the dataset, choosing an evaluation as subset of the train
    # While keepning validation interaction in the train set
    if train_interactions == 1:
        # In this case the train data will contain validation data to avoid cold users
        assert_disjoint_matrices([URM_train, URM_test])
        assert_disjoint_matrices([URM_validation, URM_test])
        exclude_seen_validation = False
        URM_train_last_test = URM_train
    else:
        assert_disjoint_matrices([URM_train, URM_validation, URM_test])
        exclude_seen_validation = True
        URM_train_last_test = URM_train + URM_validation

    assert_implicit_data([URM_train_last_test])



    # If directory does not exist, create
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)



    from Base.Evaluation.Evaluator import EvaluatorHoldout

    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[150], exclude_seen = exclude_seen_validation)
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[50, 100, 150, 200, 250, 300])


    ################################################################################################
    ######
    ######      DL ALGORITHM
    ######

    if flag_DL_article_default:

        try:

            collaborativeDL_article_hyperparameters = {
                "para_lv": 10,
                "para_lu": 1,
                "para_ln": 1e3,
                "batch_size": 128,
                "epoch_sdae": 200,
                "epoch_dae": 200,
            }


            parameterSearch = SearchSingleCase(CollaborativeDL_Matlab_RecommenderWrapper,
                                               evaluator_validation=evaluator_validation,
                                               evaluator_test=evaluator_test)

            recommender_input_args = SearchInputRecommenderArgs(
                                                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train, dataset.ICM_DICT["ICM_tokens_TFIDF"]],
                                                FIT_KEYWORD_ARGS = {})

            recommender_input_args_last_test = recommender_input_args.copy()
            recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train_last_test


            parameterSearch.search(recommender_input_args,
                                   recommender_input_args_last_test = recommender_input_args_last_test,
                                   fit_hyperparameters_values=collaborativeDL_article_hyperparameters,
                                   output_folder_path = result_folder_path,
                                   resume_from_saved = True,
                                   output_file_name_root = CollaborativeDL_Matlab_RecommenderWrapper.RECOMMENDER_NAME)


        except Exception as e:

            print("On recommender {} Exception {}".format(CollaborativeDL_Matlab_RecommenderWrapper, str(e)))
            traceback.print_exc()




    ################################################################################################
    ######
    ######      PRINT RESULTS
    ######

    if flag_print_results:

        n_test_users = np.sum(np.ediff1d(URM_test.indptr)>=1)
        ICM_names_to_report_list = list(dataset.ICM_DICT.keys())
        dataset_name = "{}_{}".format(dataset_variant, train_interactions)
        file_name = "{}..//{}_{}_".format(result_folder_path, ALGORITHM_NAME, dataset_name)

        result_loader = ResultFolderLoader(result_folder_path,
                                         base_algorithm_list = None,
                                         other_algorithm_list = [CollaborativeDL_Matlab_RecommenderWrapper],
                                         KNN_similarity_list = KNN_similarity_to_report_list,
                                         ICM_names_list = ICM_names_to_report_list,
                                         UCM_names_list = None)


        result_loader.generate_latex_results(file_name + "{}_latex_results.txt".format("article_metrics"),
                                           metrics_list = ["RECALL"],
                                           cutoffs_list = [50, 100, 150, 200, 250, 300],
                                           table_title = None,
                                           highlight_best = True)

        result_loader.generate_latex_results(file_name + "{}_latex_results.txt".format("all_metrics"),
                                           metrics_list = ["PRECISION", "RECALL", "MAP", "MRR", "NDCG", "F1", "HIT_RATE", "ARHR",
                                                           "NOVELTY", "DIVERSITY_MEAN_INTER_LIST", "DIVERSITY_HERFINDAHL", "COVERAGE_ITEM", "DIVERSITY_GINI", "SHANNON_ENTROPY"],
                                           cutoffs_list = [150],
                                           table_title = None,
                                           highlight_best = True)

        result_loader.generate_latex_time_statistics(file_name + "{}_latex_results.txt".format("time"),
                                           n_evaluation_users=n_test_users,
                                           table_title = None)




if __name__ == '__main__':

    ALGORITHM_NAME = "CollaborativeDL"
    CONFERENCE_NAME = "KDD"



    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--baseline_tune',        help="Baseline hyperparameter search", type = bool, default = False)
    parser.add_argument('-a', '--DL_article_default',   help="Train the DL model with article hyperparameters", type = bool, default = False)
    parser.add_argument('-p', '--print_results',        help="Print results", type = bool, default = True)


    input_flags = parser.parse_args()
    print(input_flags)

    KNN_similarity_to_report_list = ["cosine", "dice", "jaccard", "asymmetric", "tversky"]


    dataset_variant_list = ["a", "t"]
    train_interactions_list = [1, 10]

    for dataset_variant in dataset_variant_list:

        for train_interactions in train_interactions_list:

            read_data_split_and_search(dataset_variant, train_interactions,
                                        flag_baselines_tune=input_flags.baseline_tune,
                                        flag_DL_article_default= input_flags.DL_article_default,
                                        flag_print_results = input_flags.print_results,
                                        )


    if input_flags.print_results:
        generate_latex_hyperparameters(result_folder_path ="result_experiments/{}/".format(CONFERENCE_NAME),
                                      algorithm_name= ALGORITHM_NAME,
                                      experiment_subfolder_list = [
                                          "citeulike_{}_{}".format(dataset_variant, train_interactions) for dataset_variant in dataset_variant_list for train_interactions in train_interactions_list
                                          ],
                                      ICM_names_to_report_list = ["ICM_tokens_TFIDF", "ICM_tokens_bool"],
                                      KNN_similarity_to_report_list = KNN_similarity_to_report_list,
                                      other_algorithm_list = [CollaborativeDL_Matlab_RecommenderWrapper],
                                      split_per_algorithm_type = True)
