#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/02/19

@author: Simone Boglio
"""


from Recommender_import_list import *



from ParameterTuning.run_parameter_search import runParameterSearch_Collaborative
from ParameterTuning.SearchSingleCase import SearchSingleCase
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs


from Utils.ResultFolderLoader import ResultFolderLoader, generate_latex_hyperparameters
from Utils.print_negative_items_stats import print_negative_items_stats

from functools import partial
import numpy as np
import os, traceback, argparse

from Conferences.IJCAI.ConvNCF_our_interface.GowallaReader.GowallaReader import GowallaReader
from Conferences.IJCAI.ConvNCF_our_interface.YelpReader.YelpReader import YelpReader

from Conferences.IJCAI.ConvNCF_our_interface.ConvNCF_wrapper import ConvNCF_RecommenderWrapper
import Conferences.IJCAI.ConvNCF_our_interface.ConvNCF as ConvNCF






def read_data_split_and_search(dataset_name,
                                   flag_baselines_tune = False,
                                   flag_DL_article_default = False, flag_DL_tune = False,
                                   flag_print_results = False):
    

    result_folder_path = "result_experiments/{}/{}_{}/".format(CONFERENCE_NAME, ALGORITHM_NAME, dataset_name)

    if dataset_name == "gowalla":
        dataset = GowallaReader(result_folder_path)

    elif dataset_name == "yelp":
        dataset = YelpReader(result_folder_path)

    else:
        print("Dataset name not supported, current is {}".format(dataset_name))
        return

    print ('Current dataset is: {}'.format(dataset_name))


    URM_train = dataset.URM_DICT["URM_train"].copy()
    URM_validation = dataset.URM_DICT["URM_validation"].copy()
    URM_test = dataset.URM_DICT["URM_test"].copy()
    URM_test_negative = dataset.URM_DICT["URM_test_negative"].copy()

    print_negative_items_stats(URM_train, URM_validation, URM_test, URM_test_negative)

    # Ensure IMPLICIT data
    from Utils.assertions_on_data_for_experiments import assert_implicit_data, assert_disjoint_matrices

    assert_implicit_data([URM_train, URM_validation, URM_test, URM_test_negative])

    # URM_test_negative contains duplicates in both train and test
    assert_disjoint_matrices([URM_train, URM_validation, URM_test])


    # If directory does not exist, create
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)



    collaborative_algorithm_list = [
        Random,
        TopPop,
        UserKNNCFRecommender,
        ItemKNNCFRecommender,
        P3alphaRecommender,
        RP3betaRecommender,
        PureSVDRecommender,
        NMFRecommender,
        IALSRecommender,
        MatrixFactorization_BPR_Cython,
        MatrixFactorization_FunkSVD_Cython,
        EASE_R_Recommender,
        SLIM_BPR_Cython,
        SLIMElasticNetRecommender,
        ]

    metric_to_optimize = "NDCG"
    n_cases = 50
    n_random_starts = 15


    from Base.Evaluation.Evaluator import EvaluatorNegativeItemSample

    cutoff_list_validation = [10]
    cutoff_list_test = [5, 10, 20]

    evaluator_validation = EvaluatorNegativeItemSample(URM_validation, URM_test_negative, cutoff_list=cutoff_list_validation)
    evaluator_test = EvaluatorNegativeItemSample(URM_test, URM_test_negative, cutoff_list=cutoff_list_test)

    runParameterSearch_Collaborative_partial = partial(runParameterSearch_Collaborative,
                                                       URM_train = URM_train,
                                                       URM_train_last_test = URM_train + URM_validation,
                                                       metric_to_optimize = metric_to_optimize,
                                                       evaluator_validation_earlystopping = evaluator_validation,
                                                       evaluator_validation = evaluator_validation,
                                                       evaluator_test = evaluator_test,
                                                       output_folder_path = result_folder_path,
                                                       parallelizeKNN = False,
                                                       allow_weighting = True,
                                                       resume_from_saved = True,
                                                       n_cases = n_cases,
                                                       n_random_starts = n_random_starts)



    if flag_baselines_tune:
        
        for recommender_class in collaborative_algorithm_list:
            try:
                runParameterSearch_Collaborative_partial(recommender_class)
            except Exception as e:
                print("On recommender {} Exception {}".format(recommender_class, str(e)))
                traceback.print_exc()


    ################################################################################################
    ######
    ######      DL ALGORITHM
    ######

    if flag_DL_article_default:

        # Providing an empty matrix to URM_negative for the train samples
        article_hyperparameters = {

                              "batch_size": 512,
                              "epochs": 1500,
                              "epochs_MFBPR": 500,
                              "embedding_size":64,
                              "hidden_size":128,
                              "negative_sample_per_positive":1,
                              "negative_instances_per_positive":4,
                              "regularization_users_items":0.01,
                              "regularization_weights":10,
                              "regularization_filter_weights":1,
                              "learning_rate_embeddings":0.05,
                              "learning_rate_CNN":0.05,
                              "channel_size":[32, 32, 32, 32, 32, 32],
                              "dropout":0.0,
                              "epoch_verbose":1,
                              }

        earlystopping_hyperparameters = {"validation_every_n": 5,
                                    "stop_on_validation": True,
                                    "lower_validations_allowed": 5,
                                    "evaluator_object": evaluator_validation,
                                    "validation_metric": metric_to_optimize,
                                    "epochs_min": 150
                                    }






        parameterSearch = SearchSingleCase(ConvNCF_RecommenderWrapper,
                                           evaluator_validation=evaluator_validation,
                                           evaluator_test=evaluator_test)

        recommender_input_args = SearchInputRecommenderArgs(
                                            CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                                            FIT_KEYWORD_ARGS = earlystopping_hyperparameters)

        recommender_input_args_last_test = recommender_input_args.copy()
        recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train + URM_validation

        parameterSearch.search(recommender_input_args,
                               recommender_input_args_last_test = recommender_input_args_last_test,
                               fit_hyperparameters_values=article_hyperparameters,
                               output_folder_path = result_folder_path,
                               resume_from_saved = True,
                               output_file_name_root = ConvNCF_RecommenderWrapper.RECOMMENDER_NAME)


        #remember to close the global session since use global variables
        ConvNCF.close_session(verbose = True)




    ################################################################################################
    ######
    ######      PRINT RESULTS
    ######

    if flag_print_results:

        n_test_users = np.sum(np.ediff1d(URM_test.indptr)>=1)
        file_name = "{}..//{}_{}_".format(result_folder_path, ALGORITHM_NAME, dataset_name)

        result_loader = ResultFolderLoader(result_folder_path,
                                         base_algorithm_list = None,
                                         other_algorithm_list = [ConvNCF_RecommenderWrapper],
                                         KNN_similarity_list = KNN_similarity_to_report_list,
                                         ICM_names_list = None,
                                         UCM_names_list = None)


        result_loader.generate_latex_results(file_name + "{}_latex_results.txt".format("article_metrics"),
                                           metrics_list = ["HIT_RATE", "NDCG"],
                                           cutoffs_list = cutoff_list_test,
                                           table_title = None,
                                           highlight_best = True)

        result_loader.generate_latex_results(file_name + "{}_latex_results.txt".format("all_metrics"),
                                           metrics_list = ["PRECISION", "RECALL", "MAP", "MRR", "NDCG", "F1", "HIT_RATE", "ARHR",
                                                           "NOVELTY", "DIVERSITY_MEAN_INTER_LIST", "DIVERSITY_HERFINDAHL", "COVERAGE_ITEM", "DIVERSITY_GINI", "SHANNON_ENTROPY"],
                                           cutoffs_list = cutoff_list_validation,
                                           table_title = None,
                                           highlight_best = True)

        result_loader.generate_latex_time_statistics(file_name + "{}_latex_results.txt".format("time"),
                                           n_evaluation_users=n_test_users,
                                           table_title = None)



if __name__ == '__main__':


    ALGORITHM_NAME = "ConvNCF"
    CONFERENCE_NAME = "IJCAI"


    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--baseline_tune',        help="Baseline hyperparameter search", type = bool, default = False)
    parser.add_argument('-a', '--DL_article_default',   help="Train the DL model with article hyperparameters", type = bool, default = False)
    parser.add_argument('-p', '--print_results',        help="Print results", type = bool, default = True)


    input_flags = parser.parse_args()
    print(input_flags)

    KNN_similarity_to_report_list = ["cosine", "dice", "jaccard", "asymmetric", "tversky"]


    dataset_list = ["yelp", "gowalla"]

    for dataset_name in dataset_list:
        read_data_split_and_search(dataset_name,
                                        flag_baselines_tune=input_flags.baseline_tune,
                                        flag_DL_article_default= input_flags.DL_article_default,
                                        flag_print_results = input_flags.print_results,
                                        )


    if input_flags.print_results:
        generate_latex_hyperparameters(result_folder_path="result_experiments/{}/".format(CONFERENCE_NAME),
                                      algorithm_name=ALGORITHM_NAME,
                                      experiment_subfolder_list=dataset_list,
                                      other_algorithm_list=[ConvNCF_RecommenderWrapper],
                                      KNN_similarity_to_report_list = KNN_similarity_to_report_list,
                                      split_per_algorithm_type = True)
