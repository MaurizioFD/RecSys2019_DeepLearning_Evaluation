#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 28/04/19

@author: Simone Boglio
"""

import numpy as np
import os, traceback, argparse
from functools import partial


from Base.Evaluation.Evaluator import EvaluatorNegativeItemSample

from Recommender_import_list import *
# from Conferences.IJCAI.DMF_our_interface.DMFWrapper import DMF_BCE_RecommenderWrapper, DMF_NCE_RecommenderWrapper



from ParameterTuning.run_parameter_search import runParameterSearch_Collaborative
from ParameterTuning.SearchSingleCase import SearchSingleCase
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs


from Utils.ResultFolderLoader import ResultFolderLoader, generate_latex_hyperparameters
from Utils.assertions_on_data_for_experiments import assert_implicit_data, assert_disjoint_matrices
from Utils.plot_popularity import plot_popularity_bias, save_popularity_statistics


from Conferences.IJCAI.DMF_our_interface.Movielens1MReader.Movielens1MReader import Movielens1MReader
from Conferences.IJCAI.DMF_our_interface.Movielens100KReader.Movielens100KReader import Movielens100KReader
from Conferences.IJCAI.DMF_our_interface.AmazonMusicReader.AmazonMusicReader import AmazonMusicReader
from Conferences.IJCAI.DMF_our_interface.AmazonMovieReader.AmazonMovieReader import AmazonMovieReader




def cold_items_statistics(URM_train, URM_validation, URM_test, URM_test_negative):

    # Cold items experiment
    import  scipy.sparse as sps

    URM_train_validation = URM_train + URM_validation
    n_users, n_items = URM_train_validation.shape

    item_in_train_flag = np.ediff1d(sps.csc_matrix(URM_train_validation).indptr) > 0
    item_in_test_flag = np.ediff1d(sps.csc_matrix(URM_test).indptr) > 0

    test_item_not_in_train_flag = np.logical_and(item_in_test_flag, np.logical_not(item_in_train_flag))
    test_item_in_train_flag = np.logical_and(item_in_test_flag, item_in_train_flag)

    print("The test data contains {} unique items, {} ({:.2f} %) of them never appear in train data".format(
        item_in_test_flag.sum(),
        test_item_not_in_train_flag.sum(),
        test_item_not_in_train_flag.sum()/item_in_test_flag.sum()*100,
    ))



def read_data_split_and_search(dataset_name,
                                   flag_baselines_tune = False,
                                   flag_DL_article_default = False, flag_DL_tune = False,
                                   flag_print_results = False):


    result_folder_path = "result_experiments/{}/{}_{}/".format(CONFERENCE_NAME, ALGORITHM_NAME, dataset_name)


    if dataset_name == 'amazon_music_original':
        dataset = AmazonMusicReader(result_folder_path, original = True)

    elif dataset_name == 'amazon_music_ours':
        dataset = AmazonMusicReader(result_folder_path, original = False)

    elif dataset_name == 'amazon_movie':
        dataset = AmazonMovieReader(result_folder_path)

    elif dataset_name == 'movielens100k':
        dataset = Movielens100KReader(result_folder_path)

    elif dataset_name == 'movielens1m':
        dataset = Movielens1MReader(result_folder_path)

    else:
        print("Dataset name not supported, current is {}".format(dataset_name))
        return


    print ('Current dataset is: {}'.format(dataset_name))



    URM_train = dataset.URM_DICT["URM_train"].copy()
    URM_validation = dataset.URM_DICT["URM_validation"].copy()
    URM_test = dataset.URM_DICT["URM_test"].copy()
    URM_test_negative = dataset.URM_DICT["URM_test_negative"].copy()


    # Ensure DISJOINT sets. Do not ensure IMPLICIT data because the algorithm needs explicit data
    assert_disjoint_matrices([URM_train, URM_validation, URM_test, URM_test_negative])

    cold_items_statistics(URM_train, URM_validation, URM_test, URM_test_negative)

    # If directory does not exist, create
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)

    algorithm_dataset_string = "{}_{}_".format(ALGORITHM_NAME, dataset_name)

    plot_popularity_bias([URM_train + URM_validation, URM_test],
                         ["Training data", "Test data"],
                         result_folder_path + algorithm_dataset_string + "popularity_plot")

    save_popularity_statistics([URM_train + URM_validation + URM_test, URM_train + URM_validation, URM_test],
                               ["Full data", "Training data", "Test data"],
                               result_folder_path + algorithm_dataset_string + "popularity_statistics")


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

    """
    NOTICE: We did not upload the source code of DMF as it was not publicly available and the original
            authors did not respond to our request to add it to this repository
    """

    if flag_DL_article_default:

        if dataset_name in ['amazon_music_original', 'amazon_music_ours']:
            last_layer_size = 128
        else:
            last_layer_size = 64

        article_hyperparameters = {'epochs': 300,
                              'learning_rate': 0.0001,
                              'batch_size': 256,
                              'num_negatives': 7,   # As reported in the "Detailed implementation" section of the original paper
                              'last_layer_size': last_layer_size,
                              }

        earlystopping_hyperparameters = {'validation_every_n': 5,
                                    'stop_on_validation': True,
                                    'lower_validations_allowed': 5,
                                    'evaluator_object': evaluator_validation,
                                    'validation_metric': metric_to_optimize,
                                    }

        #
        # try:
        #
        #
        #     parameterSearch = SearchSingleCase(DMF_NCE_RecommenderWrapper,
        #                                        evaluator_validation=evaluator_validation,
        #                                        evaluator_test=evaluator_test)
        #
        #     recommender_input_args = SearchInputRecommenderArgs(
        #                                         CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
        #                                         FIT_KEYWORD_ARGS = earlystopping_hyperparameters)
        #
        #     recommender_input_args_last_test = recommender_input_args.copy()
        #     recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train + URM_validation
        #
        #     parameterSearch.search(recommender_input_args,
        #                            recommender_input_args_last_test = recommender_input_args_last_test,
        #                            fit_hyperparameters_values = article_hyperparameters,
        #                            output_folder_path = result_folder_path,
        #                            resume_from_saved = True,
        #                            output_file_name_root = DMF_NCE_RecommenderWrapper.RECOMMENDER_NAME)
        #
        #
        #
        # except Exception as e:
        #
        #     print("On recommender {} Exception {}".format(DMF_NCE_RecommenderWrapper, str(e)))
        #     traceback.print_exc()
        #
        #
        #
        # try:
        #
        #
        #     parameterSearch = SearchSingleCase(DMF_BCE_RecommenderWrapper,
        #                                        evaluator_validation=evaluator_validation,
        #                                        evaluator_test=evaluator_test)
        #
        #     recommender_input_args = SearchInputRecommenderArgs(
        #                                         CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
        #                                         FIT_KEYWORD_ARGS = earlystopping_hyperparameters)
        #
        #     recommender_input_args_last_test = recommender_input_args.copy()
        #     recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train + URM_validation
        #
        #     parameterSearch.search(recommender_input_args,
        #                            recommender_input_args_last_test = recommender_input_args_last_test,
        #                            fit_hyperparameters_values = article_hyperparameters,
        #                            output_folder_path = result_folder_path,
        #                            resume_from_saved = True,
        #                            output_file_name_root = DMF_BCE_RecommenderWrapper.RECOMMENDER_NAME)
        #
        #
        # except Exception as e:
        #
        #     print("On recommender {} Exception {}".format(DMF_BCE_RecommenderWrapper, str(e)))
        #     traceback.print_exc()



    ################################################################################################
    ######
    ######      PRINT RESULTS
    ######

    if flag_print_results:

        n_test_users = np.sum(np.ediff1d(URM_test.indptr)>=1)
        file_name = "{}..//{}_{}_".format(result_folder_path, ALGORITHM_NAME, dataset_name)

        result_loader = ResultFolderLoader(result_folder_path,
                                         base_algorithm_list = None,
                                         other_algorithm_list = [DMF_NCE_RecommenderWrapper, DMF_BCE_RecommenderWrapper],
                                         KNN_similarity_list = KNN_similarity_to_report_list,
                                         ICM_names_list = None,
                                         UCM_names_list = None)


        result_loader.generate_latex_results(file_name + "{}_latex_results.txt".format("article_metrics"),
                                           metrics_list = ["HIT_RATE", "NDCG"],
                                           cutoffs_list = cutoff_list_validation,
                                           table_title = None,
                                           highlight_best = True)

        result_loader.generate_latex_results(file_name + "{}_latex_results.txt".format("all_metrics"),
                                           metrics_list = ["PRECISION", "RECALL", "MAP", "MRR", "NDCG", "F1", "HIT_RATE", "ARHR",
                                                           "NOVELTY", "DIVERSITY_MEAN_INTER_LIST", "DIVERSITY_HERFINDAHL", "COVERAGE_ITEM", "DIVERSITY_GINI", "SHANNON_ENTROPY"],
                                           cutoffs_list = [10],
                                           table_title = None,
                                           highlight_best = True)

        result_loader.generate_latex_time_statistics(file_name + "{}_latex_results.txt".format("time"),
                                           n_evaluation_users=n_test_users,
                                           table_title = None)





if __name__ == '__main__':

    ALGORITHM_NAME = "DMF"
    CONFERENCE_NAME = "IJCAI"


    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--baseline_tune',        help="Baseline hyperparameter search", type = bool, default = False)
    parser.add_argument('-a', '--DL_article_default',   help="Train the DL model with article hyperparameters", type = bool, default = False)
    parser.add_argument('-p', '--print_results',        help="Print results", type = bool, default = True)


    input_flags = parser.parse_args()
    print(input_flags)

    KNN_similarity_to_report_list = ["cosine", "dice", "jaccard", "asymmetric", "tversky"]


    dataset_list = ['amazon_music_original', 'amazon_music_ours', 'movielens100k', 'amazon_movie', 'movielens1m']

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
                                      other_algorithm_list= [DMF_NCE_RecommenderWrapper, DMF_BCE_RecommenderWrapper],
                                      KNN_similarity_to_report_list = KNN_similarity_to_report_list,
                                      split_per_algorithm_type = True)
