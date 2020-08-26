#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/07/19

@author: Simone Boglio
"""

import numpy as np
import os, traceback, argparse
import scipy.sparse as sps
from functools import partial
from Utils.plot_popularity import plot_popularity_bias, save_popularity_statistics

from Base.Evaluation.Evaluator import EvaluatorNegativeItemSample

from Recommender_import_list import *

from ParameterTuning.run_parameter_search import runParameterSearch_Collaborative
from ParameterTuning.SearchSingleCase import SearchSingleCase
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs

from Utils.ResultFolderLoader import ResultFolderLoader, generate_latex_hyperparameters
from Utils.assertions_on_data_for_experiments import assert_disjoint_matrices, assert_implicit_data


from Conferences.IJCAI.DELF_our_interface.Movielens1MReader.Movielens1MReader import Movielens1MReader
from Conferences.IJCAI.DELF_our_interface.AmazonMusicReader.AmazonMusicReader import AmazonMusicReader

from Conferences.IJCAI.DELF_our_interface.DELFWrapper import DELF_MLP_RecommenderWrapper, DELF_EF_RecommenderWrapper




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




def get_cold_items(URM):

    cold_items_flag = np.ediff1d(sps.csc_matrix(URM).indptr) == 0

    return np.arange(0, URM.shape[1])[cold_items_flag]


def read_data_split_and_search(dataset_name,
                                   flag_baselines_tune = False,
                                   flag_DL_article_default = False, flag_DL_tune = False,
                                   flag_print_results = False):

    result_folder_path = "result_experiments/{}/{}_{}/".format(CONFERENCE_NAME, ALGORITHM_NAME, dataset_name)

    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)


    # Ensure both experiments use the same data
    dataset_folder_path = "result_experiments/{}/{}_{}/".format(CONFERENCE_NAME, ALGORITHM_NAME,
                                                                dataset_name.replace("_remove_cold_items", ""))

    if not os.path.exists(dataset_folder_path):
        os.makedirs(dataset_folder_path)

    if 'amazon_music' in dataset_name:
        dataset = AmazonMusicReader(dataset_folder_path)

    elif 'movielens1m_ours' in dataset_name:
        dataset = Movielens1MReader(dataset_folder_path, type ="ours")

    elif 'movielens1m_original' in dataset_name:
        dataset = Movielens1MReader(dataset_folder_path, type ="original")

    else:
        print("Dataset name not supported, current is {}".format(dataset_name))
        return

    print ('Current dataset is: {}'.format(dataset_name))



    URM_train = dataset.URM_DICT["URM_train"].copy()
    URM_validation = dataset.URM_DICT["URM_validation"].copy()
    URM_test = dataset.URM_DICT["URM_test"].copy()
    URM_test_negative = dataset.URM_DICT["URM_test_negative"].copy()


    # Ensure IMPLICI data and DISJOINT matrices
    assert_implicit_data([URM_train, URM_validation, URM_test, URM_test_negative])
    assert_disjoint_matrices([URM_train, URM_validation, URM_test, URM_test_negative])

    cold_items_statistics(URM_train, URM_validation, URM_test, URM_test_negative)


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

    if "_remove_cold_items" in dataset_name:
        ignore_items_validation = get_cold_items(URM_train)
        ignore_items_test = get_cold_items(URM_train + URM_validation)
    else:
        ignore_items_validation = None
        ignore_items_test = None

    evaluator_validation = EvaluatorNegativeItemSample(URM_validation, URM_test_negative, cutoff_list=cutoff_list_validation, ignore_items=ignore_items_validation)
    evaluator_test = EvaluatorNegativeItemSample(URM_test, URM_test_negative, cutoff_list=cutoff_list_test, ignore_items=ignore_items_test)

    # The Evaluator automatically skips users with no test interactions
    # in this case we need the evaluation done with and without cold items to be comparable
    # So we ensure the users that are included in the evaluation are the same in both cases.
    evaluator_validation.users_to_evaluate = np.arange(URM_train.shape[0])
    evaluator_test.users_to_evaluate = np.arange(URM_train.shape[0])

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

    
        earlystopping_hyperparameters = {'validation_every_n': 5,
                                    'stop_on_validation': True,
                                    'lower_validations_allowed': 5,
                                    'evaluator_object': evaluator_validation,
                                    'validation_metric': metric_to_optimize,
                                    }

        num_factors = 64
    
        article_hyperparameters = {'epochs': 500,
                              'learning_rate': 0.001,
                              'batch_size': 256,
                              'num_negatives': 4,
                              'layers': (num_factors*4, num_factors*2, num_factors),
                              'regularization_layers': (0, 0, 0),
                              'learner': 'adam',
                              'verbose': False,
                              }



        parameterSearch = SearchSingleCase(DELF_MLP_RecommenderWrapper,
                                           evaluator_validation=evaluator_validation,
                                           evaluator_test=evaluator_test)

        recommender_input_args = SearchInputRecommenderArgs(CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                                                            FIT_KEYWORD_ARGS = earlystopping_hyperparameters)

        recommender_input_args_last_test = recommender_input_args.copy()
        recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train + URM_validation

        parameterSearch.search(recommender_input_args,
                               recommender_input_args_last_test = recommender_input_args_last_test,
                               fit_hyperparameters_values= article_hyperparameters,
                               output_folder_path = result_folder_path,
                               resume_from_saved = True,
                               output_file_name_root = DELF_MLP_RecommenderWrapper.RECOMMENDER_NAME)





        parameterSearch = SearchSingleCase(DELF_EF_RecommenderWrapper,
                                           evaluator_validation=evaluator_validation,
                                           evaluator_test=evaluator_test)

        recommender_input_args = SearchInputRecommenderArgs(CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],
                                                            FIT_KEYWORD_ARGS=earlystopping_hyperparameters)


        recommender_input_args_last_test = recommender_input_args.copy()
        recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train + URM_validation

        parameterSearch.search(recommender_input_args,
                               recommender_input_args_last_test = recommender_input_args_last_test,
                               fit_hyperparameters_values=article_hyperparameters,
                               output_folder_path=result_folder_path,
                               resume_from_saved = True,
                               output_file_name_root=DELF_EF_RecommenderWrapper.RECOMMENDER_NAME)


    ################################################################################################
    ######
    ######      PRINT RESULTS
    ######

    if flag_print_results:

        n_test_users = np.sum(np.ediff1d(URM_test.indptr)>=1)
        file_name = "{}..//{}_{}_".format(result_folder_path, ALGORITHM_NAME, dataset_name)

        result_loader = ResultFolderLoader(result_folder_path,
                                         base_algorithm_list = None,
                                         other_algorithm_list = [DELF_MLP_RecommenderWrapper, DELF_EF_RecommenderWrapper],
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
                                           cutoffs_list = [10],
                                           table_title = None,
                                           highlight_best = True)

        result_loader.generate_latex_time_statistics(file_name + "{}_latex_results.txt".format("time"),
                                           n_evaluation_users=n_test_users,
                                           table_title = None)



if __name__ == '__main__':

    CONFERENCE_NAME = 'IJCAI'
    ALGORITHM_NAME = 'DELF'


    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--baseline_tune',        help="Baseline hyperparameter search", type = bool, default = False)
    parser.add_argument('-a', '--DL_article_default',   help="Train the DL model with article hyperparameters", type = bool, default = False)
    parser.add_argument('-p', '--print_results',        help="Print results", type = bool, default = True)

    input_flags = parser.parse_args()
    print(input_flags)
    
    
    KNN_similarity_to_report_list = ["cosine", "dice", "jaccard", "asymmetric", "tversky"]

    dataset_list = ['amazon_music', 'movielens1m_ours', 'amazon_music_remove_cold_items', 'movielens1m_ours_remove_cold_items']

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
                                     other_algorithm_list=[DELF_MLP_RecommenderWrapper, DELF_EF_RecommenderWrapper],
                                     KNN_similarity_to_report_list = KNN_similarity_to_report_list,
                                     split_per_algorithm_type = True)
