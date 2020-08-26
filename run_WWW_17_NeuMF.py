#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17

@author: Maurizio Ferrari Dacrema
"""

from Recommender_import_list import *
from Conferences.WWW.NeuMF_our_interface.NeuMF_RecommenderWrapper import NeuMF_RecommenderWrapper


from ParameterTuning.run_parameter_search import runParameterSearch_Collaborative
from ParameterTuning.SearchSingleCase import SearchSingleCase
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs

import os, traceback, argparse
from functools import partial
import numpy as np

from Utils.assertions_on_data_for_experiments import assert_implicit_data, assert_disjoint_matrices
from Utils.plot_popularity import plot_popularity_bias, save_popularity_statistics


from Utils.ResultFolderLoader import ResultFolderLoader, generate_latex_hyperparameters



def read_data_split_and_search(dataset_name,
                                   flag_baselines_tune = False,
                                   flag_DL_article_default = False, flag_DL_tune = False,
                                   flag_print_results = False):


    from Conferences.WWW.NeuMF_our_interface.Movielens1M.Movielens1MReader import Movielens1MReader
    from Conferences.WWW.NeuMF_our_interface.Pinterest.PinterestICCVReader import PinterestICCVReader


    result_folder_path = "result_experiments/{}/{}_{}/".format(CONFERENCE_NAME, ALGORITHM_NAME, dataset_name)


    if dataset_name == "movielens1m":
        dataset = Movielens1MReader(result_folder_path)

    elif dataset_name == "pinterest":
        dataset = PinterestICCVReader(result_folder_path)


    URM_train = dataset.URM_DICT["URM_train"].copy()
    URM_validation = dataset.URM_DICT["URM_validation"].copy()
    URM_test = dataset.URM_DICT["URM_test"].copy()
    URM_test_negative = dataset.URM_DICT["URM_test_negative"].copy()


    # Ensure IMPLICIT data and DISJOINT sets
    assert_implicit_data([URM_train, URM_validation, URM_test, URM_test_negative])

    assert_disjoint_matrices([URM_train, URM_validation, URM_test])
    assert_disjoint_matrices([URM_train, URM_validation, URM_test_negative])



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

    metric_to_optimize = "HIT_RATE"
    n_cases = 50
    n_random_starts = 15


    from Base.Evaluation.Evaluator import EvaluatorNegativeItemSample

    evaluator_validation = EvaluatorNegativeItemSample(URM_validation, URM_test_negative, cutoff_list=[10])
    evaluator_test = EvaluatorNegativeItemSample(URM_test, URM_test_negative, cutoff_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


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

        try:


            if dataset_name == "movielens1m":
                num_factors = 64
            elif dataset_name == "pinterest":
                num_factors = 16


            neuMF_article_hyperparameters = {
                "epochs": 100,
                "epochs_gmf": 100,
                "epochs_mlp": 100,
                "batch_size": 256,
                "num_factors": num_factors,
                "layers": [num_factors*4, num_factors*2, num_factors],
                "reg_mf": 0.0,
                "reg_layers": [0,0,0],
                "num_negatives": 4,
                "learning_rate": 1e-3,
                "learning_rate_pretrain": 1e-3,
                "learner": "sgd",
                "learner_pretrain": "adam",
                "pretrain": True
            }


            neuMF_earlystopping_hyperparameters = {
                "validation_every_n": 5,
                "stop_on_validation": True,
                "evaluator_object": evaluator_validation,
                "lower_validations_allowed": 5,
                "validation_metric": metric_to_optimize
            }



            parameterSearch = SearchSingleCase(NeuMF_RecommenderWrapper,
                                               evaluator_validation=evaluator_validation,
                                               evaluator_test=evaluator_test)

            recommender_input_args = SearchInputRecommenderArgs(
                                                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                                                FIT_KEYWORD_ARGS = neuMF_earlystopping_hyperparameters)

            recommender_input_args_last_test = recommender_input_args.copy()
            recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train + URM_validation

            parameterSearch.search(recommender_input_args,
                                   recommender_input_args_last_test = recommender_input_args_last_test,
                                   fit_hyperparameters_values=neuMF_article_hyperparameters,
                                   output_folder_path = result_folder_path,
                                   resume_from_saved = True,
                                   output_file_name_root = NeuMF_RecommenderWrapper.RECOMMENDER_NAME)



        except Exception as e:

            print("On recommender {} Exception {}".format(NeuMF_RecommenderWrapper, str(e)))
            traceback.print_exc()








    ################################################################################################
    ######
    ######      PRINT RESULTS
    ######

    if flag_print_results:

        n_test_users = np.sum(np.ediff1d(URM_test.indptr)>=1)
        file_name = "{}..//{}_{}_".format(result_folder_path, ALGORITHM_NAME, dataset_name)

        result_loader = ResultFolderLoader(result_folder_path,
                                         base_algorithm_list = None,
                                         other_algorithm_list = [NeuMF_RecommenderWrapper],
                                         KNN_similarity_list = KNN_similarity_to_report_list,
                                         ICM_names_list = None,
                                         UCM_names_list = None)


        result_loader.generate_latex_results(file_name + "{}_latex_results.txt".format("article_metrics"),
                                           metrics_list = ["HIT_RATE", "NDCG"],
                                           cutoffs_list = [1, 5, 10],
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

    ALGORITHM_NAME = "NeuMF"
    CONFERENCE_NAME = "WWW"


    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--baseline_tune',        help="Baseline hyperparameter search", type = bool, default = False)
    parser.add_argument('-a', '--DL_article_default',   help="Train the DL model with article hyperparameters", type = bool, default = False)
    parser.add_argument('-p', '--print_results',        help="Print results", type = bool, default = True)

    input_flags = parser.parse_args()
    print(input_flags)

    KNN_similarity_to_report_list = ["cosine", "dice", "jaccard", "asymmetric", "tversky"]



    dataset_list = ["movielens1m", "pinterest"]


    for dataset_name in dataset_list:

        read_data_split_and_search(dataset_name,
                                        flag_baselines_tune=input_flags.baseline_tune,
                                        flag_DL_article_default= input_flags.DL_article_default,
                                        flag_print_results = input_flags.print_results,
                                        )



    if input_flags.print_results:
        generate_latex_hyperparameters(result_folder_path ="result_experiments/{}/".format(CONFERENCE_NAME),
                                       algorithm_name= ALGORITHM_NAME,
                                       experiment_subfolder_list = dataset_list,
                                       other_algorithm_list = [NeuMF_RecommenderWrapper],
                                       KNN_similarity_to_report_list = KNN_similarity_to_report_list,
                                       split_per_algorithm_type = True)
