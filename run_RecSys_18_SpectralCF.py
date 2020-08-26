#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17

@author: Maurizio Ferrari Dacrema
@author: Simone Boglio

"""

from Recommender_import_list import *
from Conferences.RecSys.SpectralCF_our_interface.SpectralCF_RecommenderWrapper import SpectralCF_RecommenderWrapper


from ParameterTuning.run_parameter_search import runParameterSearch_Collaborative
from ParameterTuning.SearchSingleCase import SearchSingleCase
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs



from Utils.ResultFolderLoader import ResultFolderLoader, generate_latex_hyperparameters
from Utils.assertions_on_data_for_experiments import assert_implicit_data, assert_disjoint_matrices
from Utils.plot_popularity import plot_popularity_bias, save_popularity_statistics

from functools import partial
import numpy as np
import os, traceback, argparse

from Conferences.RecSys.SpectralCF_our_interface.Movielens1M.Movielens1MReader import Movielens1MReader
from Conferences.RecSys.SpectralCF_our_interface.MovielensHetrec2011.MovielensHetrec2011Reader import MovielensHetrec2011Reader
from Conferences.RecSys.SpectralCF_our_interface.AmazonInstantVideo.AmazonInstantVideoReader import AmazonInstantVideoReader



######################################################################
from skopt.space import Real, Integer, Categorical

from ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from ParameterTuning.SearchSingleCase import SearchSingleCase
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs


def runParameterSearch_SpectralCF(recommender_class, URM_train, earlystopping_hyperparameters, output_file_name_root, URM_train_last_test = None,
                                  n_cases = 35, n_random_starts = 5,
                                  evaluator_validation= None, evaluator_test=None, metric_to_optimize = "RECALL",
                                  output_folder_path ="result_experiments/"):



    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)


    parameterSearch = SearchBayesianSkopt(recommender_class, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)


    ##########################################################################################################

    if recommender_class is SpectralCF_RecommenderWrapper:

        hyperparameters_range_dictionary = {}
        hyperparameters_range_dictionary["batch_size"] = Categorical([128, 256, 512, 1024, 2048])
        hyperparameters_range_dictionary["embedding_size"] =  Categorical([4, 8, 16, 32])
        hyperparameters_range_dictionary["decay"] = Real(low = 1e-5, high = 1e-1, prior = 'log-uniform')
        hyperparameters_range_dictionary["learning_rate"] = Real(low = 1e-5, high = 1e-2, prior = 'log-uniform')
        hyperparameters_range_dictionary["k"] = Integer(low = 1, high = 6)

        recommender_input_args = SearchInputRecommenderArgs(
            CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
            CONSTRUCTOR_KEYWORD_ARGS = {},
            FIT_POSITIONAL_ARGS = [],
            FIT_KEYWORD_ARGS = earlystopping_hyperparameters
        )


    #########################################################################################################

    if URM_train_last_test is not None:
        recommender_input_args_last_test = recommender_input_args.copy()
        recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train_last_test
    else:
        recommender_input_args_last_test = None

    parameterSearch.search(recommender_input_args,
                           parameter_search_space = hyperparameters_range_dictionary,
                           n_cases = n_cases,
                           n_random_starts = n_random_starts,
                           resume_from_saved = True,
                           output_folder_path = output_folder_path,
                           output_file_name_root = output_file_name_root,
                           metric_to_optimize = metric_to_optimize,
                           recommender_input_args_last_test = recommender_input_args_last_test)





def read_data_split_and_search(dataset_name, cold_start = False, cold_items=None,
                                          flag_baselines_tune = False,
                                          flag_DL_article_default = False, flag_DL_tune = False,
                                          flag_print_results = False):


    if not cold_start:
        result_folder_path = "result_experiments/{}/{}_{}/".format(CONFERENCE_NAME, ALGORITHM_NAME, dataset_name)
    else:
        result_folder_path = "result_experiments/{}/{}_cold_{}_{}/".format(CONFERENCE_NAME, ALGORITHM_NAME, cold_items, dataset_name)


    if dataset_name == "movielens1m_original":
        assert(cold_start is not True)
        dataset = Movielens1MReader(result_folder_path, type ="original")

    elif dataset_name == "movielens1m_ours":
        dataset = Movielens1MReader(result_folder_path, type ="ours", cold_start=cold_start, cold_items=cold_items)

    elif dataset_name == "hetrec":
        assert (cold_start is not True)
        dataset = MovielensHetrec2011Reader(result_folder_path)

    elif dataset_name == "amazon_instant_video":
        assert (cold_start is not True)
        dataset = AmazonInstantVideoReader(result_folder_path)


    URM_train = dataset.URM_DICT["URM_train"].copy()
    URM_validation = dataset.URM_DICT["URM_validation"].copy()
    URM_test = dataset.URM_DICT["URM_test"].copy()

    # Ensure IMPLICIT data and DISJOINT sets
    assert_implicit_data([URM_train, URM_validation, URM_test])
    assert_disjoint_matrices([URM_train, URM_validation, URM_test])


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


    metric_to_optimize = "RECALL"
    n_cases = 50
    n_random_starts = 15


    from Base.Evaluation.Evaluator import EvaluatorHoldout

    if not cold_start:
        cutoff_list_validation = [50]
        cutoff_list_test = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    else:
        cutoff_list_validation = [20]
        cutoff_list_test = [20]

    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=cutoff_list_validation)
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list_test)

    ################################################################################################
    ###### KNN CF

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

            spectralCF_article_hyperparameters = {
                "epochs": 1000,
                "batch_size": 1024,
                "embedding_size": 16,
                "decay": 0.001,
                "k": 3,
                "learning_rate": 1e-3,
            }

            spectralCF_earlystopping_hyperparameters = {
                "validation_every_n": 5,
                "stop_on_validation": True,
                "lower_validations_allowed": 5,
                "evaluator_object": evaluator_validation,
                "validation_metric": metric_to_optimize,
                "epochs_min": 400,
            }

            parameterSearch = SearchSingleCase(SpectralCF_RecommenderWrapper,
                                               evaluator_validation=evaluator_validation,
                                               evaluator_test=evaluator_test)

            recommender_input_args = SearchInputRecommenderArgs(
                                                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                                                FIT_KEYWORD_ARGS = spectralCF_earlystopping_hyperparameters)

            recommender_input_args_last_test = recommender_input_args.copy()
            recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train + URM_validation

            parameterSearch.search(recommender_input_args,
                                   recommender_input_args_last_test = recommender_input_args_last_test,
                                   fit_hyperparameters_values= spectralCF_article_hyperparameters,
                                   output_folder_path = result_folder_path,
                                   resume_from_saved = True,
                                   output_file_name_root = SpectralCF_RecommenderWrapper.RECOMMENDER_NAME + "_article_default")


        except Exception as e:

            print("On recommender {} Exception {}".format(SpectralCF_RecommenderWrapper, str(e)))
            traceback.print_exc()


    if flag_DL_tune:

        try:

            spectralCF_earlystopping_hyperparameters = {
                "validation_every_n": 5,
                "stop_on_validation": True,
                "lower_validations_allowed": 5,
                "evaluator_object": evaluator_validation,
                "validation_metric": metric_to_optimize,
                "epochs_min": 400,
                "epochs": 2000
            }

            runParameterSearch_SpectralCF(SpectralCF_RecommenderWrapper,
                                             URM_train = URM_train,
                                             URM_train_last_test = URM_train + URM_validation,
                                             earlystopping_hyperparameters = spectralCF_earlystopping_hyperparameters,
                                             metric_to_optimize = metric_to_optimize,
                                             evaluator_validation = evaluator_validation,
                                             evaluator_test = evaluator_test,
                                             output_folder_path = result_folder_path,
                                             n_cases = n_cases,
                                             n_random_starts = n_random_starts,
                                             output_file_name_root = SpectralCF_RecommenderWrapper.RECOMMENDER_NAME)


        except Exception as e:

            print("On recommender {} Exception {}".format(SpectralCF_RecommenderWrapper, str(e)))
            traceback.print_exc()



    ################################################################################################
    ######
    ######      PRINT RESULTS
    ######

    if flag_print_results:

        n_test_users = np.sum(np.ediff1d(URM_test.indptr)>=1)

        file_name = "{}..//{}_{}_".format(result_folder_path,
                                          ALGORITHM_NAME if not cold_start else "{}_cold_{}".format(ALGORITHM_NAME, cold_items),
                                          dataset_name)

        if cold_start:
            cutoffs_to_report_list = [20]
        else:
            cutoffs_to_report_list = [20, 40, 60, 80, 100]

        result_loader = ResultFolderLoader(result_folder_path,
                                         base_algorithm_list = None,
                                         other_algorithm_list = other_algorithm_list,
                                         KNN_similarity_list = KNN_similarity_to_report_list,
                                         ICM_names_list = None,
                                         UCM_names_list = None)


        result_loader.generate_latex_results(file_name + "{}_latex_results.txt".format("article_metrics"),
                                           metrics_list = ["RECALL", "MAP"],
                                           cutoffs_list = cutoffs_to_report_list,
                                           table_title = None,
                                           highlight_best = True)

        result_loader.generate_latex_results(file_name + "{}_latex_results.txt".format("beyond_accuracy_metrics"),
                                           metrics_list = ["DIVERSITY_MEAN_INTER_LIST", "DIVERSITY_HERFINDAHL", "COVERAGE_ITEM", "DIVERSITY_GINI", "SHANNON_ENTROPY"],
                                           cutoffs_list = [50],
                                           table_title = None,
                                           highlight_best = True)

        result_loader.generate_latex_results(file_name + "{}_latex_results.txt".format("all_metrics"),
                                           metrics_list = ["PRECISION", "RECALL", "MAP", "MRR", "NDCG", "F1", "HIT_RATE", "ARHR",
                                                           "NOVELTY", "DIVERSITY_MEAN_INTER_LIST", "DIVERSITY_HERFINDAHL", "COVERAGE_ITEM", "DIVERSITY_GINI", "SHANNON_ENTROPY"],
                                           cutoffs_list = [50],
                                           table_title = None,
                                           highlight_best = True)

        result_loader.generate_latex_time_statistics(file_name + "{}_latex_results.txt".format("time"),
                                           n_evaluation_users=n_test_users,
                                           table_title = None)


if __name__ == '__main__':

    ALGORITHM_NAME = "SpectralCF"
    CONFERENCE_NAME = "RecSys"

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--baseline_tune',        help="Baseline hyperparameter search", type = bool, default = False)
    parser.add_argument('-a', '--DL_article_default',   help="Train the DL model with article hyperparameters", type = bool, default = False)
    parser.add_argument('-p', '--print_results',        help="Print results", type = bool, default = True)


    parser.add_argument('-t', '--DL_tune',              help="DL model hyperparameter search", type = bool, default = False)
    parser.add_argument('-c', '--cold_start',           help="DL model cold start experiment", type = bool, default = False)

    input_flags = parser.parse_args()
    print(input_flags)



    dataset_list = ["movielens1m_ours", "movielens1m_original", "hetrec", "amazon_instant_video"]
    dataset_cold_start_list = ["movielens1m_ours"]
    cold_start_items_list = [1, 2, 3, 4, 5]

    KNN_similarity_to_report_list = ["cosine", "dice", "jaccard", "asymmetric", "tversky"]



    from collections import namedtuple

    CustomRecommenderName = namedtuple('CustomRecommenderName', ['RECOMMENDER_NAME'])

    other_algorithm_list_names = [SpectralCF_RecommenderWrapper.RECOMMENDER_NAME + hyperparameter_set for hyperparameter_set in ["",  "_article_default"]]
    other_algorithm_list = [CustomRecommenderName(RECOMMENDER_NAME = recommender_name) for recommender_name in other_algorithm_list_names]



    if input_flags.cold_start:
        for dataset_name in dataset_cold_start_list:
           for cold_start_items in cold_start_items_list:
               read_data_split_and_search(dataset_name,
                                                     cold_start = input_flags.cold_start,
                                                     cold_items=cold_start_items,
                                                     flag_baselines_tune=input_flags.baseline_tune,
                                                     flag_DL_article_default = input_flags.DL_article_default,
                                                     flag_DL_tune = input_flags.DL_tune,
                                                     flag_print_results= input_flags.print_results
                                                     )


    else:
        for dataset_name in dataset_list:
            read_data_split_and_search(dataset_name,
                                                 cold_start = input_flags.cold_start,
                                                 flag_baselines_tune=input_flags.baseline_tune,
                                                 flag_DL_article_default = input_flags.DL_article_default,
                                                 flag_DL_tune = input_flags.DL_tune,
                                                 flag_print_results= input_flags.print_results
                                                  )


    # mantain compatibility with latex parameteres function
    if input_flags.cold_start and input_flags.print_results:
        for n_cold_item in cold_start_items_list:
            generate_latex_hyperparameters(result_folder_path ="result_experiments/{}/".format(CONFERENCE_NAME),
                                              algorithm_name="{}_cold_{}".format(ALGORITHM_NAME, n_cold_item),
                                              experiment_subfolder_list = dataset_cold_start_list,
                                              other_algorithm_list = other_algorithm_list,
                                              KNN_similarity_to_report_list = KNN_similarity_to_report_list,
                                              split_per_algorithm_type = True)

    elif not input_flags.cold_start and input_flags.print_results:
        generate_latex_hyperparameters(result_folder_path ="result_experiments/{}/".format(CONFERENCE_NAME),
                                          algorithm_name= ALGORITHM_NAME,
                                          experiment_subfolder_list = dataset_list,
                                          other_algorithm_list = other_algorithm_list,
                                          KNN_similarity_to_report_list = KNN_similarity_to_report_list,
                                          split_per_algorithm_type = True)
