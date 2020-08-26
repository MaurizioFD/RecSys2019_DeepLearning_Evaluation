#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 20/02/19

@author: Simone Boglio
"""

from Recommender_import_list import *


from ParameterTuning.run_parameter_search import runParameterSearch_Collaborative
from ParameterTuning.SearchSingleCase import SearchSingleCase
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs

from Conferences.IJCAI.NeuRec_our_interface.UNeuRecWrapper import UNeuRec_RecommenderWrapper
from Conferences.IJCAI.NeuRec_our_interface.INeuRecWrapper import INeuRec_RecommenderWrapper


from Utils.ResultFolderLoader import ResultFolderLoader, generate_latex_hyperparameters


from functools import partial
import numpy as np
import os, traceback, argparse

from Conferences.IJCAI.NeuRec_our_interface.Movielens1M.Movielens1MReader import Movielens1MReader
from Conferences.IJCAI.NeuRec_our_interface.FilmTrust.FilmTrustReader import FilmTrustReader
from Conferences.IJCAI.NeuRec_our_interface.MovielensHetrec2011.MovielensHetrec2011Reader import MovielensHetrec2011Reader
from Conferences.IJCAI.NeuRec_our_interface.Frappe.FrappeReader import FrappeReader



######################################################################
from skopt.space import Real, Integer, Categorical
import traceback



from ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from ParameterTuning.SearchSingleCase import SearchSingleCase
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs


def runParameterSearch_NeuRec(recommender_class, URM_train, earlystopping_hyperparameters, output_file_name_root, URM_train_last_test = None,
                                  n_cases = 35, n_random_starts = 5,
                                  evaluator_validation= None, evaluator_test=None, metric_to_optimize = "RECALL",
                                  output_folder_path ="result_experiments/"):



    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)


    parameterSearch = SearchBayesianSkopt(recommender_class, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)


    ##########################################################################################################

    hyperparameters_range_dictionary = {}
    hyperparameters_range_dictionary["epochs"] =  Categorical([1500])
    hyperparameters_range_dictionary["num_neurons"] = Integer(100, 400)
    hyperparameters_range_dictionary["num_factors"] = Integer(20, 70)
    hyperparameters_range_dictionary["dropout_percentage"] = Real(low = 0.0, high = 0.1, prior = 'uniform')
    hyperparameters_range_dictionary["learning_rate"] = Real(low = 1e-5, high = 1e-3, prior = 'log-uniform')
    hyperparameters_range_dictionary["regularization_rate"] = Real(low = 0.0, high = 0.2, prior = 'uniform')
    hyperparameters_range_dictionary["batch_size"] =  Categorical([128, 256, 512, 1024, 2048])

    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
        CONSTRUCTOR_KEYWORD_ARGS = {},
        FIT_POSITIONAL_ARGS = [],
        FIT_KEYWORD_ARGS = {**earlystopping_hyperparameters,
                            "use_gpu": False,
                            "epochs_min": 200,
                            "display_epoch": None,
                            "display_step": None,
                            "verbose": False}
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
                           output_folder_path = output_folder_path,
                           output_file_name_root = output_file_name_root,
                           metric_to_optimize = metric_to_optimize,
                           resume_from_saved = True,
                           recommender_input_args_last_test = recommender_input_args_last_test)






def read_data_split_and_search(dataset_name,
                                   flag_baselines_tune = False,
                                   flag_DL_article_default = False, flag_DL_tune = False,
                                   flag_print_results = False):
    
    result_folder_path = "result_experiments/{}/{}_{}/".format(CONFERENCE_NAME, ALGORITHM_NAME, dataset_name)


    if dataset_name == "movielens1m":
        dataset = Movielens1MReader(result_folder_path)
        article_hyperparameters = {'num_neurons': 300,
                                   'num_factors': 50,
                                   'dropout_percentage': 0.03,
                                   'learning_rate': 1e-4,
                                   'regularization_rate': 0.1,
                                   'epochs': 1500,
                                   'batch_size': 1024,
                                   'display_epoch': None,
                                   'display_step': None,
                                   'verbose': True
                             }
        early_stopping_epochs_min = 800


    elif dataset_name == "hetrec":
        dataset = MovielensHetrec2011Reader(result_folder_path)
        article_hyperparameters = {'num_neurons': 300,
                                   'num_factors': 50,
                                   'dropout_percentage': 0.03,
                                   'learning_rate': 1e-4,
                                   'regularization_rate': 0.1,
                                   'epochs': 1500,
                                   'batch_size': 1024,
                                   'display_epoch': None,
                                   'display_step': None,
                                   'verbose': True
                             }
        early_stopping_epochs_min = 800


    elif dataset_name == "filmtrust":
        dataset = FilmTrustReader(result_folder_path)
        article_hyperparameters = {'num_neurons': 150,
                                   'num_factors': 40,
                                   'dropout_percentage': 0.00,
                                   'learning_rate': 5e-5,
                                   'regularization_rate': 0.1,
                                   'epochs': 100,
                                   'batch_size': 1024,
                                   'display_epoch': None,
                                   'display_step': None,
                                   'verbose': True
                             }
        early_stopping_epochs_min = 0

    elif dataset_name == "frappe":
        dataset = FrappeReader(result_folder_path)
        article_hyperparameters = {'num_neurons': 300,
                                   'num_factors': 50,
                                   'dropout_percentage': 0.03,
                                   'learning_rate': 1e-4,
                                   'regularization_rate': 0.01,
                                   'epochs': 100,
                                   'batch_size': 1024,
                                   'display_epoch': None,
                                   'display_step': None,
                                   'verbose': True
                             }
        early_stopping_epochs_min = 0



    print ('Current dataset is: {}'.format(dataset_name))


    URM_train = dataset.URM_DICT["URM_train"].copy()
    URM_validation = dataset.URM_DICT["URM_validation"].copy()
    URM_test = dataset.URM_DICT["URM_test"].copy()



    # Ensure IMPLICIT data
    from Utils.assertions_on_data_for_experiments import assert_implicit_data, assert_disjoint_matrices

    assert_implicit_data([URM_train, URM_validation, URM_test])
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


    from Base.Evaluation.Evaluator import EvaluatorHoldout

    # use max cutoff to compute full MAP and NDCG
    max_cutoff = URM_train.shape[1]-1

    cutoff_list_validation = [10]
    cutoff_list_test=[5, 10, 50, max_cutoff]

    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=cutoff_list_validation)
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list_test)

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
                                    'lower_validations_allowed': 20,
                                    'evaluator_object': evaluator_validation,
                                    'validation_metric': metric_to_optimize,
                                    'epochs_min': early_stopping_epochs_min
                                    }
    

        try:


            parameterSearch = SearchSingleCase(UNeuRec_RecommenderWrapper,
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
                                   output_file_name_root = UNeuRec_RecommenderWrapper.RECOMMENDER_NAME)





        except Exception as e:

            print("On recommender {} Exception {}".format(UNeuRec_RecommenderWrapper, str(e)))
            traceback.print_exc()



        try:

            parameterSearch = SearchSingleCase(INeuRec_RecommenderWrapper,
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
                                   output_file_name_root = INeuRec_RecommenderWrapper.RECOMMENDER_NAME)


        except Exception as e:

            print("On recommender {} Exception {}".format(INeuRec_RecommenderWrapper, str(e)))
            traceback.print_exc()




    # if isUNeuRec_tune:
    #
    #     try:
    #
    #         runParameterSearch_NeuRec(UNeuRec_RecommenderWrapper,
    #                                  URM_train = URM_train,
    #                                  URM_train_last_test = URM_train + URM_validation,
    #                                  earlystopping_hyperparameters = earlystopping_hyperparameters,
    #                                  metric_to_optimize = metric_to_optimize,
    #                                  evaluator_validation = evaluator_validation,
    #                                  evaluator_test = evaluator_test,
    #                                  result_folder_path = result_folder_path,
    #                                  n_cases = n_cases,
    #                                  n_random_starts = n_random_starts,
    #                                  output_file_name_root = UNeuRec_RecommenderWrapper.RECOMMENDER_NAME)
    #
    #
    #     except Exception as e:
    #
    #         print("On recommender {} Exception {}".format(UNeuRec_RecommenderWrapper, str(e)))
    #         traceback.print_exc()
    #
    #
    #
    #
    #
    # if isINeuRec_tune:
    #
    #     try:
    #
    #         runParameterSearch_NeuRec(INeuRec_RecommenderWrapper,
    #                                  URM_train = URM_train,
    #                                  URM_train_last_test = URM_train + URM_validation,
    #                                  earlystopping_hyperparameters = earlystopping_hyperparameters,
    #                                  metric_to_optimize = metric_to_optimize,
    #                                  evaluator_validation = evaluator_validation,
    #                                  evaluator_test = evaluator_test,
    #                                  result_folder_path = result_folder_path,
    #                                  n_cases = n_cases,
    #                                  n_random_starts = n_random_starts,
    #                                  output_file_name_root = INeuRec_RecommenderWrapper.RECOMMENDER_NAME)
    #
    #
    #     except Exception as e:
    #
    #         print("On recommender {} Exception {}".format(INeuRec_RecommenderWrapper, str(e)))
    #         traceback.print_exc()
    #





    ################################################################################################
    ######
    ######      PRINT RESULTS
    ######

    if flag_print_results:

        n_test_users = np.sum(np.ediff1d(URM_test.indptr)>=1)
        file_name = "{}..//{}_{}_".format(result_folder_path, ALGORITHM_NAME, dataset_name)

        result_loader = ResultFolderLoader(result_folder_path,
                                         base_algorithm_list = None,
                                         other_algorithm_list = [INeuRec_RecommenderWrapper, UNeuRec_RecommenderWrapper],
                                         KNN_similarity_list = KNN_similarity_to_report_list,
                                         ICM_names_list = None,
                                         UCM_names_list = None)


        result_loader.generate_latex_results(file_name + "{}_latex_results.txt".format("article_metrics"),
                                           metrics_list = ["PRECISION", "RECALL", "MAP", "NDCG", "MRR"],
                                           cutoffs_list = [5, 10, 50],
                                           table_title = None,
                                           highlight_best = True)

        result_loader.generate_latex_results(file_name + "{}_latex_results.txt".format("beyond_accuracy_metrics"),
                                           metrics_list = ["DIVERSITY_MEAN_INTER_LIST", "DIVERSITY_HERFINDAHL", "COVERAGE_ITEM", "DIVERSITY_GINI", "SHANNON_ENTROPY"],
                                           cutoffs_list = [10],
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


    ALGORITHM_NAME = "NeuREC"
    CONFERENCE_NAME = "IJCAI"

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--baseline_tune',        help="Baseline hyperparameter search", type = bool, default = False)
    parser.add_argument('-a', '--DL_article_default',   help="Train the DL model with article hyperparameters", type = bool, default = False)
    parser.add_argument('-p', '--print_results',        help="Print results", type = bool, default = True)


    input_flags = parser.parse_args()
    print(input_flags)

    KNN_similarity_to_report_list = ["cosine", "dice", "jaccard", "asymmetric", "tversky"]


    dataset_list = ['frappe', 'filmtrust', 'movielens1m', 'hetrec']

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
                                      other_algorithm_list=[INeuRec_RecommenderWrapper, UNeuRec_RecommenderWrapper],
                                      KNN_similarity_to_report_list = KNN_similarity_to_report_list,
                                      split_per_algorithm_type = True)
