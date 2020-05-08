#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17

@author: Maurizio Ferrari Dacrema
"""


from Recommender_import_list import *
from Conferences.WWW.MultiVAE_our_interface.MultiVAE_RecommenderWrapper import Mult_VAE_RecommenderWrapper


from ParameterTuning.SearchSingleCase import SearchSingleCase
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs

from ParameterTuning.run_parameter_search import runParameterSearch_Collaborative
from Utils.ResultFolderLoader import ResultFolderLoader, generate_latex_hyperparameters


from functools import partial
import os, traceback, argparse
import numpy as np

from Conferences.WWW.MultiVAE_our_interface.EvaluatorUserSubsetWrapper import EvaluatorUserSubsetWrapper, MF_cold_user_wrapper
from Utils.assertions_on_data_for_experiments import assert_implicit_data, assert_disjoint_matrices




######################################################################
from skopt.space import Real, Integer, Categorical


from ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from ParameterTuning.SearchSingleCase import SearchSingleCase
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs



def runParameterSearch_cold_user_MF(recommender_class, URM_train, URM_train_last_test = None, metric_to_optimize = "PRECISION",
                                     evaluator_validation = None, evaluator_test = None, evaluator_validation_earlystopping = None,
                                     output_folder_path ="result_experiments/",
                                     n_cases = 35, n_random_starts = 5, resume_from_saved = True):




    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    earlystopping_keywargs = {"validation_every_n": 5,
                              "stop_on_validation": True,
                              "evaluator_object": evaluator_validation_earlystopping,
                              "lower_validations_allowed": 5,
                              "validation_metric": metric_to_optimize,
                              }

    URM_train = URM_train.copy()

    if URM_train_last_test is not None:
        URM_train_last_test = URM_train_last_test.copy()

    try:

        output_file_name_root = recommender_class.RECOMMENDER_NAME


        ##########################################################################################################

        if recommender_class is MatrixFactorization_FunkSVD_Cython:

            hyperparameters_range_dictionary = {}
            hyperparameters_range_dictionary["sgd_mode"] = Categorical(["sgd", "adagrad", "adam"])
            hyperparameters_range_dictionary["epochs"] = Categorical([500])
            hyperparameters_range_dictionary["use_bias"] = Categorical([True, False])
            hyperparameters_range_dictionary["batch_size"] = Categorical([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
            hyperparameters_range_dictionary["num_factors"] = Integer(1, 200)
            hyperparameters_range_dictionary["item_reg"] = Real(low = 1e-5, high = 1e-2, prior = 'log-uniform')
            hyperparameters_range_dictionary["user_reg"] = Real(low = 1e-5, high = 1e-2, prior = 'log-uniform')
            hyperparameters_range_dictionary["learning_rate"] = Real(low = 1e-4, high = 1e-1, prior = 'log-uniform')
            hyperparameters_range_dictionary["negative_interactions_quota"] = Real(low = 0.0, high = 0.5, prior = 'uniform')

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [recommender_class, URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = earlystopping_keywargs
            )

        ##########################################################################################################

        if recommender_class is MatrixFactorization_AsySVD_Cython:

            hyperparameters_range_dictionary = {}
            hyperparameters_range_dictionary["sgd_mode"] = Categorical(["sgd", "adagrad", "adam"])
            hyperparameters_range_dictionary["epochs"] = Categorical([500])
            hyperparameters_range_dictionary["use_bias"] = Categorical([True, False])
            hyperparameters_range_dictionary["batch_size"] = Categorical([1])
            hyperparameters_range_dictionary["num_factors"] = Integer(1, 200)
            hyperparameters_range_dictionary["item_reg"] = Real(low = 1e-5, high = 1e-2, prior = 'log-uniform')
            hyperparameters_range_dictionary["user_reg"] = Real(low = 1e-5, high = 1e-2, prior = 'log-uniform')
            hyperparameters_range_dictionary["learning_rate"] = Real(low = 1e-4, high = 1e-1, prior = 'log-uniform')
            hyperparameters_range_dictionary["negative_interactions_quota"] = Real(low = 0.0, high = 0.5, prior = 'uniform')

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [recommender_class, URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = earlystopping_keywargs
            )

        ##########################################################################################################

        if recommender_class is MatrixFactorization_BPR_Cython:

            hyperparameters_range_dictionary = {}
            hyperparameters_range_dictionary["sgd_mode"] = Categorical(["sgd", "adagrad", "adam"])
            hyperparameters_range_dictionary["epochs"] = Categorical([1500])
            hyperparameters_range_dictionary["num_factors"] = Integer(1, 200)
            hyperparameters_range_dictionary["batch_size"] = Categorical([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
            hyperparameters_range_dictionary["positive_reg"] = Real(low = 1e-5, high = 1e-2, prior = 'log-uniform')
            hyperparameters_range_dictionary["negative_reg"] = Real(low = 1e-5, high = 1e-2, prior = 'log-uniform')
            hyperparameters_range_dictionary["learning_rate"] = Real(low = 1e-4, high = 1e-1, prior = 'log-uniform')

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [recommender_class, URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {**earlystopping_keywargs,
                                    "positive_threshold_BPR": None}
            )

        ##########################################################################################################

        if recommender_class is IALSRecommender:

            hyperparameters_range_dictionary = {}
            hyperparameters_range_dictionary["num_factors"] = Integer(1, 200)
            hyperparameters_range_dictionary["confidence_scaling"] = Categorical(["linear", "log"])
            hyperparameters_range_dictionary["alpha"] = Real(low = 1e-3, high = 50.0, prior = 'log-uniform')
            hyperparameters_range_dictionary["epsilon"] = Real(low = 1e-3, high = 10.0, prior = 'log-uniform')
            hyperparameters_range_dictionary["reg"] = Real(low = 1e-5, high = 1e-2, prior = 'log-uniform')

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [recommender_class, URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = earlystopping_keywargs
            )


        ##########################################################################################################

        if recommender_class is PureSVDRecommender:

            hyperparameters_range_dictionary = {}
            hyperparameters_range_dictionary["num_factors"] = Integer(1, 350)

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [recommender_class, URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {}
            )


        ##########################################################################################################

        if recommender_class is NMFRecommender:

            hyperparameters_range_dictionary = {}
            hyperparameters_range_dictionary["num_factors"] = Integer(1, 350)
            hyperparameters_range_dictionary["solver"] = Categorical(["coordinate_descent", "multiplicative_update"])
            hyperparameters_range_dictionary["init_type"] = Categorical(["random", "nndsvda"])
            hyperparameters_range_dictionary["beta_loss"] = Categorical(["frobenius", "kullback-leibler"])

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [recommender_class, URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {}
            )


        #########################################################################################################

        if URM_train_last_test is not None:
            recommender_input_args_last_test = recommender_input_args.copy()
            recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[1] = URM_train_last_test
        else:
            recommender_input_args_last_test = None


        parameterSearch = SearchBayesianSkopt(MF_cold_user_wrapper, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)

        hyperparameters_range_dictionary["estimate_model_for_cold_users"] = Categorical(["itemKNN", "mean_item_factors"])
        hyperparameters_range_dictionary["estimate_model_for_cold_users_topK"] = Integer(5, 1000)

        ## Final step, after the hyperparameter range has been defined for each type of algorithm
        parameterSearch.search(recommender_input_args,
                               parameter_search_space = hyperparameters_range_dictionary,
                               n_cases = n_cases,
                               n_random_starts = n_random_starts,
                               output_folder_path = output_folder_path,
                               output_file_name_root = output_file_name_root,
                               metric_to_optimize = metric_to_optimize,
                               resume_from_saved = resume_from_saved,
                               recommender_input_args_last_test = recommender_input_args_last_test)




    except Exception as e:

        print("On recommender {} Exception {}".format(recommender_class, str(e)))
        traceback.print_exc()

        error_file = open(output_folder_path + "ErrorLog.txt", "a")
        error_file.write("On recommender {} Exception {}\n".format(recommender_class, str(e)))
        error_file.close()







def read_data_split_and_search(dataset_name,
                                   flag_baselines_tune = False,
                                   flag_DL_article_default = False, flag_MF_baselines_tune = False, flag_DL_tune = False,
                                   flag_print_results = False):


    from Conferences.WWW.MultiVAE_our_interface.Movielens20M.Movielens20MReader import Movielens20MReader
    from Conferences.WWW.MultiVAE_our_interface.NetflixPrize.NetflixPrizeReader import NetflixPrizeReader

    split_type = "cold_user"

    result_folder_path = "result_experiments/{}/{}_{}_{}/".format(CONFERENCE_NAME, ALGORITHM_NAME, dataset_name, split_type)


    if dataset_name == "movielens20m":
        dataset = Movielens20MReader(result_folder_path, split_type = split_type)

    elif dataset_name == "netflixPrize":
        dataset = NetflixPrizeReader(result_folder_path)

    # If directory does not exist, create
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)


    metric_to_optimize = "NDCG"
    n_cases = 50
    n_random_starts = 15


    if split_type == "cold_user":


        collaborative_algorithm_list = [
            Random,
            TopPop,
            # UserKNNCFRecommender,
            ItemKNNCFRecommender,
            P3alphaRecommender,
            RP3betaRecommender,
            # PureSVDRecommender,
            # IALSRecommender,
            # NMFRecommender,
            # MatrixFactorization_BPR_Cython,
            # MatrixFactorization_FunkSVD_Cython,
            EASE_R_Recommender,
            SLIM_BPR_Cython,
            SLIMElasticNetRecommender,
        ]


        URM_train = dataset.URM_DICT["URM_train"].copy()
        URM_train_all = dataset.URM_DICT["URM_train_all"].copy()
        URM_validation = dataset.URM_DICT["URM_validation"].copy()
        URM_test = dataset.URM_DICT["URM_test"].copy()


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
    ###### Matrix Factorization Cold users

    collaborative_MF_algorithm_list = [
        PureSVDRecommender,
        IALSRecommender,
        NMFRecommender,
        MatrixFactorization_BPR_Cython,
        MatrixFactorization_FunkSVD_Cython,
    ]


    runParameterSearch_cold_user_MF_partial = partial(runParameterSearch_cold_user_MF,
                                                       URM_train = URM_train,
                                                       URM_train_last_test = URM_train + URM_validation,
                                                       metric_to_optimize = metric_to_optimize,
                                                       evaluator_validation_earlystopping = evaluator_validation,
                                                       evaluator_validation = evaluator_validation,
                                                       evaluator_test = evaluator_test,
                                                       output_folder_path = result_folder_path,
                                                       resume_from_saved = True,
                                                       n_cases = n_cases,
                                                       n_random_starts = n_random_starts)


    if flag_MF_baselines_tune:

        for recommender_class in collaborative_MF_algorithm_list:

            try:
                runParameterSearch_cold_user_MF_partial(recommender_class)

            except Exception as e:

                print("On recommender {} Exception {}".format(recommender_class, str(e)))
                traceback.print_exc()



    ################################################################################################
    ######
    ######      DL ALGORITHM
    ######

    if flag_DL_article_default:

        try:


            if dataset_name == "movielens20m":
                epochs = 100

            elif dataset_name == "netflixPrize":
                epochs = 200


            multiVAE_article_hyperparameters = {
                "epochs": epochs,
                "batch_size": 500,
                "total_anneal_steps": 200000,
                "p_dims": None,
            }

            multiVAE_earlystopping_hyperparameters = {
                "validation_every_n": 5,
                "stop_on_validation": True,
                "evaluator_object": evaluator_validation,
                "lower_validations_allowed": 5,
                "validation_metric": metric_to_optimize,
            }


            parameterSearch = SearchSingleCase(Mult_VAE_RecommenderWrapper,
                                               evaluator_validation=evaluator_validation,
                                               evaluator_test=evaluator_test)

            recommender_input_args = SearchInputRecommenderArgs(
                                                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                                                FIT_KEYWORD_ARGS = multiVAE_earlystopping_hyperparameters)

            recommender_input_args_last_test = recommender_input_args.copy()
            recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train + URM_validation

            parameterSearch.search(recommender_input_args,
                                   recommender_input_args_last_test = recommender_input_args_last_test,
                                   fit_hyperparameters_values=multiVAE_article_hyperparameters,
                                   output_folder_path = result_folder_path,
                                   resume_from_saved = True,
                                   output_file_name_root = Mult_VAE_RecommenderWrapper.RECOMMENDER_NAME)



        except Exception as e:

            print("On recommender {} Exception {}".format(Mult_VAE_RecommenderWrapper, str(e)))
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
                                         other_algorithm_list = [Mult_VAE_RecommenderWrapper],
                                         KNN_similarity_list = KNN_similarity_to_report_list,
                                         ICM_names_list = None,
                                         UCM_names_list = None)


        result_loader.generate_latex_results(file_name + "{}_latex_results.txt".format("article_metrics"),
                                           metrics_list = ["RECALL", "NDCG"],
                                           cutoffs_list = [20, 50, 100],
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





from functools import partial






if __name__ == '__main__':

    ALGORITHM_NAME = "Mult_VAE"
    CONFERENCE_NAME = "WWW"


    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--baseline_tune',        help="Baseline hyperparameter search", type = bool, default = False)
    parser.add_argument('-a', '--DL_article_default',   help="Train the DL model with article hyperparameters", type = bool, default = False)
    parser.add_argument('-p', '--print_results',        help="Print results", type = bool, default = True)

    parser.add_argument('-m', '--MF_baseline_tune',     help="Matrix Factorization hyperparameter search", type = bool, default = False)

    input_flags = parser.parse_args()
    print(input_flags)

    KNN_similarity_to_report_list = ["cosine", "dice", "jaccard", "asymmetric", "tversky"]


    dataset_list = ["movielens20m", "netflixPrize"]

    for dataset_name in dataset_list:

        read_data_split_and_search(dataset_name,
                                        flag_baselines_tune=input_flags.baseline_tune,
                                        flag_MF_baselines_tune = input_flags.MF_baseline_tune,
                                        flag_DL_article_default= input_flags.DL_article_default,
                                        flag_print_results = input_flags.print_results,
                                        )

    if input_flags.print_results:
        generate_latex_hyperparameters(result_folder_path ="result_experiments/{}/".format(CONFERENCE_NAME),
                                      algorithm_name= ALGORITHM_NAME,
                                      experiment_subfolder_list = ["{}_cold_user".format(dataset) for dataset in dataset_list],
                                      other_algorithm_list = [Mult_VAE_RecommenderWrapper],
                                      KNN_similarity_to_report_list = KNN_similarity_to_report_list,
                                      split_per_algorithm_type = True)
