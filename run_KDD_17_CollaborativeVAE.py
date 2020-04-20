#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17

@author: Maurizio Ferrari Dacrema
"""


from Recommender_import_list import *
from Conferences.KDD.CollaborativeVAE_our_interface.CollaborativeVAE_RecommenderWrapper import CollaborativeVAE_RecommenderWrapper

from ParameterTuning.run_parameter_search import runParameterSearch_Collaborative, runParameterSearch_Content, runParameterSearch_Hybrid

from ParameterTuning.SearchSingleCase import SearchSingleCase
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs

from functools import partial
import os, traceback, argparse
import numpy as np



from Utils.ResultFolderLoader import ResultFolderLoader, generate_latex_hyperparameters
from Utils.assertions_on_data_for_experiments import assert_implicit_data, assert_disjoint_matrices







def read_data_split_and_search(dataset_variant, train_interactions,
                                   flag_baselines_tune = False,
                                   flag_DL_article_default = False, flag_DL_tune = False,
                                   flag_print_results = False):



    from Conferences.KDD.CollaborativeVAE_our_interface.Citeulike.CiteulikeReader import CiteulikeReader


    result_folder_path = "result_experiments/{}/{}_citeulike_{}_{}/".format(CONFERENCE_NAME, ALGORITHM_NAME, dataset_variant, train_interactions)

    dataset = CiteulikeReader(result_folder_path, dataset_variant = dataset_variant, train_interactions = train_interactions)


    URM_train = dataset.URM_DICT["URM_train"].copy()
    URM_validation = dataset.URM_DICT["URM_validation"].copy()
    URM_test = dataset.URM_DICT["URM_test"].copy()
    del dataset.ICM_DICT["ICM_tokens_bool"]

    # Ensure IMPLICIT data
    assert_implicit_data([URM_train, URM_validation, URM_test])

    # Due to the sparsity of the dataset, choosing an evaluation as subset of the train
    # While keeping validation interaction in the train set
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

    metric_to_optimize = "RECALL"
    n_cases = 50
    n_random_starts = 15


    runParameterSearch_Collaborative_partial = partial(runParameterSearch_Collaborative,
                                                       URM_train = URM_train,
                                                       URM_train_last_test = URM_train_last_test,
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
        ###### Content Baselines

        for ICM_name, ICM_object in dataset.ICM_DICT.items():

            try:

                runParameterSearch_Content(ItemKNNCBFRecommender,
                                            URM_train = URM_train,
                                            URM_train_last_test = URM_train_last_test,
                                            metric_to_optimize = metric_to_optimize,
                                            evaluator_validation = evaluator_validation,
                                            evaluator_test = evaluator_test,
                                            output_folder_path = result_folder_path,
                                            parallelizeKNN = False,
                                            allow_weighting = True,
                                            resume_from_saved = True,
                                            ICM_name = ICM_name,
                                            ICM_object = ICM_object.copy(),
                                            n_cases = n_cases,
                                            n_random_starts = n_random_starts)

            except Exception as e:

                print("On CBF recommender for ICM {} Exception {}".format(ICM_name, str(e)))
                traceback.print_exc()


        ################################################################################################
        ###### Hybrid

        for ICM_name, ICM_object in dataset.ICM_DICT.items():

            try:

                runParameterSearch_Hybrid(ItemKNN_CFCBF_Hybrid_Recommender,
                                            URM_train = URM_train,
                                            URM_train_last_test = URM_train_last_test,
                                            metric_to_optimize = metric_to_optimize,
                                            evaluator_validation = evaluator_validation,
                                            evaluator_test = evaluator_test,
                                            output_folder_path = result_folder_path,
                                            parallelizeKNN = False,
                                            allow_weighting = True,
                                            resume_from_saved = True,
                                            ICM_name = ICM_name,
                                            ICM_object = ICM_object.copy(),
                                            n_cases = n_cases,
                                            n_random_starts = n_random_starts)


            except Exception as e:

                print("On recommender {} Exception {}".format(ItemKNN_CFCBF_Hybrid_Recommender, str(e)))
                traceback.print_exc()



    ################################################################################################
    ######
    ######      DL ALGORITHM
    ######

    if flag_DL_article_default:

        try:

            cvae_recommender_article_hyperparameters = {
                "epochs": 200,
                "learning_rate_vae": 1e-2,
                "learning_rate_cvae": 1e-3,
                "num_factors": 50,
                "dimensions_vae": [200, 100],
                "epochs_vae": [50, 50],
                "batch_size": 128,
                "lambda_u": 0.1,
                "lambda_v": 10,
                "lambda_r": 1,
                "a": 1,
                "b": 0.01,
                "M": 300,
            }



            cvae_earlystopping_hyperparameters = {
                "validation_every_n": 5,
                "stop_on_validation": True,
                "evaluator_object": evaluator_validation,
                "lower_validations_allowed": 5,
                "validation_metric": metric_to_optimize
            }

            parameterSearch = SearchSingleCase(CollaborativeVAE_RecommenderWrapper,
                                               evaluator_validation=evaluator_validation,
                                               evaluator_test=evaluator_test)

            recommender_input_args = SearchInputRecommenderArgs(
                                                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train, dataset.ICM_DICT["ICM_tokens_TFIDF"]],
                                                FIT_KEYWORD_ARGS = cvae_earlystopping_hyperparameters)


            recommender_input_args_last_test = recommender_input_args.copy()
            recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train_last_test


            parameterSearch.search(recommender_input_args,
                                   recommender_input_args_last_test = recommender_input_args_last_test,
                                   fit_hyperparameters_values=cvae_recommender_article_hyperparameters,
                                   output_folder_path = result_folder_path,
                                   resume_from_saved = True,
                                   output_file_name_root = CollaborativeVAE_RecommenderWrapper.RECOMMENDER_NAME)




        except Exception as e:

            print("On recommender {} Exception {}".format(CollaborativeVAE_RecommenderWrapper, str(e)))
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
                                         other_algorithm_list = other_algorithm_list,
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





__name__ = "__main__"


if __name__ == '__main__':

    ALGORITHM_NAME = "CollaborativeVAE"
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


    from collections import namedtuple

    CustomRecommenderName = namedtuple('CustomRecommenderName', ['RECOMMENDER_NAME'])

    other_algorithm_list_names = [CollaborativeVAE_RecommenderWrapper.RECOMMENDER_NAME, "CollaborativeDL_Matlab_RecommenderWrapper"]
    other_algorithm_list = [CustomRecommenderName(RECOMMENDER_NAME = recommender_name) for recommender_name in other_algorithm_list_names]



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
                                      ICM_names_to_report_list = ["ICM_tokens_TFIDF"],
                                      KNN_similarity_to_report_list = KNN_similarity_to_report_list,
                                      other_algorithm_list = other_algorithm_list,
                                     split_per_algorithm_type = True)
