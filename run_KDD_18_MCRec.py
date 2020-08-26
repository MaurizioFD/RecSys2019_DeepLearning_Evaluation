#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17

@author: Maurizio Ferrari Dacrema
"""

from Recommender_import_list import *
from Conferences.KDD.MCRec_our_interface.MCRecRecommenderWrapper import MCRecML100k_RecommenderWrapper

from ParameterTuning.SearchSingleCase import SearchSingleCase
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs

from ParameterTuning.run_parameter_search import runParameterSearch_Collaborative, runParameterSearch_Content, runParameterSearch_Hybrid

import os, traceback, argparse
import numpy as np


from Utils.ResultFolderLoader import ResultFolderLoader, generate_latex_hyperparameters
from Utils.assertions_on_data_for_experiments import assert_implicit_data, assert_disjoint_matrices
from Utils.plot_popularity import plot_popularity_bias, save_popularity_statistics


def read_data_split_and_search(dataset_name,
                                   flag_baselines_tune = False,
                                   flag_DL_article_default = False, flag_DL_tune = False,
                                   flag_print_results = False):

    from Conferences.KDD.MCRec_our_interface.Movielens100K.Movielens100KReader import Movielens100KReader

    result_folder_path = "result_experiments/{}/{}_{}/".format(CONFERENCE_NAME, ALGORITHM_NAME, dataset_name)

    if dataset_name == "movielens100k":
        dataset = Movielens100KReader(result_folder_path)


    URM_train = dataset.URM_DICT["URM_train"].copy()
    URM_validation = dataset.URM_DICT["URM_validation"].copy()
    URM_test = dataset.URM_DICT["URM_test"].copy()
    URM_test_negative = dataset.URM_DICT["URM_test_negative"].copy()


    # Ensure IMPLICIT data and DISJOINT sets
    assert_implicit_data([URM_train, URM_validation, URM_test, URM_test_negative])
    assert_disjoint_matrices([URM_train, URM_validation, URM_test, URM_test_negative])



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



    from Base.Evaluation.Evaluator import EvaluatorNegativeItemSample

    evaluator_validation = EvaluatorNegativeItemSample(URM_validation, URM_test_negative, cutoff_list=[10])
    evaluator_test = EvaluatorNegativeItemSample(URM_test, URM_test_negative, cutoff_list=[10])



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

    metric_to_optimize = "PRECISION"
    n_cases = 50
    n_random_starts = 15


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
        ###### Content Baselines

        for ICM_name, ICM_object in dataset.ICM_DICT.items():

            try:

                runParameterSearch_Content(ItemKNNCBFRecommender,
                                            URM_train = URM_train,
                                            URM_train_last_test = URM_train + URM_validation,
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
                                            URM_train_last_test = URM_train + URM_validation,
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

        if dataset_name == "movielens100k":
            """
            The code provided by the original authors of MCRec can be used only for the original data.
            Here I am passing to the Wrapper the URM_train matrix that is only required for its shape,
            the train will be done using the preprocessed data the original authors provided
            """
            from Conferences.KDD.MCRec_github.code.Dataset import Dataset

            original_dataset_reader = Dataset('Conferences/KDD/MCRec_github/data/' + 'ml-100k')

            MCRec_article_hyperparameters = {
                "epochs": 200,
                "latent_dim": 128,
                "reg_latent": 0,
                "layers": [512, 256, 128, 64],
                "reg_layes": [0 ,0, 0, 0],
                "learning_rate": 1e-3,
                "batch_size": 256,
                "num_negatives": 4,
            }

            MCRec_earlystopping_hyperparameters = {
                "validation_every_n": 5,
                "stop_on_validation": True,
                "evaluator_object": evaluator_validation,
                "lower_validations_allowed": 5,
                "validation_metric": metric_to_optimize
            }



            parameterSearch = SearchSingleCase(MCRecML100k_RecommenderWrapper,
                                               evaluator_validation=evaluator_validation,
                                               evaluator_test=evaluator_test)

            recommender_input_args = SearchInputRecommenderArgs(
                                                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train, original_dataset_reader],
                                                FIT_KEYWORD_ARGS = MCRec_earlystopping_hyperparameters)

            recommender_input_args_last_test = recommender_input_args.copy()
            recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train + URM_validation

            parameterSearch.search(recommender_input_args,
                                   recommender_input_args_last_test = recommender_input_args_last_test,
                                   fit_hyperparameters_values=MCRec_article_hyperparameters,
                                   output_folder_path = result_folder_path,
                                   resume_from_saved = True,
                                   output_file_name_root = MCRecML100k_RecommenderWrapper.RECOMMENDER_NAME)




    ################################################################################################
    ######
    ######      PRINT RESULTS
    ######

    if flag_print_results:

        n_test_users = np.sum(np.ediff1d(URM_test.indptr)>=1)
        file_name = "{}..//{}_{}_".format(result_folder_path, ALGORITHM_NAME, dataset_name)

        ICM_names_to_report_list = list(dataset.ICM_DICT.keys())

        result_loader = ResultFolderLoader(result_folder_path,
                                         base_algorithm_list = None,
                                         other_algorithm_list = [MCRecML100k_RecommenderWrapper],
                                         KNN_similarity_list = KNN_similarity_to_report_list,
                                         ICM_names_list = ICM_names_to_report_list,
                                         UCM_names_list = None)


        result_loader.generate_latex_results(file_name + "{}_latex_results.txt".format("article_metrics"),
                                           metrics_list = ["PRECISION", "RECALL", "NDCG"],
                                           cutoffs_list = [10],
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



from functools import partial

if __name__ == '__main__':

    ALGORITHM_NAME = "MCRec"
    CONFERENCE_NAME = "KDD"


    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--baseline_tune',        help="Baseline hyperparameter search", type = bool, default = False)
    parser.add_argument('-a', '--DL_article_default',   help="Train the DL model with article hyperparameters", type = bool, default = False)
    parser.add_argument('-p', '--print_results',        help="Print results", type = bool, default = True)


    input_flags = parser.parse_args()
    print(input_flags)

    KNN_similarity_to_report_list = ["cosine", "dice", "jaccard", "asymmetric", "tversky"]



    dataset_list = ["movielens100k"]

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
                                      ICM_names_to_report_list = ["ICM_genre"],
                                      other_algorithm_list = [MCRecML100k_RecommenderWrapper],
                                      KNN_similarity_to_report_list = KNN_similarity_to_report_list,
                                      split_per_algorithm_type = True)
