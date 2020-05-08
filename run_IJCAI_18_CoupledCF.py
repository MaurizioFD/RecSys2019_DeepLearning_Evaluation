#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 13/03/19

@author: Simone Boglio
"""

from ParameterTuning.run_parameter_search import runParameterSearch_Collaborative, runParameterSearch_Content, runParameterSearch_Hybrid
# from ParameterTuning.parameter_search_ensemble import runParameterSearch_Ensemble
from ParameterTuning.SearchSingleCase import SearchSingleCase
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs


from Recommender_import_list import *

from Utils.ResultFolderLoader import ResultFolderLoader, generate_latex_hyperparameters
from Utils.print_negative_items_stats import print_negative_items_stats

from functools import partial
import numpy as np
import os, traceback, argparse

from Conferences.IJCAI.CoupledCF_our_interface.Movielens1MReader.Movielens1MReader import Movielens1MReader
from Conferences.IJCAI.CoupledCF_our_interface.TafengReader.TafengReader import TafengReader
from Conferences.IJCAI.CoupledCF_our_interface.CoupledCFWrapper import CoupledCF_RecommenderWrapper
from Conferences.IJCAI.CoupledCF_our_interface.DeepCFWrapper import DeepCF_RecommenderWrapper

def read_data_split_and_search(dataset_name,
                                   flag_baselines_tune = False,
                                   flag_DL_article_default = False, flag_DL_tune = False,
                                   flag_print_results = False):
                                   
                                   
                                   
    result_folder_path = "result_experiments/IJCAI/CoupledCF_{}/".format(dataset_name)

    #Logger(path=result_folder_path, name_file='CoupledCF_' + dataset_name)

    if dataset_name.startswith("movielens1m"):

        if dataset_name.endswith("_original"):
            dataset = Movielens1MReader(result_folder_path, type ='original')
        elif dataset_name.endswith("_ours"):
            dataset = Movielens1MReader(result_folder_path, type ='ours')
        else:
            print("Dataset name not supported, current is {}".format(dataset_name))
            return


        UCM_to_report = ["UCM_all"]
        ICM_to_report = ["ICM_all"]

        UCM_CoupledCF = dataset.ICM_DICT["UCM_all"]
        ICM_CoupledCF = dataset.ICM_DICT["ICM_all"]


    elif dataset_name.startswith("tafeng"):

        if dataset_name.endswith("_original"):
            dataset = TafengReader(result_folder_path, type ='original')
        elif dataset_name.endswith("_ours"):
            dataset = TafengReader(result_folder_path, type ='ours')
        else:
            print("Dataset name not supported, current is {}".format(dataset_name))
            return

        UCM_to_report = ["UCM_all"]
        ICM_to_report = ["ICM_original"]

        UCM_CoupledCF = dataset.ICM_DICT["UCM_all"]
        ICM_CoupledCF = dataset.ICM_DICT["ICM_original"]

    else:
        print("Dataset name not supported, current is {}".format(dataset_name))
        return

    print ('Current dataset is: {}'.format(dataset_name))


    UCM_dict = {UCM_name:UCM_object for (UCM_name,UCM_object) in dataset.ICM_DICT.items() if "UCM" in UCM_name}
    ICM_dict = {UCM_name:UCM_object for (UCM_name,UCM_object) in dataset.ICM_DICT.items() if "ICM" in UCM_name}



    URM_train = dataset.URM_DICT["URM_train"].copy()
    URM_validation = dataset.URM_DICT["URM_validation"].copy()
    URM_test = dataset.URM_DICT["URM_test"].copy()
    URM_test_negative = dataset.URM_DICT["URM_test_negative"].copy()

    # Matrices are 1-indexed, so remove first row
    print_negative_items_stats(URM_train[1:], URM_validation[1:], URM_test[1:], URM_test_negative[1:])

    # Ensure IMPLICIT data
    from Utils.assertions_on_data_for_experiments import assert_implicit_data, assert_disjoint_matrices

    assert_implicit_data([URM_train, URM_validation, URM_test, URM_test_negative])
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

    cutoff_list_validation = [5]
    cutoff_list_test = [1,2,3,4,5,6,7,8,9,10]
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


        ###############################################################################################
        ##### Item Content Baselines

        for ICM_name, ICM_object in ICM_dict.items():

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

                print("On CBF recommender for ICM {} Exception {}".format(ICM_name, str(e)))
                traceback.print_exc()




        ################################################################################################
        ###### User Content Baselines

        for UCM_name, UCM_object in UCM_dict.items():

            try:

                runParameterSearch_Content(UserKNNCBFRecommender,
                                            URM_train = URM_train,
                                            URM_train_last_test = URM_train + URM_validation,
                                            metric_to_optimize = metric_to_optimize,
                                            evaluator_validation = evaluator_validation,
                                            evaluator_test = evaluator_test,
                                            output_folder_path = result_folder_path,
                                            parallelizeKNN = False,
                                            allow_weighting = True,
                                            resume_from_saved = True,
                                            ICM_name = UCM_name,
                                            ICM_object = UCM_object.copy(),
                                            n_cases = n_cases,
                                            n_random_starts = n_random_starts)



                runParameterSearch_Hybrid(UserKNN_CFCBF_Hybrid_Recommender,
                                            URM_train = URM_train,
                                            URM_train_last_test = URM_train + URM_validation,
                                            metric_to_optimize = metric_to_optimize,
                                            evaluator_validation = evaluator_validation,
                                            evaluator_test = evaluator_test,
                                            output_folder_path = result_folder_path,
                                            parallelizeKNN = False,
                                            allow_weighting = True,
                                            resume_from_saved = True,
                                            ICM_name = UCM_name,
                                            ICM_object = UCM_object.copy(),
                                            n_cases = n_cases,
                                            n_random_starts = n_random_starts)

            except Exception as e:

                print("On CBF recommender for UCM {} Exception {}".format(UCM_name, str(e)))
                traceback.print_exc()


    ################################################################################################
    ######
    ######      DL ALGORITHM
    ######

    if flag_DL_article_default:
        
        model_name = dataset.DATASET_NAME

    
        earlystopping_hyperparameters = {
                                    'validation_every_n': 5,
                                    'stop_on_validation': True,
                                    'lower_validations_allowed': 5,
                                    'evaluator_object': evaluator_validation,
                                    'validation_metric': metric_to_optimize
                                    }


        if 'tafeng' in dataset_name:
            model_number = 3
            article_hyperparameters = {'learning_rate': 0.005,
                                  'epochs': 100,
                                  'n_negative_sample': 4,
                                  'temp_file_folder': None,
                                  'dataset_name': model_name,
                                  'number_model': model_number,
                                  'verbose': 0,
                                  'plot_model': False,
                                  }
        else:
            # movielens1m and other dataset
            model_number = 3
            article_hyperparameters = {'learning_rate': 0.001,
                                  'epochs': 100,
                                  'n_negative_sample': 4,
                                  'temp_file_folder': None,
                                  'dataset_name': model_name,
                                  'number_model': model_number,
                                  'verbose': 0,
                                  'plot_model': False,
                                  }


        parameterSearch = SearchSingleCase(DeepCF_RecommenderWrapper,
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
                               output_file_name_root=DeepCF_RecommenderWrapper.RECOMMENDER_NAME)



        if 'tafeng' in dataset_name:
            # tafeng model has a different structure
            model_number = 2
            article_hyperparameters = {'learning_rate': 0.005,
                                  'epochs': 100,
                                  'n_negative_sample': 4,
                                  'temp_file_folder': None,
                                  'dataset_name': "Tafeng",
                                  'number_model': model_number,
                                  'verbose': 0,
                                  'plot_model': False,
                                  }
        else:
            # movielens1m use this tructure with model 2
            model_number = 2
            article_hyperparameters = {'learning_rate': 0.001,
                                  'epochs': 100,
                                  'n_negative_sample': 4,
                                  'temp_file_folder': None,
                                  'dataset_name': "Movielens1M",
                                  'number_model': model_number,
                                  'verbose': 0,
                                  'plot_model': False,
                                  }



        parameterSearch = SearchSingleCase(CoupledCF_RecommenderWrapper,
                                           evaluator_validation=evaluator_validation,
                                           evaluator_test=evaluator_test)

        recommender_input_args = SearchInputRecommenderArgs(CONSTRUCTOR_POSITIONAL_ARGS=[URM_train, UCM_CoupledCF, ICM_CoupledCF],
                                                            FIT_KEYWORD_ARGS=earlystopping_hyperparameters)


        recommender_input_args_last_test = recommender_input_args.copy()
        recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train + URM_validation


        parameterSearch.search(recommender_input_args,
                               recommender_input_args_last_test = recommender_input_args_last_test,
                               fit_hyperparameters_values=article_hyperparameters,
                               output_folder_path=result_folder_path,
                               resume_from_saved = True,
                               output_file_name_root=CoupledCF_RecommenderWrapper.RECOMMENDER_NAME)





    ################################################################################################
    ######
    ######      PRINT RESULTS
    ######

    if flag_print_results:

        n_test_users = np.sum(np.ediff1d(URM_test.indptr)>=1)
        file_name = "{}..//{}_{}_".format(result_folder_path, ALGORITHM_NAME, dataset_name)

        result_loader = ResultFolderLoader(result_folder_path,
                                         base_algorithm_list = None,
                                         other_algorithm_list = [DeepCF_RecommenderWrapper, CoupledCF_RecommenderWrapper],
                                         KNN_similarity_list = KNN_similarity_to_report_list,
                                         ICM_names_list = ICM_to_report,
                                         UCM_names_list = UCM_to_report)


        result_loader.generate_latex_results(file_name + "{}_latex_results.txt".format("article_metrics"),
                                           metrics_list = ["HIT_RATE", "NDCG"],
                                           cutoffs_list = [1, 5, 10],
                                           table_title = None,
                                           highlight_best = True)

        result_loader.generate_latex_results(file_name + "{}_latex_results.txt".format("beyond_accuracy_metrics"),
                                           metrics_list = ["DIVERSITY_MEAN_INTER_LIST", "DIVERSITY_HERFINDAHL", "COVERAGE_ITEM", "DIVERSITY_GINI", "SHANNON_ENTROPY"],
                                           cutoffs_list = [5],
                                           table_title = None,
                                           highlight_best = True)

        result_loader.generate_latex_results(file_name + "{}_latex_results.txt".format("all_metrics"),
                                           metrics_list = ["PRECISION", "RECALL", "MAP", "MRR", "NDCG", "F1", "HIT_RATE", "ARHR",
                                                           "NOVELTY", "DIVERSITY_MEAN_INTER_LIST", "DIVERSITY_HERFINDAHL", "COVERAGE_ITEM", "DIVERSITY_GINI", "SHANNON_ENTROPY"],
                                           cutoffs_list = [5],
                                           table_title = None,
                                           highlight_best = True)

        result_loader.generate_latex_time_statistics(file_name + "{}_latex_results.txt".format("time"),
                                           n_evaluation_users=n_test_users,
                                           table_title = None)


if __name__ == '__main__':


    ALGORITHM_NAME = "CoupledCF"
    CONFERENCE_NAME = "IJCAI"


    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--baseline_tune',        help="Baseline hyperparameter search", type = bool, default = False)
    parser.add_argument('-a', '--DL_article_default',   help="Train the DL model with article hyperparameters", type = bool, default = False)
    parser.add_argument('-p', '--print_results',        help="Print results", type = bool, default = True)


    input_flags = parser.parse_args()
    print(input_flags)

    KNN_similarity_to_report_list = ["cosine", "dice", "jaccard", "asymmetric", "tversky"]

    dataset_list = ['movielens1m_original', 'movielens1m_ours', 'tafeng_original', 'tafeng_ours']


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
                                     ICM_names_to_report_list=["ICM_all", "ICM_original"],
                                     UCM_names_to_report_list=["UCM_all"],
                                     other_algorithm_list=[DeepCF_RecommenderWrapper, CoupledCF_RecommenderWrapper],
                                     KNN_similarity_to_report_list = KNN_similarity_to_report_list,
                                     split_per_algorithm_type = True)
