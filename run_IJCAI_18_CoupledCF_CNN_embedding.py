#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 13/03/19

@author: Maurizio Ferrari Dacrema
"""

from ParameterTuning.SearchSingleCase import SearchSingleCase
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from CNN_on_embeddings.run_CNN_embedding_evaluation_ablation import run_evaluation_ablation
from Base.DataIO import DataIO
import os, argparse
from Recommender_import_list import *
from functools import partial
import multiprocessing

from CNN_on_embeddings.IJCAI.CoupledCF_our_interface.Movielens1MReader.Movielens1MReader import Movielens1MReader_Wrapper
from CNN_on_embeddings.IJCAI.CoupledCF_our_interface.TafengReader.TafengReader import TafengReader_Wrapper
from CNN_on_embeddings.IJCAI.CoupledCF_our_interface.CoupledCFWrapper import CoupledCF_RecommenderWrapper

from Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out
from Data_manager.DataSplitter_k_fold_random import DataSplitter_k_fold_random_fromDataSplitter
from Utils.assertions_on_data_for_experiments import assert_implicit_data, assert_disjoint_matrices
from Base.Evaluation.Evaluator import EvaluatorNegativeItemSample
from CNN_on_embeddings.read_CNN_embedding_evaluation_results import read_permutation_results
from ParameterTuning.run_parameter_search import runParameterSearch_Collaborative
from Utils.ResultFolderLoader import ResultFolderLoader


import numpy as np

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Lambda
import tensorflow as tf


def get_CoupledCF_assert_model(embedding_size, map_mode = "full_map"):

    map_mode_flag_main_diagonal = map_mode == "main_diagonal"
    map_mode_flag_off_diagonal = map_mode == "off_diagonal"

    merge_attr_embedding = Input(shape=(embedding_size, embedding_size), dtype='float32', name='merge_attr_embedding')

    # If using only the diagonal, remove everything not in the diagonal
    if map_mode_flag_main_diagonal:
        print("CoupledCF: Using main diagonal elements.")
        diagonal = Lambda(lambda x: tf.linalg.diag_part(x))(merge_attr_embedding)
        merge_attr_embedding_mode = Lambda(lambda x: tf.linalg.set_diag(K.zeros_like(merge_attr_embedding), x) )(diagonal)

    elif map_mode_flag_off_diagonal:
        print("CoupledCF: Using off diagonal elements.")
        diagonal = K.zeros_like( Lambda(lambda x: tf.linalg.diag_part(x))(merge_attr_embedding) )
        merge_attr_embedding_mode = Lambda(lambda x: tf.linalg.set_diag(x, diagonal) )(merge_attr_embedding)

    else:
        print("CoupledCF: Using all map elements.")
        merge_attr_embedding_mode = merge_attr_embedding

    # Final prediction layer
    model = Model(inputs = [merge_attr_embedding],
                  outputs = merge_attr_embedding_mode)

    return model




def get_hyperparameters_for_dataset(dataset_name):

    if dataset_name == 'tafeng':

        article_hyperparameters = {
            'learning_rate': 0.005,
            'epochs': 100,
            'n_negative_sample': 4,
            'dataset_name': "Tafeng",
            'number_model': 2,
            'verbose': 0,
            'plot_model': False,
        }
    elif dataset_name == 'movielens1m':

        article_hyperparameters = {
            "learning_rate": 0.001,
            "epochs": 100,
            "n_negative_sample": 4,
            "dataset_name": "Movielens1M",
            "number_model": 2,
            "verbose": 0,
            "plot_model": False,
        }
    else:
        raise ValueError("Invalid dataset name")

    return article_hyperparameters





def run_train_with_early_stopping(dataset_name, URM_train, URM_validation,
                                  UCM_CoupledCF, ICM_CoupledCF,
                                  evaluator_validation, evaluator_test,
                                  metric_to_optimize, result_folder_path,
                                  map_mode):


    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)


    article_hyperparameters = get_hyperparameters_for_dataset(dataset_name)
    article_hyperparameters["map_mode"] = map_mode

    earlystopping_hyperparameters = {
        "validation_every_n": 5,
        "stop_on_validation": True,
        "lower_validations_allowed": 5,
        "evaluator_object": evaluator_validation,
        "validation_metric": metric_to_optimize
    }

    parameterSearch = SearchSingleCase(CoupledCF_RecommenderWrapper,
                                       evaluator_validation=evaluator_validation,
                                       evaluator_test=evaluator_test)

    recommender_input_args = SearchInputRecommenderArgs(CONSTRUCTOR_POSITIONAL_ARGS=[URM_train, UCM_CoupledCF, ICM_CoupledCF],
                                                        FIT_KEYWORD_ARGS=earlystopping_hyperparameters)

    recommender_input_args_last_test = recommender_input_args.copy()
    recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train + URM_validation

    parameterSearch.search(recommender_input_args,
                            recommender_input_args_last_test=recommender_input_args_last_test,
                            fit_hyperparameters_values=article_hyperparameters,
                            output_folder_path=result_folder_path,
                            output_file_name_root=CoupledCF_RecommenderWrapper.RECOMMENDER_NAME,
                            save_model = "last",
                            resume_from_saved=True,
                            evaluate_on_test = "last")


    dataIO = DataIO(result_folder_path)
    search_metadata = dataIO.load_data(file_name=CoupledCF_RecommenderWrapper.RECOMMENDER_NAME + "_metadata.zip")

    return search_metadata



def get_URM_negatives_without_cold_users(removed_cold_users, URM_test_negative):

    if removed_cold_users is None:
        return URM_test_negative.copy()

    users_to_preserve_mask = np.ones(URM_test_negative.shape[0], dtype=np.bool)
    users_to_preserve_mask[removed_cold_users] = False
    URM_test_negative_fold = URM_test_negative[users_to_preserve_mask,:]

    return URM_test_negative_fold



if __name__ == '__main__':

    ALGORITHM_NAME = "CoupledCF"
    CONFERENCE_NAME = "IJCAI"

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_name',        help = "Dataset name", type = str, default = "movielens1m")
    parser.add_argument('-b', '--run_baselines',        help = "Run hyperparameter tuning", type = bool, default = True)
    parser.add_argument('-a', '--run_eval_ablation',   help = "Run Study 2 experiments", type = bool, default = True)
    parser.add_argument('-n', '--n_folds',             help = "Number of folds", type = int, default = 20)

    input_flags = parser.parse_args()
    print(input_flags)


    output_folder_path = "result_experiments/CoupledCF_{}/".format(input_flags.dataset_name)


    if input_flags.dataset_name == "movielens1m":
        data_reader = Movielens1MReader_Wrapper(output_folder_path + "data/", type="original")

    elif input_flags.dataset_name == "tafeng":
        data_reader = TafengReader_Wrapper(output_folder_path + "data/", type="original")

    else:
        print("Dataset name not supported, current is {}".format(input_flags.dataset_name))
        exit()


    print ("Current dataset is: {}".format(input_flags.dataset_name))

    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    data_loaded = data_reader.load_data()
    URM_test_negative = data_loaded.AVAILABLE_URM["URM_test_negative"].copy()

    dataSplitter_kwargs = {
        "k_out_value": 1,
        "use_validation_set": True,
        "leave_random_out": True,
    }

    dataSplitter_k_fold = DataSplitter_k_fold_random_fromDataSplitter(data_reader, DataSplitter_leave_k_out,
                                                                    dataSplitter_kwargs = dataSplitter_kwargs,
                                                                    n_folds = input_flags.n_folds,
                                                                    preload_all = False)

    dataSplitter_k_fold.load_data(save_folder_path = output_folder_path + "data/folds/")


    cutoff_list_validation = [5]
    cutoff_list_test = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    metric_to_optimize = "NDCG"

    ################################################################################################################################################
    ###############################
    ###############################         Test code on fake object to verify the alterations to the interaction map do what they are supposed to
    ###############################
    ################################################################################################################################################


    embedding_size = 8
    interaction_map = np.ones((1, embedding_size, embedding_size))

    result_all_map = interaction_map.copy().squeeze()

    model = get_CoupledCF_assert_model(embedding_size, map_mode = "main_diagonal")
    result_main_diag = model.predict(interaction_map).squeeze()

    model = get_CoupledCF_assert_model(embedding_size, map_mode = "off_diagonal")
    result_off_diag = model.predict(interaction_map).squeeze()

    assert np.allclose(result_main_diag.diagonal(), result_all_map.diagonal()), "two operations have different diagonal"
    assert np.allclose(result_main_diag, np.diag(result_main_diag.diagonal())), "result_main_diag has off diagonal elements"
    assert not np.allclose(result_all_map, np.diag(result_all_map.diagonal())), "result_all_map has NO off diagonal elements"

    assert np.allclose(result_all_map, result_main_diag + result_off_diag), "triangular composition non consistent"



    ################################################################################################################################################
    ###############################
    ###############################         ABLATION EXPERIMENT
    ###############################
    ################################################################################################################################################


    if input_flags.run_eval_ablation:

        for fold_index, dataSplitter_fold in enumerate(dataSplitter_k_fold):

            URM_train, URM_validation, URM_test = dataSplitter_fold.get_holdout_split()
            UCM_CoupledCF = dataSplitter_fold.get_UCM_from_name("UCM_all")
            ICM_CoupledCF = dataSplitter_fold.get_ICM_from_name("ICM_all")

            # Ensure negative items are consistent with positive items, accounting for removed cold users
            URM_test_negative_fold = get_URM_negatives_without_cold_users(dataSplitter_fold.removed_cold_users, URM_test_negative)

            # ensure IMPLICIT data
            assert_implicit_data([URM_train, URM_validation, URM_test, URM_test_negative_fold])
            assert_disjoint_matrices([URM_train, URM_validation, URM_test])

            evaluator_validation = EvaluatorNegativeItemSample(URM_validation, URM_test_negative_fold, cutoff_list=cutoff_list_validation)
            evaluator_test = EvaluatorNegativeItemSample(URM_test, URM_test_negative_fold, cutoff_list=cutoff_list_test)
            
            recommender_input_args = SearchInputRecommenderArgs(CONSTRUCTOR_POSITIONAL_ARGS=[URM_train, UCM_CoupledCF, ICM_CoupledCF])


            # Ablation with training on selected mode
            for map_mode in ["all_map", "main_diagonal", "off_diagonal"]:

                result_folder_path = os.path.join(output_folder_path, "fit_ablation_{}/{}_{}/".format(map_mode, map_mode, fold_index))

                search_metadata = run_train_with_early_stopping(input_flags.dataset_name,
                                                                URM_train, URM_validation,
                                                                UCM_CoupledCF, ICM_CoupledCF,
                                                                evaluator_validation,
                                                                evaluator_test,
                                                                metric_to_optimize,
                                                                result_folder_path,
                                                                map_mode = map_mode)


            # Ablation evaluating full map mode
            for map_mode in ["all_map", "main_diagonal", "off_diagonal"]:

                input_folder_path = os.path.join(output_folder_path, "fit_ablation_{}/{}_{}/".format("all_map", "all_map", fold_index))
                result_folder_path = os.path.join(output_folder_path, "evaluation_ablation_{}/{}_{}/".format(map_mode, map_mode, fold_index))

                run_evaluation_ablation(recommender_class=CoupledCF_RecommenderWrapper,
                                        recommender_input_args = recommender_input_args,
                                        evaluator_test = evaluator_test,
                                        input_folder_path = input_folder_path,
                                        result_folder_path = result_folder_path,
                                        map_mode = map_mode)



    read_permutation_results(output_folder_path, input_flags.n_folds, 10,
                             ["PRECISION", "MAP", "NDCG", "F1", "HIT_RATE"],
                             file_result_name_root = "latex_fit_ablation_results",
                             convolution_model_name = CoupledCF_RecommenderWrapper.RECOMMENDER_NAME,
                             pretrained_model_name = None,
                             pretrained_model_class = None,
                             experiment_type = "fit_ablation")


    read_permutation_results(output_folder_path, input_flags.n_folds, 10,
                             ["PRECISION", "MAP", "NDCG", "F1", "HIT_RATE"],
                             file_result_name_root = "latex_evaluation_ablation_results",
                             convolution_model_name = CoupledCF_RecommenderWrapper.RECOMMENDER_NAME,
                             pretrained_model_name = None,
                             pretrained_model_class = None,
                             experiment_type = "evaluation_ablation")








    ################################################################################################################################################
    ###############################
    ###############################         HYPERPARAMETER TUNING BASELINES
    ###############################
    ################################################################################################################################################

    collaborative_algorithm_list = [
        Random,
        TopPop,
        UserKNNCFRecommender,
        ItemKNNCFRecommender,
        P3alphaRecommender,
        RP3betaRecommender,
        PureSVDRecommender,
        # NMFRecommender,
        IALSRecommender,
        # MatrixFactorization_BPR_Cython,
        # MatrixFactorization_FunkSVD_Cython,
        # EASE_R_Recommender,
    ]

    n_cases = 50
    n_random_starts = 15
    result_baselines_folder_path = output_folder_path + "baselines/"

    dataSplitter_fold = dataSplitter_k_fold[0]
    URM_train, URM_validation, URM_test = dataSplitter_fold.get_holdout_split()

    # Ensure negative items are consistent with positive items, accounting for removed cold users
    URM_test_negative_fold = get_URM_negatives_without_cold_users(dataSplitter_fold.removed_cold_users, URM_test_negative)

    # ensure IMPLICIT data
    assert_implicit_data([URM_train, URM_validation, URM_test, URM_test_negative_fold])
    assert_disjoint_matrices([URM_train, URM_validation, URM_test])

    evaluator_validation = EvaluatorNegativeItemSample(URM_validation, URM_test_negative_fold, cutoff_list=cutoff_list_validation)
    evaluator_test = EvaluatorNegativeItemSample(URM_test, URM_test_negative_fold, cutoff_list=cutoff_list_test)


    hyperparameter_search_collaborative_partial = partial(runParameterSearch_Collaborative,
                                                          URM_train = URM_train,
                                                          URM_train_last_test = URM_train + URM_validation,
                                                          metric_to_optimize = metric_to_optimize,
                                                          evaluator_validation_earlystopping = evaluator_validation,
                                                          evaluator_validation = evaluator_validation,
                                                          evaluator_test = evaluator_test,
                                                          output_folder_path = result_baselines_folder_path,
                                                          parallelizeKNN = False,
                                                          allow_weighting = True,
                                                          resume_from_saved = True,
                                                          n_cases = n_cases,
                                                          n_random_starts = n_random_starts)



    if input_flags.run_baselines:

        pool = multiprocessing.Pool(processes=3, maxtasksperchild=1)
        pool.map(hyperparameter_search_collaborative_partial, collaborative_algorithm_list)

        pool.close()
        pool.join()


        n_test_users = np.sum(np.ediff1d(URM_test.indptr)>=1)
        file_name = "{}..//{}_{}_".format(result_baselines_folder_path, ALGORITHM_NAME, input_flags.dataset_name)

        KNN_similarity_to_report_list = ["cosine", "dice", "jaccard", "asymmetric", "tversky"]

        # Put results for the CNN algorithm in the baseline folder for it to be subsequently loaded
        dataIO = DataIO(folder_path = output_folder_path + "fit_ablation_all_map/all_map_0/" )
        search_metadata = dataIO.load_data(CoupledCF_RecommenderWrapper.RECOMMENDER_NAME + "_metadata")
        dataIO = DataIO(folder_path = result_baselines_folder_path)
        dataIO.save_data(CoupledCF_RecommenderWrapper.RECOMMENDER_NAME + "_metadata", search_metadata)


        result_loader = ResultFolderLoader(result_baselines_folder_path,
                                         base_algorithm_list = None,
                                         other_algorithm_list = [CoupledCF_RecommenderWrapper],
                                         KNN_similarity_list = KNN_similarity_to_report_list,
                                         ICM_names_list = None,
                                         UCM_names_list = None)


        result_loader.generate_latex_results(file_name + "{}_latex_results.txt".format("article_metrics"),
                                           metrics_list = ["HIT_RATE", "NDCG"],
                                           cutoffs_list = [1, 5, 10],
                                           table_title = None,
                                           highlight_best = True)

        result_loader.generate_latex_time_statistics(file_name + "{}_latex_results.txt".format("time"),
                                           n_evaluation_users=n_test_users,
                                           table_title = None)







