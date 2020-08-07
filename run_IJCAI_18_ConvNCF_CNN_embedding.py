#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 20/06/2019

@author: Maurizio Ferrari Dacrema
"""

import os, argparse, traceback
import numpy as np
from Base.DataIO import DataIO
from Base.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender

from Conferences.IJCAI.ConvNCF_our_interface.GowallaReader.GowallaReader import GowallaReader
from Conferences.IJCAI.ConvNCF_our_interface.YelpReader.YelpReader import YelpReader
from CNN_on_embeddings.IJCAI.ConvNCF_our_interface.ConvNCF_wrapper import ConvNCF_RecommenderWrapper
from Conferences.IJCAI.ConvNCF_our_interface.MFBPR_Wrapper import MFBPR_Wrapper

from Utils.ResultFolderLoader import ResultFolderLoader
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from ParameterTuning.SearchSingleCase import SearchSingleCase
from CNN_on_embeddings.run_CNN_embedding_evaluation_ablation import run_evaluation_ablation

from Utils.assertions_on_data_for_experiments import assert_implicit_data, assert_disjoint_matrices
from CNN_on_embeddings.read_CNN_embedding_evaluation_results import read_permutation_results

import tensorflow as tf

from functools import partial
import multiprocessing
from Recommender_import_list import *
from ParameterTuning.run_parameter_search import runParameterSearch_Collaborative

class MatrixFactorizationCustomFactorsRecommender(BaseMatrixFactorizationRecommender):
    """ BaseMatrixFactorizationRecommender"""

    RECOMMENDER_NAME = "BaseMatrixFactorizationRecommender"

    def fit(self, USER_factors = None, ITEM_factors = None):

        assert USER_factors is not None and ITEM_factors is not None

        self.USER_factors = USER_factors.copy()
        self.ITEM_factors = ITEM_factors.copy()



# prediction model
class ConvNCF_assert:
    def __init__(self, n_factors, map_mode = "full_map"):

        map_mode_flag_main_diagonal = map_mode == "main_diagonal"
        map_mode_flag_off_diagonal = map_mode == "off_diagonal"

        self.embedding_p = tf.placeholder(tf.float64, shape=[None, 1, n_factors], name='embedding_p')
        self.embedding_q = tf.placeholder(tf.float64, shape=[None, 1, n_factors], name='embedding_q')

        # outer product of P_u and Q_i
        self.relation = tf.matmul(tf.transpose(self.embedding_p, perm=[0, 2, 1]), self.embedding_q)

        # If using only the diagonal, remove everything not in the diagonal
        if map_mode_flag_main_diagonal:
            relation_main_diagonal = tf.zeros_like(self.relation)
            relation_main_diagonal = tf.linalg.set_diag(relation_main_diagonal, tf.linalg.diag_part(self.relation))

            self.relation = relation_main_diagonal

        elif map_mode_flag_off_diagonal:
            self.relation = tf.linalg.set_diag(self.relation, tf.zeros_like(tf.linalg.diag_part(self.relation)))



def pretrain_MFBPR(URM_train,
                   URM_train_full,
                   evaluator_validation,
                   evaluator_test,
                   result_folder_path,
                   metric_to_optimize,
                   ):

    article_hyperparameters = {
        "batch_size": 512,
        "epochs": 500,
        "embed_size":64,
        "negative_sample_per_positive":1,
        "learning_rate":0.05,
        "path_partial_results":result_folder_path,
        }


    earlystopping_keywargs = {
        "validation_every_n": 5,
        "stop_on_validation": True,
        "lower_validations_allowed": 5,
        "evaluator_object": evaluator_validation,
        "validation_metric": metric_to_optimize
    }

    parameterSearch = SearchSingleCase(MFBPR_Wrapper,
                                       evaluator_validation=evaluator_validation,
                                       evaluator_test=evaluator_test)

    recommender_input_args = SearchInputRecommenderArgs(CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],
                                                        FIT_KEYWORD_ARGS=earlystopping_keywargs)

    recommender_input_args_last_test = SearchInputRecommenderArgs(CONSTRUCTOR_POSITIONAL_ARGS=[URM_train_full])

    parameterSearch.search(recommender_input_args,
                           recommender_input_args_last_test=recommender_input_args_last_test,
                           fit_hyperparameters_values=article_hyperparameters,
                           output_folder_path=result_folder_path,
                           output_file_name_root=MFBPR_Wrapper.RECOMMENDER_NAME,
                           save_model = "last",
                           resume_from_saved=True,
                           evaluate_on_test = "last")





def run_train_with_early_stopping(output_folder_path, permutation_index,
                                  USER_factors_perm, ITEM_factors_perm,
                                  map_mode, metric_to_optimize,
                                  evaluator_validation, evaluator_test,
                                  URM_train, URM_validation):

    output_folder_path_permutation = output_folder_path + "fit_ablation_{}/{}_{}/".format(map_mode, map_mode, permutation_index)

    # If directory does not exist, create
    if not os.path.exists(output_folder_path_permutation):
        os.makedirs(output_folder_path_permutation)


    assert USER_factors_perm.shape == (n_users, n_factors)
    assert ITEM_factors_perm.shape == (n_items, n_factors)

    np.save(output_folder_path_permutation + "best_model_latent_factors", [USER_factors_perm, ITEM_factors_perm])

    optimal_hyperparameters = {
        "batch_size": 512,
        "epochs": 1500,
        "load_pretrained_MFBPR_if_available": True,
        "MF_latent_factors_folder": output_folder_path_permutation,
        "embedding_size": 64,
        "hidden_size": 128,
        "negative_sample_per_positive": 1,
        "negative_instances_per_positive": 4,
        "regularization_users_items": 0.01,
        "regularization_weights": 10,
        "regularization_filter_weights": 1,
        "learning_rate_embeddings": 0.05,
        "learning_rate_CNN": 0.05,
        "channel_size": [32, 32, 32, 32, 32, 32],
        "dropout": 0.0,
        "epoch_verbose": 1,
        "temp_file_folder": None,
        }


    optimal_hyperparameters["map_mode"] = map_mode



    earlystopping_hyperparameters = {
        "validation_every_n": 5,
        "stop_on_validation": True,
        "lower_validations_allowed": 5,
        "evaluator_object": evaluator_validation,
        "validation_metric": metric_to_optimize
    }

    parameterSearch = SearchSingleCase(ConvNCF_RecommenderWrapper,
                                       evaluator_validation=evaluator_validation,
                                       evaluator_test=evaluator_test)

    recommender_input_args = SearchInputRecommenderArgs(CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],
                                                        FIT_KEYWORD_ARGS=earlystopping_hyperparameters)

    recommender_input_args_last_test = recommender_input_args.copy()
    recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train + URM_validation

    parameterSearch.search(recommender_input_args,
                            recommender_input_args_last_test=recommender_input_args_last_test,
                            fit_hyperparameters_values=optimal_hyperparameters,
                            output_folder_path=output_folder_path_permutation,
                            output_file_name_root=ConvNCF_RecommenderWrapper.RECOMMENDER_NAME,
                            save_model = "last",
                            resume_from_saved=True,
                            evaluate_on_test = "last")



def run_permutation_BPRMF(output_folder_path, permutation_index, USER_factors_perm, ITEM_factors_perm):

    output_folder_path_permutation = output_folder_path + "{}/{}_{}/".format("BPRMF", "BPRMF", permutation_index)

    # If directory does not exist, create
    if not os.path.exists(output_folder_path_permutation):
        os.makedirs(output_folder_path_permutation)

    assert USER_factors_perm.shape == (n_users, n_factors)
    assert ITEM_factors_perm.shape == (n_items, n_factors)


    parameterSearch = SearchSingleCase(MatrixFactorizationCustomFactorsRecommender,
                                       evaluator_validation = None,
                                       evaluator_test = evaluator_test)

    recommender_input_args = SearchInputRecommenderArgs(
                                        CONSTRUCTOR_POSITIONAL_ARGS = [URM_train + URM_validation],
                                        FIT_KEYWORD_ARGS = {
                                            "USER_factors": USER_factors,
                                            "ITEM_factors": ITEM_factors
                                        })

    parameterSearch.search(recommender_input_args,
                           save_model = "no",
                           resume_from_saved=True,
                           fit_hyperparameters_values = {},
                           output_folder_path = output_folder_path_permutation,
                           output_file_name_root = MatrixFactorizationCustomFactorsRecommender.RECOMMENDER_NAME)





def shuffle_matrix(factor_matrix_input):

    factor_matrix = factor_matrix_input.copy()

    n_rows, n_factors = factor_matrix.shape

    data_points_array = np.reshape(factor_matrix, (1,-1)).ravel()

    np.random.shuffle(data_points_array)

    factor_matrix = np.reshape(data_points_array, (n_rows, n_factors))

    assert not np.all(np.equal(factor_matrix, factor_matrix_input))

    return factor_matrix





def get_new_permutation(output_folder_path, permutation_index, USER_factors, ITEM_factors):

    # If directory does not exist, create
    if not os.path.exists(output_folder_path + "permutations/"):
        os.makedirs(output_folder_path + "permutations/")

    try:
        permutation = np.load(output_folder_path + "permutations/permutation_{}.npy".format(permutation_index))

    except FileNotFoundError:
        n_factors = USER_factors.shape[1]
        permutation = np.arange(n_factors)
        np.random.shuffle(permutation)
        np.save(output_folder_path + "permutations/permutation_{}".format(permutation_index), permutation)


    USER_factors_perm = USER_factors[:,permutation]
    ITEM_factors_perm = ITEM_factors[:,permutation]

    assert not np.all(np.equal(USER_factors_perm, USER_factors))
    assert not np.all(np.equal(ITEM_factors_perm, ITEM_factors))

    return USER_factors_perm, ITEM_factors_perm





if __name__ == '__main__':

    ALGORITHM_NAME = "ConvNCF"
    CONFERENCE_NAME = "IJCAI"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_name',         help = "Dataset name", type = str, default = "yelp")
    parser.add_argument('-p', '--run_fit_ablation',     help = "Run permutation and Study 1 experiments", type = bool, default = True)
    parser.add_argument('-a', '--run_eval_ablation',    help = "Run Study 2 experiments", type = bool, default = True)
    parser.add_argument('-b', '--run_baselines',        help = "Run hyperparameter tuning", type = bool, default = True)
    parser.add_argument('-n', '--n_permutations',       help = "Number of permutations", type = int, default = 20)
 
    input_flags = parser.parse_args()
    print(input_flags)



    output_folder_path = "result_experiments/{}/{}/".format(ALGORITHM_NAME, input_flags.dataset_name)

    if input_flags.dataset_name == "gowalla":
        dataset = GowallaReader(output_folder_path + "data/")

    elif input_flags.dataset_name == "yelp":
        dataset = YelpReader(output_folder_path + "data/")



    print ('Current dataset is: {}'.format(input_flags.dataset_name))


    URM_train = dataset.URM_DICT["URM_train"].copy()
    URM_validation = dataset.URM_DICT["URM_validation"].copy()
    URM_test = dataset.URM_DICT["URM_test"].copy()
    URM_test_negative = dataset.URM_DICT["URM_test_negative"].copy()

    assert_implicit_data([URM_train, URM_validation, URM_test, URM_test_negative])
    assert_disjoint_matrices([URM_train, URM_validation, URM_test])

    from Base.Evaluation.Evaluator import EvaluatorNegativeItemSample

    cutoff_list_validation = [10]
    cutoff_list_test = [5, 10, 20]
    metric_to_optimize = "NDCG"

    evaluator_validation = EvaluatorNegativeItemSample(URM_validation, URM_test_negative, cutoff_list=cutoff_list_validation)
    evaluator_test = EvaluatorNegativeItemSample(URM_test, URM_test_negative, cutoff_list=cutoff_list_test)


    ################################################################################################################################################
    ###############################
    ###############################         PRETRAINING MFBPR
    ###############################
    ################################################################################################################################################

    pretrain_folder_path = output_folder_path + "pretrained_model_data/".format(input_flags.dataset_name)

    pretrain_MFBPR(URM_train = URM_train,
                   URM_train_full = URM_train + URM_validation,
                   evaluator_validation = evaluator_validation,
                   evaluator_test = evaluator_test,
                   result_folder_path = pretrain_folder_path,
                   metric_to_optimize = metric_to_optimize)


    # Load latent factors
    latent_factors = np.load(pretrain_folder_path + "_latent_factors.npy", allow_pickle=True)

    USER_factors, ITEM_factors = latent_factors[0], latent_factors[1]

    n_users, n_items = URM_train.shape

    assert USER_factors.shape[0] == n_users
    assert ITEM_factors.shape == (n_items, USER_factors.shape[1])


    #####################################################
    #### Test code on fake object to verify the alterations to the interaction map do what they are supposed to


    mu, sigma = 0, 0.1 # mean and standard deviation

    USER_factors_random = np.random.normal(mu, sigma, (1, 1, USER_factors.shape[1]))
    ITEM_factors_random = np.random.normal(mu, sigma, (1, 1, ITEM_factors.shape[1]))

    n_factors = USER_factors.shape[1]

    ConvNCF_assert_object = ConvNCF_assert(n_factors, map_mode = "full_map")

    with tf.Session() as session:
        result_all_map = session.run(ConvNCF_assert_object.relation, feed_dict={
            ConvNCF_assert_object.embedding_p: USER_factors_random,
            ConvNCF_assert_object.embedding_q: ITEM_factors_random
        })

    ConvNCF_assert_object = ConvNCF_assert(n_factors, map_mode = "main_diagonal")

    with tf.Session() as session:
        result_main_diag = session.run(ConvNCF_assert_object.relation, feed_dict={
            ConvNCF_assert_object.embedding_p: USER_factors_random,
            ConvNCF_assert_object.embedding_q: ITEM_factors_random
        })


    ConvNCF_assert_object = ConvNCF_assert(n_factors, map_mode = "off_diagonal")

    with tf.Session() as session:
        result_off_diag = session.run(ConvNCF_assert_object.relation, feed_dict={
            ConvNCF_assert_object.embedding_p: USER_factors_random,
            ConvNCF_assert_object.embedding_q: ITEM_factors_random
        })

    
    result_all_map = result_all_map.squeeze()
    result_main_diag = result_main_diag.squeeze()
    result_off_diag = result_off_diag.squeeze()
    

    assert np.allclose(result_main_diag.diagonal(), result_all_map.diagonal()), "two operations have different diagonal"
    assert np.allclose(result_main_diag, np.diag(result_main_diag.diagonal())), "result_main_diag has off diagonal elements"
    assert not np.allclose(result_all_map, np.diag(result_all_map.diagonal())), "result_all_map has NO off diagonal elements"

    assert np.allclose(result_all_map, result_main_diag + result_off_diag), "triangular composition non consistent"


    ################################################################################################################################################
    ###############################
    ###############################         PERMUTATION EXPERIMENT
    ###############################
    ###############################         FIT ABLATION EXPERIMENT
    ###############################
    ################################################################################################################################################

    if input_flags.run_fit_ablation:

        for permutation_index in range(input_flags.n_permutations):
            
            try:

                USER_factors_perm, ITEM_factors_perm = get_new_permutation(output_folder_path, permutation_index, USER_factors, ITEM_factors)

                ## Evaluate permutated pretraining model
                run_permutation_BPRMF(output_folder_path, permutation_index, USER_factors_perm, ITEM_factors_perm)

                ### Fit model with the different interaction map modes
                for map_mode in ["all_map", "main_diagonal", "off_diagonal"]:

                    run_train_with_early_stopping(output_folder_path = output_folder_path,
                                                  permutation_index = permutation_index,
                                                  USER_factors_perm = USER_factors_perm,
                                                  ITEM_factors_perm = ITEM_factors_perm,
                                                  map_mode = map_mode,
                                                  metric_to_optimize = metric_to_optimize,
                                                  evaluator_validation = evaluator_validation,
                                                  evaluator_test = evaluator_test,
                                                  URM_train = URM_train,
                                                  URM_validation = URM_validation)

            except:
                traceback.print_exc()

        read_permutation_results(output_folder_path, input_flags.n_permutations, 10,
                                 ["PRECISION", "MAP", "NDCG", "F1", "HIT_RATE"],
                                 file_result_name_root = "latex_fit_ablation_results",
                                 convolution_model_name = ConvNCF_RecommenderWrapper.RECOMMENDER_NAME,
                                 pretrained_model_name = 'BPRMF',
                                 pretrained_model_class = MatrixFactorizationCustomFactorsRecommender,
                                 experiment_type = "fit_ablation")


    ################################################################################################################################################
    ###############################
    ###############################         EVALUATION ABLATION EXPERIMENT
    ###############################
    ################################################################################################################################################

    if input_flags.run_eval_ablation:

        for permutation_index in range(input_flags.n_permutations):

            # Run evaluation of the full map fitted model with the different interaction map modes
            for map_mode in ["all_map", "main_diagonal", "off_diagonal"]:

                input_folder_path = os.path.join(output_folder_path, "fit_ablation_{}/{}_{}/".format("all_map", "all_map", permutation_index))
                result_folder_path = os.path.join(output_folder_path, "evaluation_ablation_{}/{}_{}/".format(map_mode, map_mode, permutation_index))

                recommender_input_args = SearchInputRecommenderArgs(CONSTRUCTOR_POSITIONAL_ARGS=[URM_train])

                run_evaluation_ablation(recommender_class=ConvNCF_RecommenderWrapper,
                                        recommender_input_args = recommender_input_args,
                                        evaluator_test = evaluator_test,
                                        input_folder_path = input_folder_path,
                                        result_folder_path = result_folder_path,
                                        map_mode = map_mode)


        read_permutation_results(output_folder_path, input_flags.n_permutations, 10,
                                 ["PRECISION", "MAP", "NDCG", "F1", "HIT_RATE"],
                                 file_result_name_root = "latex_evaluation_ablation_results",
                                 convolution_model_name = ConvNCF_RecommenderWrapper.RECOMMENDER_NAME,
                                 pretrained_model_name = 'BPRMF',
                                 pretrained_model_class = MatrixFactorizationCustomFactorsRecommender,
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
        search_metadata = dataIO.load_data(ConvNCF_RecommenderWrapper.RECOMMENDER_NAME + "_metadata")
        dataIO = DataIO(folder_path = result_baselines_folder_path)
        dataIO.save_data(ConvNCF_RecommenderWrapper.RECOMMENDER_NAME + "_metadata", search_metadata)


        result_loader = ResultFolderLoader(result_baselines_folder_path,
                                         base_algorithm_list = None,
                                         other_algorithm_list = [ConvNCF_RecommenderWrapper],
                                         KNN_similarity_list = KNN_similarity_to_report_list,
                                         ICM_names_list = None,
                                         UCM_names_list = None)


        result_loader.generate_latex_results(file_name + "{}_latex_results.txt".format("article_metrics"),
                                           metrics_list = ["HIT_RATE", "NDCG"],
                                           cutoffs_list = cutoff_list_test,
                                           table_title = None,
                                           highlight_best = True)

        result_loader.generate_latex_time_statistics(file_name + "{}_latex_results.txt".format("time"),
                                           n_evaluation_users=n_test_users,
                                           table_title = None)

