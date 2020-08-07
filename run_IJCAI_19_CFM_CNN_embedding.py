#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 20/06/2019

@author: Maurizio Ferrari Dacrema
"""

import os, argparse, traceback, shutil
import numpy as np
from Base.DataIO import DataIO
from CNN_on_embeddings.IJCAI.CFM_our_interface.FMWrapper import FM_Wrapper
from CNN_on_embeddings.IJCAI.CFM_our_interface.CFMWrapper import CFM_wrapper

from CNN_on_embeddings.IJCAI.CFM_our_interface.Dataset_wrapper import DatasetCFMReader
from Utils.assertions_on_data_for_experiments import assert_implicit_data, assert_disjoint_matrices
from Utils.ResultFolderLoader import ResultFolderLoader

from CNN_on_embeddings.run_CNN_embedding_evaluation_ablation import run_evaluation_ablation

from ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from ParameterTuning.SearchSingleCase import SearchSingleCase
from CNN_on_embeddings.read_CNN_embedding_evaluation_results import read_permutation_results


from functools import partial
import multiprocessing
from Recommender_import_list import *
from ParameterTuning.run_parameter_search import runParameterSearch_Collaborative


import tensorflow as tf

from CNN_on_embeddings.IJCAI.CFM_our_interface.CFM import _get_interaction_map

# prediction model
class CFM_assert:
    def __init__(self, n_factors, map_mode = "full_map"):

        self.map_mode = map_mode

        self.embedding_x = tf.placeholder(tf.float64, shape=[None, 1, n_factors], name='embedding_x')
        self.embedding_y = tf.placeholder(tf.float64, shape=[None, 1, n_factors], name='embedding_y')

        self.relation = tf.matmul(tf.transpose(self.embedding_x, perm=[0, 2, 1]), self.embedding_y)

        self.relation = _get_interaction_map(self.relation, map_mode)




def get_FM_hyperparameters_for_dataset(dataset_name):

    if dataset_name == 'lastfm':

        article_hyperparameters = {
        'pretrain_flag': -1,
        'hidden_factor': 64,
        'epochs': 500,
        'batch_size': 256,
        'learning_rate': 0.01,
        'lamda_bilinear': 0,
        'keep': 0.8,
        'optimizer_type': 'AdagradOptimizer',
        'batch_norm': 0,
        'permutation': None,
        'verbose': 0,
        'random_seed': 2016,
        'temp_file_folder': None,
        }

    else:
        raise ValueError("Invalid dataset name")

    return article_hyperparameters



def pretrain_FMwrapper(URM_train_tuning_only,
                       URM_train_full,
                       evaluator_validation,
                       evaluator_test,
                       CFM_data_class_validation,
                       CFM_data_class_full,
                       result_folder_path:str,
                       # hidden_factors:int,
                       metric_to_optimize:str,
                       dataset_name):


    # search best epoch
    article_hyperparameters = get_FM_hyperparameters_for_dataset(dataset_name)

    earlystopping_keywargs = {
        "validation_every_n": 5,
        "stop_on_validation": True,
        "lower_validations_allowed": 5,
        "evaluator_object": evaluator_validation,
        "validation_metric": metric_to_optimize
    }

    parameterSearch = SearchSingleCase(FM_Wrapper,
                                       evaluator_validation=evaluator_validation,
                                       evaluator_test=evaluator_test)

    recommender_input_args = SearchInputRecommenderArgs(CONSTRUCTOR_POSITIONAL_ARGS=[URM_train_tuning_only, CFM_data_class_validation],
                                                        FIT_KEYWORD_ARGS=earlystopping_keywargs)

    recommender_input_args_last_test = SearchInputRecommenderArgs(CONSTRUCTOR_POSITIONAL_ARGS=[URM_train_full, CFM_data_class_full])

    parameterSearch.search(recommender_input_args,
                           recommender_input_args_last_test=recommender_input_args_last_test,
                           fit_hyperparameters_values=article_hyperparameters,
                           output_folder_path=result_folder_path,
                           output_file_name_root=FM_Wrapper.RECOMMENDER_NAME,
                           save_model = "last",
                           resume_from_saved=True,
                           evaluate_on_test = "last")




def get_new_permutation(output_folder_path, permutation_index, embedding_size):

    # If directory does not exist, create
    if not os.path.exists(output_folder_path + "permutations/"):
        os.makedirs(output_folder_path + "permutations/")

    try:
        permutation = np.load(output_folder_path + "permutations/permutation_{}.npy".format(permutation_index))

    except FileNotFoundError:
        permutation = np.arange(embedding_size)
        np.random.shuffle(permutation)
        np.save(output_folder_path + "permutations/permutation_{}".format(permutation_index), permutation)

    assert not np.all(permutation == np.arange(embedding_size)), "Permutation not correct"

    return permutation





def run_permutation_pretrained_FM(URM_train_full, CFM_data_class_full, pretrained_model_folder_path, result_folder_path, permutation_index, permutation):

    result_folder_path_permutation = result_folder_path + "{}/{}_{}/".format("FM", "FM", permutation_index)

    # Read the pretraining data and put the permutation in it
    recommender_object = FM_Wrapper(URM_train_full, CFM_data_class_full)
    file_name_input = recommender_object.RECOMMENDER_NAME + "_best_model_last"
    file_name_output = recommender_object.RECOMMENDER_NAME

    if os.path.exists(result_folder_path_permutation + file_name_output + "_metadata.zip"):
        return


    result_folder_path_temp = result_folder_path_permutation + "__temp_model/"

    # If directory does not exist, create
    if not os.path.exists(result_folder_path_temp):
        os.makedirs(result_folder_path_temp)

    recommender_object.load_model(pretrained_model_folder_path, file_name_input)
    recommender_object.save_model(result_folder_path_temp, file_name_output)

    # Alter saved object to force in the desired permutation
    dataIO = DataIO(folder_path = result_folder_path_temp)
    data_dict = dataIO.load_data(file_name = file_name_output)

    data_dict["permutation"] = permutation
    dataIO.save_data(file_name = file_name_output, data_dict_to_save = data_dict)


    recommender_object = FM_Wrapper(URM_train_full, CFM_data_class_full)
    recommender_object.load_model(result_folder_path_temp,
                                  file_name=file_name_output)

    results_dict, results_run_string = evaluator_test.evaluateRecommender(recommender_object)

    shutil.rmtree(result_folder_path_temp, ignore_errors=True)

    result_file = open(result_folder_path_permutation + file_name_output + ".txt", "w")
    result_file.write(results_run_string)
    result_file.close()

    results_dict = {"result_on_last": results_dict}

    dataIO = DataIO(folder_path = result_folder_path_permutation)
    dataIO.save_data(file_name = file_name_output + "_metadata",
                     data_dict_to_save = results_dict)





def run_train_with_early_stopping(URM_train_tuning_only,
                               URM_train_full,
                               evaluator_validation,
                               evaluator_test,
                               CFM_data_class_validation,
                               CFM_data_class_full,
                               pretrained_FM_folder_path,
                               output_folder_path,
                               permutation_index,
                               map_mode,
                               metric_to_optimize):

    output_folder_path_permutation = output_folder_path + "fit_ablation_{}/{}_{}/".format(map_mode, map_mode, permutation_index)

    # If directory does not exist, create
    if not os.path.exists(output_folder_path_permutation):
        os.makedirs(output_folder_path_permutation)

    if os.path.isfile(output_folder_path_permutation + CFM_wrapper.RECOMMENDER_NAME + "_metadata.zip"):
        return

    article_hyperparameters = {
        'pretrain_flag': 1,
        'pretrained_FM_folder_path': pretrained_FM_folder_path,
        'hidden_factor': 64,
        'epochs': 300,
        'batch_size': 256,
        'learning_rate': 0.01,
        'lamda_bilinear': 0,
        'keep': 0.8,
        'optimizer_type': 'AdagradOptimizer',
        'batch_norm': 0,
        'verbose': False,
        'regs': '[10,1]',
        'attention_size': 32,
        'attentive_pooling': False,
        'net_channel': '[32,32,32,32,32,32]',
        'num_field': 4,
        'permutation': list(permutation),
        'map_mode': map_mode
    }



    earlystopping_hyperparameters = {
        "epochs_min": int(article_hyperparameters["epochs"]/2),
        "validation_every_n": 5,
        "stop_on_validation": True,
        "lower_validations_allowed": 5,
        "evaluator_object": evaluator_validation,
        "validation_metric": metric_to_optimize
    }

    # Due to the extremely long evaluation time it is computationally too expensive to run earlystopping on all
    # permutations (estimated >60 days on high end GPU)
    # So, select the epochs only at permutation 0 independently for each of the three modes: "all_map", "main_diagonal", "off_diagonal"

    # try to load selected number of epochs, if not present run earlystopping again
    folder_permutation_0 = output_folder_path + "fit_ablation_{}/{}_{}/".format(map_mode, map_mode, 0)

    if permutation_index == 0:

        parameterSearch = SearchSingleCase(CFM_wrapper,
                                           evaluator_validation=evaluator_validation,
                                           evaluator_test=evaluator_test)

        recommender_input_args = SearchInputRecommenderArgs(CONSTRUCTOR_POSITIONAL_ARGS=[URM_train_tuning_only, CFM_data_class_validation],
                                                            FIT_KEYWORD_ARGS=earlystopping_hyperparameters)

        recommender_input_args_last_test = SearchInputRecommenderArgs(CONSTRUCTOR_POSITIONAL_ARGS=[URM_train_full, CFM_data_class_full])

        parameterSearch.search(recommender_input_args,
                               recommender_input_args_last_test=recommender_input_args_last_test,
                               fit_hyperparameters_values=article_hyperparameters,
                               output_folder_path=output_folder_path_permutation,
                               output_file_name_root=CFM_wrapper.RECOMMENDER_NAME,
                               save_model = "last",
                               resume_from_saved=True,
                               evaluate_on_test = "last")



    else:


        dataIO = DataIO(folder_path = folder_permutation_0)
        data_dict = dataIO.load_data(file_name = CFM_wrapper.RECOMMENDER_NAME + "_metadata.zip")

        selected_epochs = data_dict["hyperparameters_best"]["epochs"]

        article_hyperparameters["epochs"] = selected_epochs

        parameterSearch = SearchSingleCase(CFM_wrapper,
                                           evaluator_validation=evaluator_test,
                                           evaluator_test=evaluator_test)

        recommender_input_args_last_test = SearchInputRecommenderArgs(CONSTRUCTOR_POSITIONAL_ARGS=[URM_train_full, CFM_data_class_full])

        parameterSearch.search(recommender_input_args_last_test,
                               recommender_input_args_last_test=None,
                               fit_hyperparameters_values=article_hyperparameters,
                               output_folder_path=output_folder_path_permutation,
                               output_file_name_root=CFM_wrapper.RECOMMENDER_NAME,
                               save_model = "best",
                               resume_from_saved=True,
                               evaluate_on_test = "best")



        # Get the data in the correct format to be readable for the data parsing script
        # Put the results in the "result_on_last" field of the metadata file
        # Change the final model file name into the _best_model_last suffix

        metadata_file_name = CFM_wrapper.RECOMMENDER_NAME + "_metadata.zip"

        dataIO = DataIO(folder_path = output_folder_path_permutation)
        search_metadata = dataIO.load_data(file_name = metadata_file_name)

        search_metadata["result_on_last"] = search_metadata["result_on_test_best"]
        dataIO.save_data(file_name = metadata_file_name, data_dict_to_save = search_metadata)

        recommender_object = CFM_wrapper(URM_train_full, CFM_data_class_full)
        recommender_object.load_model(output_folder_path_permutation,
                                      file_name=CFM_wrapper.RECOMMENDER_NAME + "_best_model")

        recommender_object.save_model(output_folder_path_permutation,
                                      file_name=CFM_wrapper.RECOMMENDER_NAME + "_best_model_last")










if __name__ == '__main__':

    ALGORITHM_NAME = "CFM"
    CONFERENCE_NAME = "IJCAI"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_name',         help = "Dataset name",      type = str, default = "lastfm")
    parser.add_argument('-p', '--run_fit_ablation',     help = "Run permutation and Study 1 experiments", type = bool, default = True)
    parser.add_argument('-a', '--run_eval_ablation',    help = "Run Study 2 experiments",  type = bool, default = True)
    parser.add_argument('-b', '--run_baselines',        help = "Run hyperparameter tuning", type = bool, default = True)
    parser.add_argument('-n', '--n_permutations',       help = "Number of permutations",    type = int, default = 20)
 
    input_flags = parser.parse_args()
    print(input_flags)



    output_folder_path = "result_experiments/{}/{}/".format(ALGORITHM_NAME, input_flags.dataset_name)

    dataset = DatasetCFMReader(output_folder_path + "data/", input_flags.dataset_name)

    print ('Current dataset is: {}'.format(input_flags.dataset_name))

    """
    WARNING: Due to how the data reader of the CFM original implementation works, the indices of the features
    associated to each user are not those contained in the file but are allocated, based on them, at runtime. 
    This means that changes in the ordering of the file will produce different, although equivalent, data structures.
    
    Because of this the data structures used during the validation phase are incompatible with those used during the test phase.
    Since the validation phase is only necessary for the selection of the epochs and no part of the model is used for the 
    subsequent testing phase, this inconsistency has no impact on the results since the model will always be 
    trained and evaluated on consistent data.
    """

    URM_train_tuning_only = dataset.URM_DICT["URM_train_tuning_only"].copy()
    URM_validation_tuning_only = dataset.URM_DICT["URM_validation_tuning_only"].copy()

    URM_train_full = dataset.URM_DICT["URM_train_full"].copy()
    URM_test_full = dataset.URM_DICT["URM_test_full"].copy()

    assert_implicit_data([URM_train_tuning_only, URM_train_full, URM_validation_tuning_only, URM_test_full])
    assert_disjoint_matrices([URM_train_tuning_only, URM_validation_tuning_only])
    assert_disjoint_matrices([URM_train_full, URM_test_full])


    from Base.Evaluation.Evaluator import EvaluatorHoldout

    cutoff_list_validation = [10]
    cutoff_list_test = [5, 10, 20]
    metric_to_optimize = "NDCG"

    evaluator_validation = EvaluatorHoldout(URM_validation_tuning_only, cutoff_list=cutoff_list_validation)
    evaluator_test = EvaluatorHoldout(URM_test_full, cutoff_list=cutoff_list_test)


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
        NMFRecommender,
        MatrixFactorization_BPR_Cython,
        MatrixFactorization_FunkSVD_Cython,
        IALSRecommender,
        SLIMElasticNetRecommender,
        SLIM_BPR_Cython,
        EASE_R_Recommender,
    ]

    n_cases = 50
    n_random_starts = 15
    result_baselines_folder_path = output_folder_path + "baselines/"

    hyperparameter_search_collaborative_partial = partial(runParameterSearch_Collaborative,
                                                          URM_train = URM_train_tuning_only,
                                                          URM_train_last_test = URM_train_full,
                                                          metric_to_optimize = metric_to_optimize,
                                                          evaluator_validation_earlystopping = evaluator_validation,
                                                          evaluator_validation = evaluator_validation,
                                                          evaluator_test = evaluator_test,
                                                          output_folder_path = result_baselines_folder_path,
                                                          parallelizeKNN = False,
                                                          allow_weighting = True,
                                                          n_cases = n_cases,
                                                          n_random_starts = n_random_starts,
                                                          resume_from_saved = True,
                                                          save_model = "last",
                                                          evaluate_on_test = "last")



    if input_flags.run_baselines:

        pool = multiprocessing.Pool(processes=3, maxtasksperchild=1)
        pool.map(hyperparameter_search_collaborative_partial, collaborative_algorithm_list)

        pool.close()
        pool.join()




        n_test_users = np.sum(np.ediff1d(URM_test_full.indptr)>=1)
        file_name = "{}..//{}_{}_".format(result_baselines_folder_path, ALGORITHM_NAME, input_flags.dataset_name)

        KNN_similarity_to_report_list = ["cosine", "dice", "jaccard", "asymmetric", "tversky"]

        result_loader = ResultFolderLoader(result_baselines_folder_path,
                                         base_algorithm_list = None,
                                         other_algorithm_list = [CFM_wrapper],
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




    ################################################################################################################################################
    ###############################
    ###############################         PRETRAINING FACTORIZATION MACHINE
    ###############################
    ################################################################################################################################################

    pretrain_folder_path = output_folder_path + "pretrained_model_data/".format(input_flags.dataset_name)

    pretrain_FMwrapper(URM_train_tuning_only = URM_train_tuning_only,
                       URM_train_full = URM_train_full,
                       evaluator_validation = evaluator_validation,
                       evaluator_test = evaluator_test,
                       CFM_data_class_validation = dataset.CFM_data_class_validation,
                       CFM_data_class_full = dataset.CFM_data_class_full,
                       result_folder_path = pretrain_folder_path,
                       metric_to_optimize = metric_to_optimize,
                       dataset_name = input_flags.dataset_name)


    ################################################################################################################################################
    ###############################
    ###############################         Test code on fake object to verify the alterations to the interaction map do what they are supposed to
    ###############################
    ################################################################################################################################################

    mu, sigma = 0, 0.1 # mean and standard deviation
    n_factors = 64

    embedding_X = np.random.normal(mu, sigma, (1, 1, n_factors))
    embedding_Y = np.random.normal(mu, sigma, (1, 1, n_factors))

    CFM_assert_object = CFM_assert(n_factors, map_mode ="full_map")

    with tf.Session() as session:
        result_all_map = session.run(CFM_assert_object.relation, feed_dict={
            CFM_assert_object.embedding_x: embedding_X,
            CFM_assert_object.embedding_y: embedding_Y
        })

    CFM_assert_object = CFM_assert(n_factors, map_mode ="main_diagonal")

    with tf.Session() as session:
        result_main_diag = session.run(CFM_assert_object.relation, feed_dict={
            CFM_assert_object.embedding_x: embedding_X,
            CFM_assert_object.embedding_y: embedding_Y
        })


    CFM_assert_object = CFM_assert(n_factors, map_mode ="off_diagonal")

    with tf.Session() as session:
        result_off_diag = session.run(CFM_assert_object.relation, feed_dict={
            CFM_assert_object.embedding_x: embedding_X,
            CFM_assert_object.embedding_y: embedding_Y
        })


    result_all_map = result_all_map.squeeze()
    result_main_diag = result_main_diag.squeeze()
    result_off_diag = result_off_diag.squeeze()


    assert np.allclose(result_main_diag.diagonal(), result_all_map.diagonal()), "two operations have different diagonal"
    assert np.allclose(result_main_diag, np.diag(result_main_diag.diagonal())), "result_main_diag has off diagonal elements"
    assert not np.allclose(result_all_map, np.diag(result_all_map.diagonal())), "result_all_map has NO off diagonal elements"

    assert np.allclose(result_all_map, result_main_diag + result_off_diag), "triangular composition non consistent"


    ###############################################################################################################################################
    ##############################
    ##############################         PERMUTATION EXPERIMENT
    ##############################
    ##############################         FIT ABLATION EXPERIMENT
    ##############################
    ###############################################################################################################################################

    article_hyperparameters = get_FM_hyperparameters_for_dataset(input_flags.dataset_name)
    embedding_size = article_hyperparameters["hidden_factor"]


    if input_flags.run_fit_ablation:

        for permutation_index in range(input_flags.n_permutations):

            try:

                # Get new permutation array
                permutation = get_new_permutation(output_folder_path, permutation_index, embedding_size)

                ## Evaluate permutated pretraining model
                run_permutation_pretrained_FM(URM_train_full = URM_train_full,
                                              CFM_data_class_full = dataset.CFM_data_class_full,
                                              pretrained_model_folder_path = pretrain_folder_path,
                                              result_folder_path = output_folder_path,
                                              permutation_index = permutation_index,
                                              permutation = permutation)

                ### Fit model with the different interaction map modes
                for map_mode in ["all_map", "main_diagonal", "off_diagonal"]:

                    run_train_with_early_stopping(URM_train_tuning_only = URM_train_tuning_only,
                                                   URM_train_full = URM_train_full,
                                                   evaluator_validation = evaluator_validation,
                                                   evaluator_test = evaluator_test,
                                                   CFM_data_class_validation = dataset.CFM_data_class_validation,
                                                   CFM_data_class_full = dataset.CFM_data_class_full,
                                                   pretrained_FM_folder_path = pretrain_folder_path + FM_Wrapper.RECOMMENDER_NAME + "_best_model_last",
                                                   output_folder_path = output_folder_path,
                                                   permutation_index = permutation_index,
                                                   map_mode = map_mode,
                                                   metric_to_optimize = metric_to_optimize)

            except:
                traceback.print_exc()

        read_permutation_results(output_folder_path, input_flags.n_permutations, 10,
                                 ["NDCG", "HIT_RATE"],
                                 file_result_name_root = "latex_fit_ablation_results",
                                 convolution_model_name = CFM_wrapper.RECOMMENDER_NAME,
                                 pretrained_model_name = 'FM',
                                 pretrained_model_class = FM_Wrapper,
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

                recommender_input_args_last_test = SearchInputRecommenderArgs(CONSTRUCTOR_POSITIONAL_ARGS=[URM_train_full, dataset.CFM_data_class_full])

                run_evaluation_ablation(recommender_class = CFM_wrapper,
                                        recommender_input_args = recommender_input_args_last_test,
                                        evaluator_test = evaluator_test,
                                        input_folder_path = input_folder_path,
                                        result_folder_path = result_folder_path,
                                        map_mode = map_mode)


        read_permutation_results(output_folder_path, input_flags.n_permutations, 10,
                                 ["NDCG", "HIT_RATE"],
                                 file_result_name_root = "latex_evaluation_ablation_results",
                                 convolution_model_name = CFM_wrapper.RECOMMENDER_NAME,
                                 pretrained_model_name = 'FM',
                                 pretrained_model_class = FM_Wrapper,
                                 experiment_type = "evaluation_ablation")


