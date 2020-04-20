#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 10/03/2018

@author: Maurizio Ferrari Dacrema
"""

import time, os, traceback
from Base.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
import numpy as np
from Base.DataIO import DataIO

class SearchInputRecommenderArgs(object):


    def __init__(self,
                   # Dictionary of parameters needed by the constructor
                   CONSTRUCTOR_POSITIONAL_ARGS = None,
                   CONSTRUCTOR_KEYWORD_ARGS = None,

                   # List containing all positional arguments needed by the fit function
                   FIT_POSITIONAL_ARGS = None,
                   FIT_KEYWORD_ARGS = None
                   ):


          super(SearchInputRecommenderArgs, self).__init__()

          if CONSTRUCTOR_POSITIONAL_ARGS is None:
              CONSTRUCTOR_POSITIONAL_ARGS = []

          if CONSTRUCTOR_KEYWORD_ARGS is None:
              CONSTRUCTOR_KEYWORD_ARGS = {}

          if FIT_POSITIONAL_ARGS is None:
              FIT_POSITIONAL_ARGS = []

          if FIT_KEYWORD_ARGS is None:
              FIT_KEYWORD_ARGS = {}


          assert isinstance(CONSTRUCTOR_POSITIONAL_ARGS, list), "CONSTRUCTOR_POSITIONAL_ARGS must be a list"
          assert isinstance(CONSTRUCTOR_KEYWORD_ARGS, dict), "CONSTRUCTOR_KEYWORD_ARGS must be a dict"

          assert isinstance(FIT_POSITIONAL_ARGS, list), "FIT_POSITIONAL_ARGS must be a list"
          assert isinstance(FIT_KEYWORD_ARGS, dict), "FIT_KEYWORD_ARGS must be a dict"


          self.CONSTRUCTOR_POSITIONAL_ARGS = CONSTRUCTOR_POSITIONAL_ARGS
          self.CONSTRUCTOR_KEYWORD_ARGS = CONSTRUCTOR_KEYWORD_ARGS

          self.FIT_POSITIONAL_ARGS = FIT_POSITIONAL_ARGS
          self.FIT_KEYWORD_ARGS = FIT_KEYWORD_ARGS





    def copy(self):


        clone_object = SearchInputRecommenderArgs(
                            CONSTRUCTOR_POSITIONAL_ARGS = self.CONSTRUCTOR_POSITIONAL_ARGS.copy(),
                            CONSTRUCTOR_KEYWORD_ARGS = self.CONSTRUCTOR_KEYWORD_ARGS.copy(),
                            FIT_POSITIONAL_ARGS = self.FIT_POSITIONAL_ARGS.copy(),
                            FIT_KEYWORD_ARGS = self.FIT_KEYWORD_ARGS.copy()
                            )


        return clone_object








def _compute_avg_time_non_none_values(data_list):

    non_none_values = sum([value is not None for value in data_list])
    total_value = sum([value if value is not None else 0.0 for value in data_list])

    return total_value, \
           total_value/non_none_values if non_none_values != 0 else 0.0



def get_result_string_evaluate_on_validation(results_run_single_cutoff, n_decimals=7):

    output_str = ""

    for metric in results_run_single_cutoff.keys():
        output_str += "{}: {:.{n_decimals}f}, ".format(metric, results_run_single_cutoff[metric], n_decimals = n_decimals)

    return output_str



class SearchAbstractClass(object):

    ALGORITHM_NAME = "SearchAbstractClass"

    # Available values for the save_model attribute
    _SAVE_MODEL_VALUES = ["all", "best", "last", "no"]


    # Value to be assigned to invalid configuration or if an Exception is raised
    INVALID_CONFIG_VALUE = np.finfo(np.float16).max

    def __init__(self, recommender_class,
                 evaluator_validation = None,
                 evaluator_test = None,
                 verbose = True):

        super(SearchAbstractClass, self).__init__()

        self.recommender_class = recommender_class
        self.verbose = verbose
        self.log_file = None

        self.results_test_best = {}
        self.parameter_dictionary_best = {}

        self.evaluator_validation = evaluator_validation

        if evaluator_test is None:
            self.evaluator_test = None
        else:
            self.evaluator_test = evaluator_test


    def search(self, recommender_input_args,
               parameter_search_space,
               metric_to_optimize = "MAP",
               n_cases = None,
               output_folder_path = None,
               output_file_name_root = None,
               parallelize = False,
               save_model = "best",
               evaluate_on_test_each_best_solution = True,
               save_metadata = True,
               ):

        raise NotImplementedError("Function search not implemented for this class")


    def _set_search_attributes(self, recommender_input_args,
                               recommender_input_args_last_test,
                               metric_to_optimize,
                               output_folder_path,
                               output_file_name_root,
                               resume_from_saved,
                               save_metadata,
                               save_model,
                               evaluate_on_test_each_best_solution,
                               n_cases):


        if save_model not in self._SAVE_MODEL_VALUES:
           raise ValueError("{}: parameter save_model must be in '{}', provided was '{}'.".format(self.ALGORITHM_NAME, self._SAVE_MODEL_VALUES, save_model))

        self.output_folder_path = output_folder_path
        self.output_file_name_root = output_file_name_root

        # If directory does not exist, create
        if not os.path.exists(self.output_folder_path):
            os.makedirs(self.output_folder_path)

        self.log_file = open(self.output_folder_path + self.output_file_name_root + "_{}.txt".format(self.ALGORITHM_NAME), "a")

        if save_model == "last" and recommender_input_args_last_test is None:
            self._write_log("{}: parameter save_model is 'last' but no recommender_input_args_last_test provided, saving best model on train data alone.".format(self.ALGORITHM_NAME))
            save_model = "best"



        self.recommender_input_args = recommender_input_args
        self.recommender_input_args_last_test = recommender_input_args_last_test
        self.metric_to_optimize = metric_to_optimize
        self.save_model = save_model
        self.resume_from_saved = resume_from_saved
        self.save_metadata = save_metadata
        self.evaluate_on_test_each_best_solution = evaluate_on_test_each_best_solution

        self.model_counter = 0
        self._init_metadata_dict(n_cases = n_cases)

        if self.save_metadata:
            self.dataIO = DataIO(folder_path = self.output_folder_path)



    def _init_metadata_dict(self, n_cases):

        self.metadata_dict = {"algorithm_name_search": self.ALGORITHM_NAME,
                              "algorithm_name_recommender": self.recommender_class.RECOMMENDER_NAME,
                              "exception_list": [None]*n_cases,

                              "hyperparameters_list": [None]*n_cases,
                              "hyperparameters_best": None,
                              "hyperparameters_best_index": None,

                              "result_on_validation_list": [None]*n_cases,
                              "result_on_validation_best": None,
                              "result_on_test_list": [None]*n_cases,
                              "result_on_test_best": None,

                              "time_on_train_list": [None]*n_cases,
                              "time_on_train_total": 0.0,
                              "time_on_train_avg": 0.0,

                              "time_on_validation_list": [None]*n_cases,
                              "time_on_validation_total": 0.0,
                              "time_on_validation_avg": 0.0,

                              "time_on_test_list": [None]*n_cases,
                              "time_on_test_total": 0.0,
                              "time_on_test_avg": 0.0,

                              "result_on_last": None,
                              "time_on_last_train": None,
                              "time_on_last_test": None,
                              }


    def _print(self, string):

        if self.verbose:
            print(string)


    def _write_log(self, string):

        self._print(string)

        if self.log_file is not None:
            self.log_file.write(string)
            self.log_file.flush()


    def _fit_model(self, current_fit_parameters):

        start_time = time.time()

        # Construct a new recommender instance
        recommender_instance = self.recommender_class(*self.recommender_input_args.CONSTRUCTOR_POSITIONAL_ARGS,
                                                      **self.recommender_input_args.CONSTRUCTOR_KEYWORD_ARGS)


        self._print("{}: Testing config: {}".format(self.ALGORITHM_NAME, current_fit_parameters))


        recommender_instance.fit(*self.recommender_input_args.FIT_POSITIONAL_ARGS,
                                 **self.recommender_input_args.FIT_KEYWORD_ARGS,
                                 **current_fit_parameters)

        train_time = time.time() - start_time

        return recommender_instance, train_time



    def _evaluate_on_validation(self, current_fit_parameters):

        recommender_instance, train_time = self._fit_model(current_fit_parameters)

        start_time = time.time()

        # Evaluate recommender and get results for the first cutoff
        result_dict, _ = self.evaluator_validation.evaluateRecommender(recommender_instance)
        result_dict = result_dict[list(result_dict.keys())[0]]

        evaluation_time = time.time() - start_time

        result_string = get_result_string_evaluate_on_validation(result_dict, n_decimals=7)

        return result_dict, result_string, recommender_instance, train_time, evaluation_time



    def _evaluate_on_test(self, recommender_instance, current_fit_parameters_dict, print_log = True):

        start_time = time.time()

        # Evaluate recommender and get results for the first cutoff
        result_dict, result_string = self.evaluator_test.evaluateRecommender(recommender_instance)

        evaluation_test_time = time.time() - start_time

        if print_log:
            self._write_log("{}: Best config evaluated with evaluator_test. Config: {} - results:\n{}\n".format(self.ALGORITHM_NAME,
                                                                                                         current_fit_parameters_dict,
                                                                                                         result_string))

        return result_dict, result_string, evaluation_test_time



    def _evaluate_on_test_with_data_last(self):

        start_time = time.time()

        # Construct a new recommender instance
        recommender_instance = self.recommender_class(*self.recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS,
                                                      **self.recommender_input_args_last_test.CONSTRUCTOR_KEYWORD_ARGS)

        # Check if last was already evaluated
        if self.resume_from_saved:
            result_on_last_saved_flag = self.metadata_dict["result_on_last"] is not None and \
                                        self.metadata_dict["time_on_last_train"] is not None and \
                                        self.metadata_dict["time_on_last_test"] is not None

            if result_on_last_saved_flag:
                self._print("{}: Resuming '{}'... Result on last already available.".format(self.ALGORITHM_NAME, self.output_file_name_root))
                return



        self._print("{}: Evaluation with constructor data for final test. Using best config: {}".format(self.ALGORITHM_NAME, self.metadata_dict["hyperparameters_best"]))


        # Use the hyperparameters that have been saved
        assert self.metadata_dict["hyperparameters_best"] is not None, "{}: Best hyperparameters not available, the search might have failed.".format(self.ALGORITHM_NAME)
        fit_keyword_args = self.metadata_dict["hyperparameters_best"].copy()


        recommender_instance.fit(*self.recommender_input_args_last_test.FIT_POSITIONAL_ARGS,
                                 **fit_keyword_args)

        train_time = time.time() - start_time

        result_dict_test, result_string, evaluation_test_time = self._evaluate_on_test(recommender_instance, fit_keyword_args, print_log = False)

        self._write_log("{}: Best config evaluated with evaluator_test with constructor data for final test. Config: {} - results:\n{}\n".format(self.ALGORITHM_NAME,
                                                                                                                                          self.metadata_dict["hyperparameters_best"],
                                                                                                                                          result_string))

        self.metadata_dict["result_on_last"] = result_dict_test
        self.metadata_dict["time_on_last_train"] = train_time
        self.metadata_dict["time_on_last_test"] = evaluation_test_time

        if self.save_metadata:
            self.dataIO.save_data(data_dict_to_save = self.metadata_dict.copy(),
                                  file_name = self.output_file_name_root + "_metadata")

        if self.save_model in ["all", "best", "last"]:
            self._print("{}: Saving model in {}\n".format(self.ALGORITHM_NAME, self.output_folder_path + self.output_file_name_root))
            recommender_instance.save_model(self.output_folder_path, file_name =self.output_file_name_root + "_best_model_last")





    def _objective_function(self, current_fit_parameters_dict):

        try:

            self.metadata_dict["hyperparameters_list"][self.model_counter] = current_fit_parameters_dict.copy()

            result_dict, result_string, recommender_instance, train_time, evaluation_time = self._evaluate_on_validation(current_fit_parameters_dict)

            current_result = - result_dict[self.metric_to_optimize]

            # If the recommender uses Earlystopping, get the selected number of epochs
            if isinstance(recommender_instance, Incremental_Training_Early_Stopping):

                n_epochs_early_stopping_dict = recommender_instance.get_early_stopping_final_epochs_dict()
                current_fit_parameters_dict = current_fit_parameters_dict.copy()

                for epoch_label in n_epochs_early_stopping_dict.keys():

                    epoch_value = n_epochs_early_stopping_dict[epoch_label]
                    current_fit_parameters_dict[epoch_label] = epoch_value



            # Always save best model separately
            if self.save_model in ["all"]:
                self._print("{}: Saving model in {}\n".format(self.ALGORITHM_NAME, self.output_folder_path + self.output_file_name_root))
                recommender_instance.save_model(self.output_folder_path, file_name = self.output_file_name_root + "_model_{}".format(self.model_counter))


            if self.metadata_dict["result_on_validation_best"] is None:
                new_best_config_found = True
            else:
                best_solution_val = self.metadata_dict["result_on_validation_best"][self.metric_to_optimize]
                new_best_config_found = best_solution_val < result_dict[self.metric_to_optimize]


            if new_best_config_found:

                self._write_log("{}: New best config found. Config {}: {} - results: {}\n".format(self.ALGORITHM_NAME,
                                                                                           self.model_counter,
                                                                                           current_fit_parameters_dict,
                                                                                           result_string))

                if self.save_model in ["all", "best"]:
                    self._print("{}: Saving model in {}\n".format(self.ALGORITHM_NAME, self.output_folder_path + self.output_file_name_root))
                    recommender_instance.save_model(self.output_folder_path, file_name =self.output_file_name_root + "_best_model")


                if self.evaluator_test is not None and self.evaluate_on_test_each_best_solution:
                    result_dict_test, _, evaluation_test_time = self._evaluate_on_test(recommender_instance, current_fit_parameters_dict, print_log = True)


            else:
                self._write_log("{}: Config {} is suboptimal. Config: {} - results: {}\n".format(self.ALGORITHM_NAME,
                                                                                          self.model_counter,
                                                                                          current_fit_parameters_dict,
                                                                                          result_string))



            if current_result >= self.INVALID_CONFIG_VALUE:
                self._write_log("{}: WARNING! Config {} returned a value equal or worse than the default value to be assigned to invalid configurations."
                                " If no better valid configuration is found, this parameter search may produce an invalid result.\n")



            self.metadata_dict["result_on_validation_list"][self.model_counter] = result_dict.copy()

            self.metadata_dict["time_on_train_list"][self.model_counter] = train_time
            self.metadata_dict["time_on_validation_list"][self.model_counter] = evaluation_time

            self.metadata_dict["time_on_train_total"], self.metadata_dict["time_on_train_avg"] = \
                _compute_avg_time_non_none_values(self.metadata_dict["time_on_train_list"])
            self.metadata_dict["time_on_validation_total"], self.metadata_dict["time_on_validation_avg"] = \
                _compute_avg_time_non_none_values(self.metadata_dict["time_on_validation_list"])


            if new_best_config_found:
                self.metadata_dict["hyperparameters_best"] = current_fit_parameters_dict.copy()
                self.metadata_dict["hyperparameters_best_index"] = self.model_counter
                self.metadata_dict["result_on_validation_best"] = result_dict.copy()

                if self.evaluator_test is not None and self.evaluate_on_test_each_best_solution:
                    self.metadata_dict["result_on_test_best"] = result_dict_test.copy()
                    self.metadata_dict["result_on_test_list"][self.model_counter] = result_dict_test.copy()
                    self.metadata_dict["time_on_test_list"][self.model_counter] = evaluation_test_time

                    self.metadata_dict["time_on_test_total"], self.metadata_dict["time_on_test_avg"] = \
                        _compute_avg_time_non_none_values(self.metadata_dict["time_on_test_list"])


        except (KeyboardInterrupt, SystemExit) as e:
            # If getting a interrupt, terminate without saving the exception
            raise e

        except:
            # Catch any error: Exception, Tensorflow errors etc...

            traceback_string = traceback.format_exc()

            self._write_log("{}: Config {} Exception. Config: {} - Exception: {}\n".format(self.ALGORITHM_NAME,
                                                                                  self.model_counter,
                                                                                  current_fit_parameters_dict,
                                                                                  traceback_string))

            self.metadata_dict["exception_list"][self.model_counter] = traceback_string


            # Assign to this configuration the worst possible score
            # Being a minimization problem, set it to the max value of a float
            current_result = + self.INVALID_CONFIG_VALUE

            traceback.print_exc()



        if self.save_metadata:
            self.dataIO.save_data(data_dict_to_save = self.metadata_dict.copy(),
                                  file_name = self.output_file_name_root + "_metadata")

        self.model_counter += 1

        return current_result
