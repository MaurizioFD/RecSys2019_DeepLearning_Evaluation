#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/12/18

@author: Emanuele Chioso, Maurizio Ferrari Dacrema
"""

import pickle, time, os, traceback
import numpy as np

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

from ParameterTuning.SearchAbstractClass import SearchAbstractClass, writeLog
from Base.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping



class SearchBayesianSkopt(SearchAbstractClass):

    ALGORITHM_NAME = "SearchBayesianSkopt"

    # Value to be assigned to invalid configuration or if an Exception is raised
    INVALID_CONFIG_VALUE = np.finfo(np.float16).max


    def __init__(self, recommender_class, evaluator_validation = None, evaluator_test = None):


        super(SearchBayesianSkopt, self).__init__(recommender_class,
                                                  evaluator_validation= evaluator_validation,
                                                  evaluator_test=evaluator_test)



    def _set_skopt_params(self, n_calls = 70,
                          n_random_starts = 20,
                          n_points = 10000,
                          n_jobs = 1,
                          # noise = 'gaussian',
                          noise = 1e-5,
                          acq_func = 'gp_hedge',
                          acq_optimizer = 'auto',
                          random_state = None,
                          verbose = True,
                          n_restarts_optimizer = 10,
                          xi = 0.01,
                          kappa = 1.96,
                          x0 = None,
                          y0 = None):
        """
        wrapper to change the params of the bayesian optimizator.
        for further details:
        https://scikit-optimize.github.io/#skopt.gp_minimize

        """
        self.n_point = n_points
        self.n_calls = n_calls
        self.n_random_starts = n_random_starts
        self.n_jobs = n_jobs
        self.acq_func = acq_func
        self.acq_optimizer = acq_optimizer
        self.random_state = random_state
        self.n_restarts_optimizer = n_restarts_optimizer
        self.verbose = verbose
        self.xi = xi
        self.kappa = kappa
        self.noise = noise
        self.x0 = x0
        self.y0 = y0


    def _init_metadata_dict(self):

        self.metadata_dict = {"algorithm_name": self.ALGORITHM_NAME,
                              "parameters_list": [None]*self.n_calls,
                              "validation_result_list": [None]*self.n_calls,
                              "train_time_list": [None]*self.n_calls,
                              "evaluation_time_list": [None]*self.n_calls,
                              "train_time_total": 0.0,
                              "evaluation_time_total": 0.0,
                              "train_time_avg": 0.0,
                              "evaluation_time_avg": 0.0,

                              "test_result_list": [None]*self.n_calls,
                              "evaluation_test_time_list": [None]*self.n_calls,
                              "evaluation_test_time_total": 0.0,
                              "evaluation_test_time_avg": 0.0,

                              "best_parameters": None,
                              "best_parameters_index": None,
                              "best_result_validation": None,
                              "best_result_test": None,
                              }




    def search(self, recommender_constructor_data,
               parameter_search_space,
               metric_to_optimize = "MAP",
               n_cases = 20,
               n_random_starts = 5,
               output_folder_path = None,
               output_file_name_root = None,
               save_model = "best",
               save_metadata = True,
               ):

        assert save_model in ["no", "all", "best"], "{}: parameter save_model must be in '['no', 'all', 'best']', provided was '{}'.".format(self.ALGORITHM_NAME, save_model)
        self.save_model = save_model



        self._set_skopt_params()    ### default parameters are set here

        self.recommender_constructor_data = recommender_constructor_data
        self.parameter_search_space = parameter_search_space
        self.metric_to_optimize = metric_to_optimize
        self.output_folder_path = output_folder_path
        self.output_file_name_root = output_file_name_root
        self.n_random_starts = n_random_starts
        self.n_calls = n_cases


        # If directory does not exist, create
        if not os.path.exists(self.output_folder_path):
            os.makedirs(self.output_folder_path)


        self.log_file = open(self.output_folder_path + self.output_file_name_root + "_{}.txt".format(self.ALGORITHM_NAME), "a")
        self.model_counter = 0
        self.best_solution_val = None
        self.best_solution_counter = 0

        self.n_jobs = 1
        self.save_metadata = save_metadata


        if self.save_metadata:
            self._init_metadata_dict()




        self.hyperparams = dict()
        self.hyperparams_names = list()
        self.hyperparams_values = list()
        self.hyperparams_single_value = dict()

        skopt_types = [Real, Integer, Categorical]

        for name, hyperparam in self.parameter_search_space.items():

            if any(isinstance(hyperparam, sko_type) for sko_type in skopt_types):
                self.hyperparams_names.append(name)
                self.hyperparams_values.append(hyperparam)
                self.hyperparams[name] = hyperparam

            elif(isinstance(hyperparam, str) or isinstance(hyperparam, int) or isinstance(hyperparam, bool)):
                self.hyperparams_single_value[name] = hyperparam

            else:
                raise ValueError("{}: Unexpected parameter type: {} - {}".format(self.ALGORITHM_NAME, str(name), str(hyperparam)))



        self.result = gp_minimize(self._objective_function_list_input,
                                  self.hyperparams_values,
                                  base_estimator=None,
                                  n_calls=self.n_calls,
                                  n_random_starts=self.n_random_starts,
                                  acq_func=self.acq_func,
                                  acq_optimizer=self.acq_optimizer,
                                  x0=self.x0,
                                  y0=self.y0,
                                  random_state=self.random_state,
                                  verbose=self.verbose,
                                  callback=None,
                                  n_points=self.n_point,
                                  n_restarts_optimizer=self.n_restarts_optimizer,
                                  xi=self.xi,
                                  kappa=self.kappa,
                                  noise=self.noise,
                                  n_jobs=self.n_jobs)

        writeLog("{}: Search complete. Best config is {}: {}\n".format(self.ALGORITHM_NAME, self.best_solution_counter, self.best_solution_parameters), self.log_file)




    def _evaluate(self, current_fit_parameters):

        start_time = time.time()

        # Construct a new recommender instance
        recommender_instance = self.recommender_class(*self.recommender_constructor_data.CONSTRUCTOR_POSITIONAL_ARGS,
                                                      **self.recommender_constructor_data.CONSTRUCTOR_KEYWORD_ARGS)


        print("{}: Testing config:".format(self.ALGORITHM_NAME), current_fit_parameters)


        recommender_instance.fit(*self.recommender_constructor_data.FIT_POSITIONAL_ARGS,
                                 **self.recommender_constructor_data.FIT_KEYWORD_ARGS,
                                 **current_fit_parameters,
                                 **self.hyperparams_single_value)

        train_time = time.time() - start_time
        start_time = time.time()

        # Evaluate recommender and get results for the first cutoff
        result_dict, result_string = self.evaluator_validation.evaluateRecommender(recommender_instance)
        result_dict = result_dict[list(result_dict.keys())[0]]

        evaluation_time = time.time() - start_time

        return result_dict, result_string, recommender_instance, train_time, evaluation_time







    def _evaluate_on_test(self, recommender_instance):

        start_time = time.time()

        # Evaluate recommender and get results for the first cutoff
        result_dict, result_string = self.evaluator_test.evaluateRecommender(recommender_instance)

        evaluation_test_time = time.time() - start_time

        writeLog("{}: Best result evaluated on URM_test. Config: {} - results:\n{}\n".format(self.ALGORITHM_NAME,
                                                                                             self.best_solution_parameters,
                                                                                             result_string), self.log_file)

        return result_dict, evaluation_test_time


    def _objective_function_list_input(self, current_fit_parameters_list_of_values):

        current_fit_parameters_dict = dict(zip(self.hyperparams_names, current_fit_parameters_list_of_values))

        return self._objective_function(current_fit_parameters_dict)



    def _objective_function(self, current_fit_parameters_dict):

        try:

            result_dict, _, recommender_instance, train_time, evaluation_time = self._evaluate(current_fit_parameters_dict)

            current_result = - result_dict[self.metric_to_optimize]

            # If the recommender uses Earlystopping, get the selected number of epochs
            if isinstance(recommender_instance, Incremental_Training_Early_Stopping):

                n_epochs_early_stopping_dict = recommender_instance.get_early_stopping_final_epochs_dict()
                current_fit_parameters_dict = current_fit_parameters_dict.copy()

                for epoch_label in n_epochs_early_stopping_dict.keys():

                    epoch_value = n_epochs_early_stopping_dict[epoch_label]
                    current_fit_parameters_dict[epoch_label] = epoch_value




            if self.save_metadata:
                self.metadata_dict["parameters_list"][self.model_counter] = current_fit_parameters_dict.copy()
                self.metadata_dict["validation_result_list"][self.model_counter] = result_dict.copy()
                self.metadata_dict["train_time_list"][self.model_counter] = train_time
                self.metadata_dict["evaluation_time_list"][self.model_counter] = evaluation_time
                self.metadata_dict["train_time_total"] += train_time
                self.metadata_dict["evaluation_time_total"] += evaluation_time

                self.metadata_dict["train_time_avg"] = self.metadata_dict["train_time_total"]/(self.model_counter+1)
                self.metadata_dict["evaluation_time_avg"] = self.metadata_dict["evaluation_time_total"]/(self.model_counter+1)




            # Always save best model separately
            if self.save_model == "all":

                print("{}: Saving model in {}\n".format(self.ALGORITHM_NAME, self.output_folder_path + self.output_file_name_root))

                recommender_instance.saveModel(self.output_folder_path, file_name = self.output_file_name_root + "_model_{}".format(self.model_counter))



            if self.best_solution_val == None or self.best_solution_val < result_dict[self.metric_to_optimize]:

                writeLog("{}: New best config found. Config {}: {} - results: {}\n".format(self.ALGORITHM_NAME,
                                                                                           self.model_counter,
                                                                                           current_fit_parameters_dict,
                                                                                           result_dict), self.log_file)

                self.best_solution_val = result_dict[self.metric_to_optimize]
                self.best_solution_counter = self.model_counter
                self.best_solution_parameters = current_fit_parameters_dict.copy()

                if self.save_metadata:
                    self.metadata_dict["best_parameters"] = current_fit_parameters_dict.copy()
                    self.metadata_dict["best_result_validation"] = result_dict.copy()
                    self.metadata_dict["best_parameters_index"] = self.best_solution_counter





                if self.save_model != "no":
                    print("{}: Saving model in {}\n".format(self.ALGORITHM_NAME, self.output_folder_path + self.output_file_name_root))
                    recommender_instance.saveModel(self.output_folder_path, file_name =self.output_file_name_root + "_best_model")


                if self.evaluator_test is not None:
                    result_dict_test, evaluation_test_time = self._evaluate_on_test(recommender_instance)


                    if self.save_metadata:
                        self.metadata_dict["best_result_test"] = result_dict_test.copy()
                        self.metadata_dict["test_result_list"][self.model_counter] = result_dict_test.copy()
                        self.metadata_dict["evaluation_test_time_list"][self.model_counter] = evaluation_test_time
                        self.metadata_dict["evaluation_test_time_total"] += evaluation_test_time

                        tested_models = sum([value is not None for value in self.metadata_dict["test_result_list"]])

                        self.metadata_dict["evaluation_test_time_avg"] = self.metadata_dict["evaluation_test_time_total"]/tested_models



            else:
                writeLog("{}: Config {} is suboptimal. Config: {} - results: {}\n".format(self.ALGORITHM_NAME,
                                                                                          self.model_counter,
                                                                                          current_fit_parameters_dict,
                                                                                          result_dict), self.log_file)



            if self.save_metadata:

                pickle.dump(self.metadata_dict.copy(),
                            open(self.output_folder_path + self.output_file_name_root + "_metadata", "wb"),
                            protocol=pickle.HIGHEST_PROTOCOL)


            if current_result >= self.INVALID_CONFIG_VALUE:
                writeLog("{}: WARNING! Config {} returned a value equal or worse than the default value to be assigned to invalid configurations."
                         " If no better valid configuration is found, this parameter search may produce an invalid result.\n", self.log_file)


        except Exception as exc:

            writeLog("{}: Config {} Exception. Config: {} - Exception: {}\n".format(self.ALGORITHM_NAME,
                                                                                  self.model_counter,
                                                                                  current_fit_parameters_dict,
                                                                                  str(exc)), self.log_file)

            # Assign to this configuration the worst possible score
            # Being a minimization problem, set it to the max value of a float
            current_result = + self.INVALID_CONFIG_VALUE

            traceback.print_exc()


        self.model_counter += 1


        return current_result
