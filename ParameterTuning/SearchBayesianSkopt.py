#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/12/18

@author: Emanuele Chioso, Maurizio Ferrari Dacrema
"""

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

from ParameterTuning.SearchAbstractClass import SearchAbstractClass
import traceback


class SearchBayesianSkopt(SearchAbstractClass):

    ALGORITHM_NAME = "SearchBayesianSkopt"

    def __init__(self, recommender_class, evaluator_validation = None, evaluator_test = None, verbose = True):

        assert evaluator_validation is not None, "{}: evaluator_validation must be provided".format(self.ALGORITHM_NAME)

        super(SearchBayesianSkopt, self).__init__(recommender_class,
                                                  evaluator_validation = evaluator_validation,
                                                  evaluator_test = evaluator_test,
                                                  verbose = verbose)



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



    def _resume_from_saved(self):

        try:
            self.metadata_dict = self.dataIO.load_data(file_name = self.output_file_name_root + "_metadata")

        except (KeyboardInterrupt, SystemExit) as e:
            # If getting a interrupt, terminate without saving the exception
            raise e

        except FileNotFoundError:
            self._write_log("{}: Resuming '{}' Failed, no such file exists.\n".format(self.ALGORITHM_NAME, self.output_file_name_root))
            self.resume_from_saved = False
            return None, None

        except Exception as e:
            self._write_log("{}: Resuming '{}' Failed, generic exception: {}.\n".format(self.ALGORITHM_NAME, self.output_file_name_root, str(e)))
            traceback.print_exc()
            self.resume_from_saved = False
            return None, None

        # Get hyperparameter list and corresponding result
        # Make sure that the hyperparameters only contain those given as input and not others like the number of epochs
        # selected by earlystopping
        hyperparameters_list_saved = self.metadata_dict['hyperparameters_list']
        result_on_validation_list_saved = self.metadata_dict['result_on_validation_list']

        hyperparameters_list_input = []
        result_on_validation_list_input = []

        # The hyperparameters are saved for all cases even if they throw an exception
        while self.model_counter<len(hyperparameters_list_saved) and hyperparameters_list_saved[self.model_counter] is not None:

            hyperparameters_config_saved = hyperparameters_list_saved[self.model_counter]

            hyperparameters_config_input = []

            # Add only those having a search space, in the correct ordering
            for index in range(len(self.hyperparams_names)):
                key = self.hyperparams_names[index]
                value_saved = hyperparameters_config_saved[key]

                # Check if single value categorical. It is aimed at intercepting
                # Hyperparameters that are chosen via early stopping and set them as the
                # maximum value as per hyperparameter search space. If not, the gp_minimize will return an error
                # as some values will be outside (lower) than the search space

                if isinstance(self.hyperparams_values[index], Categorical) and self.hyperparams_values[index].transformed_size == 1:
                    value_input = self.hyperparams_values[index].bounds[0]
                else:
                    value_input = value_saved

                hyperparameters_config_input.append(value_input)


            hyperparameters_list_input.append(hyperparameters_config_input)

            # Check if the hyperparameters have a valid result or an exception
            validation_result = result_on_validation_list_saved[self.model_counter]

            if validation_result is None:
                # Exception detected
                result_on_validation_list_input.append(+ self.INVALID_CONFIG_VALUE)

                assert self.metadata_dict["exception_list"][self.model_counter] is not None, \
                    "{}: Resuming '{}' Failed due to inconsistent data. Invalid validation result found in position {} but no corresponding exception detected.".format(self.ALGORITHM_NAME, self.output_file_name_root, self.model_counter)
            else:
                result_on_validation_list_input.append(- validation_result[self.metric_to_optimize])



            self.model_counter += 1


        self._print("{}: Resuming '{}'... Loaded {} configurations.".format(self.ALGORITHM_NAME, self.output_file_name_root, self.model_counter))


        # If the data structure exists but is empty, return None
        if len(hyperparameters_list_input) == 0:
            self.resume_from_saved = False
            return None, None

        # If loaded less configurations than desired ones
        if self.model_counter < self.n_calls:
            self.resume_from_saved = False


        return hyperparameters_list_input, result_on_validation_list_input








    def search(self, recommender_input_args,
               parameter_search_space,
               metric_to_optimize = "MAP",
               n_cases = 20,
               n_random_starts = 5,
               output_folder_path = None,
               output_file_name_root = None,
               save_model = "best",
               save_metadata = True,
               resume_from_saved = False,
               recommender_input_args_last_test = None,
               evaluate_on_test_each_best_solution = True,
               ):
        """

        :param recommender_input_args:
        :param parameter_search_space:
        :param metric_to_optimize:
        :param n_cases:
        :param n_random_starts:
        :param output_folder_path:
        :param output_file_name_root:
        :param save_model:          "no"    don't save anything
                                    "all"   save every model
                                    "best"  save the best model trained on train data alone and on last, if present
                                    "last"  save only last, if present
        :param save_metadata:
        :param recommender_input_args_last_test:
        :return:
        """


        self._set_skopt_params()    ### default parameters are set here

        self._set_search_attributes(recommender_input_args,
                                    recommender_input_args_last_test,
                                    metric_to_optimize,
                                    output_folder_path,
                                    output_file_name_root,
                                    resume_from_saved,
                                    save_metadata,
                                    save_model,
                                    evaluate_on_test_each_best_solution,
                                    n_cases)


        self.parameter_search_space = parameter_search_space
        self.n_random_starts = n_random_starts
        self.n_calls = n_cases
        self.n_jobs = 1
        self.n_loaded_counter = 0


        self.hyperparams = dict()
        self.hyperparams_names = list()
        self.hyperparams_values = list()

        skopt_types = [Real, Integer, Categorical]

        for name, hyperparam in self.parameter_search_space.items():

            if any(isinstance(hyperparam, sko_type) for sko_type in skopt_types):
                self.hyperparams_names.append(name)
                self.hyperparams_values.append(hyperparam)
                self.hyperparams[name] = hyperparam

            else:
                raise ValueError("{}: Unexpected parameter type: {} - {}".format(self.ALGORITHM_NAME, str(name), str(hyperparam)))


        if self.resume_from_saved:
            hyperparameters_list_input, result_on_validation_list_saved = self._resume_from_saved()
            self.x0 = hyperparameters_list_input
            self.y0 = result_on_validation_list_saved

            self.n_random_starts = max(0, self.n_random_starts - self.model_counter)
            self.n_calls = max(0, self.n_calls - self.model_counter)
            self.n_loaded_counter = self.model_counter



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


        if self.n_loaded_counter < self.model_counter:
            self._write_log("{}: Search complete. Best config is {}: {}\n".format(self.ALGORITHM_NAME,
                                                                           self.metadata_dict["hyperparameters_best_index"],
                                                                           self.metadata_dict["hyperparameters_best"]))


        if self.recommender_input_args_last_test is not None:
            self._evaluate_on_test_with_data_last()







    def _objective_function_list_input(self, current_fit_parameters_list_of_values):

        current_fit_parameters_dict = dict(zip(self.hyperparams_names, current_fit_parameters_list_of_values))

        return self._objective_function(current_fit_parameters_dict)

