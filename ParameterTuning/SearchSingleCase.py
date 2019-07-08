#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/12/18

@author: Emanuele Chioso, Maurizio Ferrari Dacrema
"""

import os

from ParameterTuning.SearchAbstractClass import writeLog
from ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt



class SearchSingleCase(SearchBayesianSkopt):

    ALGORITHM_NAME = "SearchSingleCase"

    def __init__(self, recommender_class, evaluator_validation = None, evaluator_test = None):

        super(SearchSingleCase, self).__init__(recommender_class,
                                               evaluator_validation= evaluator_validation,
                                               evaluator_test=evaluator_test)



    def search(self, recommender_constructor_data,
               fit_parameters_values = None,
               metric_to_optimize = "MAP",
               output_folder_path = None,
               output_file_name_root = None,
               save_metadata = True,
               ):


        assert fit_parameters_values is not None, "{}: fit_parameters_values must contain a dictionary".format(self.ALGORITHM_NAME)

        self.recommender_constructor_data = recommender_constructor_data
        self.metric_to_optimize = metric_to_optimize
        self.output_folder_path = output_folder_path
        self.output_file_name_root = output_file_name_root
        self.best_solution_val = None

        # If directory does not exist, create
        if not os.path.exists(self.output_folder_path):
            os.makedirs(self.output_folder_path)


        self.log_file = open(self.output_folder_path + self.output_file_name_root + "_{}.txt".format(self.ALGORITHM_NAME), "a")

        self.save_metadata = save_metadata
        self.n_calls = 1
        self.model_counter = 0
        self.best_solution_counter = 0
        self.save_model = "best"

        self.hyperparams_names = {}
        self.hyperparams_single_value = {}


        if self.save_metadata:
            self._init_metadata_dict()


        self._objective_function(fit_parameters_values)

        writeLog("{}: Search complete. Best config is {}: {}\n".format(self.ALGORITHM_NAME,
                                                                       self.best_solution_counter,
                                                                       fit_parameters_values), self.log_file)





