#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 10/03/2018

@author: Maurizio Ferrari Dacrema
"""



class SearchInputRecommenderParameters(object):


    def __init__(self,
                   # Dictionary of paramethers needed by the constructor
                   CONSTRUCTOR_POSITIONAL_ARGS = None,
                   CONSTRUCTOR_KEYWORD_ARGS = None,

                   # List containing all positional arguments needed by the fit function
                   FIT_POSITIONAL_ARGS = None,
                   FIT_KEYWORD_ARGS = None
                   ):


          super(SearchInputRecommenderParameters, self).__init__()

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


















def writeLog(string, logFile):

    print(string)

    if logFile is not None:
        logFile.write(string)
        logFile.flush()





class SearchAbstractClass(object):

    ALGORITHM_NAME = "SearchAbstractClass"

    def __init__(self, recommender_class,
                 evaluator_validation = None,
                 evaluator_test = None):


        super(SearchAbstractClass, self).__init__()

        self.recommender_class = recommender_class

        self.results_test_best = {}
        self.paramether_dictionary_best = {}

        if evaluator_validation is None:
            raise ValueError("{}: evaluator_validation must be provided".format(self.ALGORITHM_NAME))
        else:
            self.evaluator_validation = evaluator_validation

        if evaluator_test is None:
            self.evaluator_test = None
        else:
            self.evaluator_test = evaluator_test





    def search(self, recommender_constructor_dict,
               parameter_search_space,
               metric_to_optimize = "MAP",
               n_cases = None,
               output_folder_path = None,
               output_file_name_root = None,
               parallelize = False,
               save_model = "best",
               save_metadata = True,
               ):

        raise NotImplementedError("Function search not implementated for this class")
