import numpy as np
import pickle
import os

def get_recommender_from_dict(rec_dict, fit=True):

    rec_class = rec_dict['best_recommender_class']
    parameters = rec_dict['best_solution_parameters']
    rec_constructor = rec_dict['best_constructor_arguments']

    rec = rec_class(*rec_constructor.CONSTRUCTOR_POSITIONAL_ARGS, **rec_constructor.CONSTRUCTOR_KEYWORD_ARGS)

    if fit:
        rec.fit(*rec_constructor.FIT_POSITIONAL_ARGS, **rec_constructor.FIT_KEYWORD_ARGS, **parameters)

    return rec


def get_dict_from_parameter_search(parameterSearch):
    d = {
        'best_recommender_class': parameterSearch.recommender_class,
        'best_constructor_arguments': parameterSearch.recommender_constructor_data,
        'best_solution_counter': parameterSearch.best_solution_counter,
        'best_solution_parameters': parameterSearch.best_solution_parameters,
        'best_solution_val': parameterSearch.best_solution_val,
        }
    return d

def get_highest_configuration_dict(list_dict, verbose=True):
    best_dict = {}
    top_value = -1 * np.inf

    for d in list_dict:
        value = d['best_solution_val']
        if value > top_value:
            top_value = value
            best_dict = d

    if verbose:
        print('Best config found from skopt optimize:\n\t\tclass: {}\n\t\tparameters: {}\n\t\tvalue:{}'.format(d['best_recommender_class'].RECOMMENDER_NAME, d['best_solution_parameters'], d['best_solution_val']))

    return best_dict


def get_lowest_configuration_dict(list_dict, verbose=True):
    best_dict = {}
    lowest_value = np.inf

    for d in list_dict:
        value = d['best_solution_val']
        if value < lowest_value:
            lowest_value = value
            best_dict = d

    if verbose:
        print('Best config found from skopt optimize:\n\t\tclass: {}\n\t\tparameters: {}\n\t\tvalue:{}'.format(d['best_recommender_class'].RECOMMENDER_NAME, d['best_solution_parameters'], d['best_solution_val']))

    return best_dict

def get_best_model_from_metadata_dicts(output_folder_path, recommender_class_list, metric, filter_similarity_list=None,
                             highest_value=True, on_validation_value=True, verbose=True):
    best_dict = None
    best_recommender_class = None
    best_file_name = None
    top_value = -1 * np.inf
    lowest_value = np.inf
    list_metadata_dicts = []

    if on_validation_value:
        value_key = 'best_result_validation'
    else:
        value_key = 'best_result_test'

    if verbose and filter_similarity_list is not None:
        print('Search beast metadata model: include only model with keywords: {}'.format(filter_similarity_list))

    for recommender_class in recommender_class_list:
        string_rec_class = recommender_class.__name__
        for f in os.listdir(output_folder_path):

            if not _file_include_word(f, filter_similarity_list):
                continue

            if os.path.isfile(os.path.join(output_folder_path, f)) and string_rec_class in f and '_metadata' in f:
                try:
                    data_dict = pickle.load(open(output_folder_path + f, "rb"))
                    value = data_dict[value_key][metric]
                    list_metadata_dicts.append(f.replace('_metadata',''))

                    if highest_value and value > top_value:
                        top_value = value
                        best_file_name = f.replace('_metadata','')
                        best_dict = data_dict
                        best_recommender_class = recommender_class

                    elif not highest_value and value < lowest_value:
                        lowest_value = value
                        best_file_name = f.replace('_metadata','')
                        best_dict = data_dict
                        best_recommender_class = recommender_class

                except FileNotFoundError:
                    print('Error reading metadata file: {}'.format(f))

    if verbose:
        print('Best config found among all {} available metadata dicts:\n\tdicts:\t\t{}:\n\tbest model:\t{}\n\tparameters:\t{}\n\tvalidation:\t{}\n\ttest:\t\t{}'
              .format(len(list_metadata_dicts), list_metadata_dicts, best_recommender_class.__name__, best_dict['best_parameters'], best_dict['best_result_validation'], best_dict['best_result_test'])
              )

    best_model_name = best_file_name + '_best_model'
    return best_recommender_class, best_model_name, best_dict,

def _file_include_word(file, word_list):
    if word_list is None:
        return True
    else:
        for word in word_list:
            if word in file:
                return True
        return False


def load_recommender(recommender_class, constructor_kwargs, folder_path, model_name):
    rec = recommender_class(**constructor_kwargs)
    rec.loadModel(folder_path=folder_path, file_name=model_name)
    return rec

