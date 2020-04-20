#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 04/06/2019

@author: Maurizio Ferrari Dacrema
"""

from Base.DataIO import DataIO
from Utils.seconds_to_biggest_unit import seconds_to_biggest_unit
from Recommender_import_list import *
import re, numbers
from functools import partial

def _get_printable_recommender_name(RECOMMENDER_NAME):

    recommender_printable_name = RECOMMENDER_NAME
    recommender_printable_name = recommender_printable_name.replace("Recommender", "")
    recommender_printable_name = recommender_printable_name.replace("_", " ")
    recommender_printable_name = re.sub("CF$", " CF", recommender_printable_name)
    recommender_printable_name = re.sub("CBF$", " CBF", recommender_printable_name)
    recommender_printable_name = re.sub("SLIM", "SLIM ", recommender_printable_name)
    recommender_printable_name = recommender_printable_name.replace("MatrixFactorization", "MF")
    recommender_printable_name = recommender_printable_name.replace("SLIM", "SLIM ")
    recommender_printable_name = recommender_printable_name.replace(" Hybrid", "")
    recommender_printable_name = recommender_printable_name.replace(" Cython", "")
    recommender_printable_name = recommender_printable_name.replace(" Wrapper", "")
    recommender_printable_name = recommender_printable_name.replace(" Matlab", "")

    recommender_printable_name = re.sub(" +", " ", recommender_printable_name)

    return recommender_printable_name




################################################################################################
######
######      BUILD FILE NAMES LIST
######


def _get_algorithm_similarity_and_feature_combinations(algorithm, algorithm_row_label, algorithm_file_name,
                                                       KNN_similarity_list, feature_matrix_names_to_report_list):

    algorithm_data_to_print_list = []

    # If the list is empty, add empty string
    if len(feature_matrix_names_to_report_list) == 0:
        feature_matrix_names_to_report_list = [""]


    for similarity in KNN_similarity_list:

        for feature_matrix_name in feature_matrix_names_to_report_list:

            # If only one feature matrix, don't print its name
            if len(feature_matrix_names_to_report_list) == 1:
                feature_matrix_row_label = ""

            else:
                feature_matrix_row_label = _get_printable_recommender_name(feature_matrix_name)
                feature_matrix_row_label += " "

            if feature_matrix_name != "":
                feature_matrix_name = "_" + feature_matrix_name


            algorithm_data_to_print_list.append({
                "algorithm": algorithm,
                "algorithm_row_label": algorithm_row_label + " " + feature_matrix_row_label + similarity,
                "algorithm_file_name": algorithm_file_name + feature_matrix_name  + "_" + similarity,
            })

    return algorithm_data_to_print_list



def _get_algorithm_file_name_list(algorithm_list,
                                  KNN_similarity_list = None,
                                  ICM_names_list = None,
                                  UCM_names_list = None,
                                  ):


    if KNN_similarity_list is None:
        KNN_similarity_list = ["cosine"]

    algorithm_data_to_print_list = []


    for algorithm in algorithm_list:

        if algorithm is None:
            algorithm_data_to_print_list.append(None)
            continue


        algorithm_row_label = _get_printable_recommender_name(algorithm.RECOMMENDER_NAME)

        algorithm_file_name = algorithm.RECOMMENDER_NAME

        # If KNN collaborative, expand similarity tipe but put no feature matrix
        if algorithm in [ItemKNNCFRecommender,
                         UserKNNCFRecommender]:

            this_algorithm_data_list = _get_algorithm_similarity_and_feature_combinations(algorithm, algorithm_row_label,
                                                                                          algorithm_file_name,
                                                                                          KNN_similarity_list,
                                                                                          [])

            algorithm_data_to_print_list.extend(this_algorithm_data_list)


        # If KNN item content based or hybrid item based, expand similarity type and ICM names
        elif algorithm in [ItemKNNCBFRecommender,
                           ItemKNN_CFCBF_Hybrid_Recommender,
                           # ItemKNNCBF_FW_Recommender
                           ]:

            if ICM_names_list is not None:
                this_algorithm_data_list = _get_algorithm_similarity_and_feature_combinations(algorithm, algorithm_row_label,
                                                                                              algorithm_file_name,
                                                                                              KNN_similarity_list,
                                                                                              ICM_names_list)

                algorithm_data_to_print_list.extend(this_algorithm_data_list)


        # If KNN user content based or hybrid user based, expand similarity type and UCM names
        elif algorithm in [UserKNNCBFRecommender,
                           UserKNN_CFCBF_Hybrid_Recommender,
                           # UserKNNCBF_FW_Recommender
                           ]:

            if UCM_names_list is not None:
                this_algorithm_data_list = _get_algorithm_similarity_and_feature_combinations(algorithm, algorithm_row_label,
                                                                                              algorithm_file_name,
                                                                                              KNN_similarity_list,
                                                                                              UCM_names_list)

                algorithm_data_to_print_list.extend(this_algorithm_data_list)


        else:

            algorithm_data_to_print_list.append({
                "algorithm": algorithm,
                "algorithm_row_label": algorithm_row_label,
                "algorithm_file_name": algorithm_file_name,
            })


    return algorithm_data_to_print_list



def _get_algorithm_metadata_to_print_list(result_folder_path,
                                          algorithm_list,
                                          KNN_similarity_list = None,
                                          ICM_names_list = None,
                                          UCM_names_list = None,
                                          ):


    dataIO = DataIO(folder_path = result_folder_path)

    algorithm_file_name_list = _get_algorithm_file_name_list(
                                    algorithm_list=algorithm_list,
                                    KNN_similarity_list = KNN_similarity_list,
                                    ICM_names_list = ICM_names_list,
                                    UCM_names_list = UCM_names_list)

    algorithm_metadata_to_print_list = []


    for algorithm_file_dict in algorithm_file_name_list:

        if algorithm_file_dict is None:
            algorithm_metadata_to_print_list.append(None)
            continue

        algorithm_file_name = algorithm_file_dict["algorithm_file_name"]

        search_metadata = None

        if algorithm_file_name is not None:
            try:
                search_metadata = dataIO.load_data(algorithm_file_name + "_metadata")
            except FileNotFoundError:
                pass

        algorithm_file_dict["search_metadata"] = search_metadata
        algorithm_metadata_to_print_list.append(algorithm_file_dict)


    return algorithm_metadata_to_print_list





################################################################################################
######
######      TIME STATS
######



def _mean_and_stdd_of_array(data_array):

    mean = np.mean(data_array)

    if len(data_array) == 1:
        stddev = 0.0
    else:
        stddev = np.std(data_array, ddof=1)

    return mean, stddev


def _convert_sec_list_into_biggest_unit(data_list):
    """
    Converts a list containing seconds into an equivalent list with a bigger time unit
    adjusting standard deviation as well
    :param data_list:
    :return:
    """

    data_array = np.array(data_list)

    mean_sec, stddev_sec = _mean_and_stdd_of_array(data_array)

    _, new_time_unit, data_array = seconds_to_biggest_unit(mean_sec, data_array = data_array)

    mean_new_unit, stddev_new_unit = _mean_and_stdd_of_array(data_array)

    return mean_sec, stddev_sec, new_time_unit, mean_new_unit, stddev_new_unit



def _time_string_builder(data_list, n_decimals=4):
    """
    Creates a nice printable string from the list of time lengths
    :param data_list:
    :param n_decimals:
    :return:
    """

    data_list = [finite_val for finite_val in data_list if finite_val is not None and np.isfinite(finite_val)]


    def _measure_unit_string(mean_sec, stddev, unit, n_decimals=4):

        result_row_string = "{:.{n_decimals}f}".format(mean_sec, n_decimals=n_decimals)

        if len(data_list) > 1:
            result_row_string += " $\\pm$ {:.{n_decimals}f}".format(stddev, n_decimals=n_decimals)

        result_row_string += " [{}] ".format(unit)

        return result_row_string



    if len(data_list)>0:

        # Step1: choose the appropriate measuring unit
        mean_sec, stddev_sec, new_time_unit, mean_new_unit, stddev_new_unit = _convert_sec_list_into_biggest_unit(data_list)

        if new_time_unit != "sec":
            result_row_string = "{:.{n_decimals}f} [{}]".format(mean_sec, "sec", n_decimals=n_decimals)
            result_row_string += " / " + _measure_unit_string(mean_new_unit, stddev_new_unit, new_time_unit, n_decimals=n_decimals)

        else:
            result_row_string = _measure_unit_string(mean_sec, stddev_sec, "sec", n_decimals=n_decimals)

    else:
        result_row_string = "-"


    return result_row_string







################################################################################################
######
######      MULTIPLE FOLDER HYPERPARAMETERS
######

def _format_hyperparameter_row_values(dataframe_row_series):

    algorithm_name, hyperparameter_name = dataframe_row_series.name

    for dataset_column in dataframe_row_series.index:
        hyperparameter_value = dataframe_row_series[dataset_column]

        if isinstance(hyperparameter_value, numbers.Real) and not isinstance(hyperparameter_value, numbers.Integral):
            # If the number is real valued, use exponential notation in some cases, fixed decimal in all others
            if any(substring in hyperparameter_name for substring in ["penalty", "rate", "reg", "l1", "lambda", "l2", "decay"]):
                hyperparameter_value = "{:.2E}".format(hyperparameter_value)
            else:
                hyperparameter_value = "{:.4f}".format(hyperparameter_value)

        else:
            hyperparameter_value = "{}".format(hyperparameter_value)

        dataframe_row_series[dataset_column] = hyperparameter_value


    return dataframe_row_series




def _print_latex_hyperparameters_from_dataframe(hyperparameters_dataframe, hyperparameters_file):

    hyperparameters_dataframe = hyperparameters_dataframe.copy()
    hyperparameters_dataframe.rename_axis(["Algorithm", "Hyperparameter"], inplace=True)

    # Get the columns before changing the index structure
    n_datasets = len(hyperparameters_dataframe.columns)

    hyperparameters_dataframe = hyperparameters_dataframe[~hyperparameters_dataframe.index.get_level_values("Algorithm").str.contains("algorithm_group")]

    # Set index as the combination of algorithm label and hyperparam name, this will create a multiindex dataframe
    # and enable the automatic multirow latex code
    # hyperparameters_dataframe.set_index(["Algorithm", "Hyperparameter"], inplace=True)
    latex_code = hyperparameters_dataframe.to_latex(index = True,
                                                    multirow = True,
                                                    escape = False,
                                                    column_format="ll|" + "c"*n_datasets)

    # Replace cline with midrule
    latex_code = latex_code.replace("\\cline{{1-{}}}".format(n_datasets+2), "\\midrule")

    # Also to_latex adds extra empty header row when index has a name, known BUG
    # https://github.com/pandas-dev/pandas/issues/26111

    header_wrong_code = " +& +" + "".join(["& +{} ".format(dataset) for dataset in hyperparameters_dataframe.columns]) + "\\\\\\\\\n" + \
                        "Algorithm & Hyperparameter "+ "& +" * n_datasets + "\\\\\\\\\n"

    header_correct_code = "Algorithm & Hyperparameter " + "".join(["&\t {} ".format(dataset) for dataset in hyperparameters_dataframe.columns]) + "\\\\\\\\\n"

    latex_code = re.sub(header_wrong_code, header_correct_code, latex_code)

    # Also, when the algorithm has only 1 hyperparameter the multirow is not added, therefore no midrule is added
    # We want to add the midrule
    # Rows with hyperparameters will have spaces at the beginning, a new algoritm block will have the name
    # Select all rows which are after a data row (which ends in "\\")
    # then they should contain some non-space and non-newline characters (i.e., the name of an algorithm either simple text or \multirow)
    # then get all the cell content until the & (some algorithm names will contain a space)
    separator_wrong_code = "(\\\\\\\\\n)([^ \n]+[^&\n]+&)"
    separator_correct_code = r"\1\\midrule\n\2"

    latex_code = re.sub(separator_wrong_code, separator_correct_code, latex_code)
    latex_code = re.sub("_", " ", latex_code)

    hyperparameters_file.write(latex_code)
    hyperparameters_file.close()




def _remove_missing_runs_for_algorithm(hyperparameters_dataframe):
    """
    Sometimes an algorithm is present only for some datasets. When it is missing this will generate
    an extra row with hyperparameter "nan", that should be removed if other hyperparameters are present
    :param hyperparameters_dataframe:
    :return:
    """

    # Search all algorithms having a hyperparameter whose name is "nan"
    none_hyperpar = hyperparameters_dataframe[hyperparameters_dataframe.index.get_level_values("hyperparameter_name").isnull()]

    # Check if they have more hyperparameters
    alg_number_hyperparam = hyperparameters_dataframe.groupby('algorithm_row_label').size() > 1

    # If so, remove the single "nan" hyperparameter
    single_nan_to_remove_flag = alg_number_hyperparam[none_hyperpar.index.get_level_values("algorithm_row_label")]
    single_nan_to_remove_flag = single_nan_to_remove_flag[single_nan_to_remove_flag]

    rows_to_drop = hyperparameters_dataframe.index.get_level_values("algorithm_row_label").isin(single_nan_to_remove_flag.index) & hyperparameters_dataframe.index.get_level_values("hyperparameter_name").isnull()
    hyperparameters_dataframe = hyperparameters_dataframe[~rows_to_drop]

    return hyperparameters_dataframe








def generate_latex_hyperparameters(result_folder_path,
                                   algorithm_name,
                                   experiment_subfolder_list,
                                   other_algorithm_list,
                                   file_name_suffix = "",
                                   KNN_similarity_to_report_list = None,
                                   ICM_names_to_report_list = None,
                                   UCM_names_to_report_list = None,
                                   split_per_algorithm_type = False,
                                   ):

    hyperparameters_dataframe = None

    for experiment_subfolder in experiment_subfolder_list:

        result_loader = ResultFolderLoader("{}/{}_{}/".format(result_folder_path, algorithm_name, experiment_subfolder),
                                         base_algorithm_list = None,
                                         other_algorithm_list = other_algorithm_list,
                                         KNN_similarity_list = KNN_similarity_to_report_list,
                                         ICM_names_list = ICM_names_to_report_list,
                                         UCM_names_list = UCM_names_to_report_list)

        hyperparameters_dataframe_subfolder = result_loader.get_hyperparameters_dataframe()
        hyperparameters_dataframe_subfolder.rename(columns={"hyperparameter_value": experiment_subfolder}, inplace = True)

        if hyperparameters_dataframe is None:
            hyperparameters_dataframe = hyperparameters_dataframe_subfolder
        else:
            # Perform outer join on two keys
            # We want that each hyperparameter value is mapped to the right algorithm and the right hyperparameter name
            # By using "outer" we ensure that even if a name is present only in one of the dataframes it will appear in the result
            # Furthermore validate="one_to_one" checks that there are not duplicates in the keys
            hyperparameters_dataframe = hyperparameters_dataframe.merge(hyperparameters_dataframe_subfolder,
                                                                        validate="one_to_one",
                                                                        how='outer',
                                                                        on = ["algorithm_row_label", "hyperparameter_name"])

    hyperparameters_dataframe = _remove_missing_runs_for_algorithm(hyperparameters_dataframe)

    # Generate latex code
    # Clean dataframe

    # Format hyperparameter values BEFORE splitting the dataframe, because that sometimes changes the data types
    hyperparameters_dataframe.fillna("-", inplace = True)
    hyperparameters_dataframe.index = pd.MultiIndex.from_frame(hyperparameters_dataframe.index.to_frame().fillna('-'))
    hyperparameters_dataframe.apply(_format_hyperparameter_row_values, axis=1)

    hyperparameters_file = open(result_folder_path + algorithm_name + file_name_suffix + "_latex_hyperparameters.txt", "w")
    _print_latex_hyperparameters_from_dataframe(hyperparameters_dataframe, hyperparameters_file)


    if split_per_algorithm_type:

        algorithm_type_group = {
            "KNN": [UserKNNCFRecommender,
                    ItemKNNCFRecommender,
                    ],
            "ML_graph":[P3alphaRecommender,
                        RP3betaRecommender,
                        EASE_R_Recommender,
                        SLIM_BPR_Cython,
                        SLIMElasticNetRecommender,
                        MatrixFactorization_BPR_Cython,
                        MatrixFactorization_FunkSVD_Cython,
                        PureSVDRecommender,
                        NMFRecommender,
                        IALSRecommender,
                        ],
            "CBF": [ItemKNNCBFRecommender,
                    UserKNNCBFRecommender,
                    ],
            "CFCBF":[ItemKNN_CFCBF_Hybrid_Recommender,
                    UserKNN_CFCBF_Hybrid_Recommender,
                    ],
            "neural":other_algorithm_list,
        }


        for group_label, group_alg_list in algorithm_type_group.items():
            # Create a dataframe with only algorithms in group
            group_label_list = [_get_printable_recommender_name(recommender_class.RECOMMENDER_NAME) for recommender_class in group_alg_list]

            # The group is matched if the index contains the label of the algorithm + a space or ends (to account for ItemKNN CF cosine and such)
            group_entries_flag = hyperparameters_dataframe.index.get_level_values("algorithm_row_label").str.contains('|'.join(["{label}\s|{label}$".format(label = label) for label in group_label_list]))
            group_hyperparameters_dataframe = hyperparameters_dataframe[group_entries_flag]

            if len(group_hyperparameters_dataframe)>0:
                hyperparameters_file = open(result_folder_path + algorithm_name + file_name_suffix + "_latex_hyperparameters_" + group_label + ".txt", "w")
                _print_latex_hyperparameters_from_dataframe(group_hyperparameters_dataframe, hyperparameters_file)






def _remove_duplicate_group_separator(result_dataframe):

        group_separator_flag = result_dataframe.index.str.startswith('algorithm_group')
        duplicate_consecutive_separators = np.logical_and(group_separator_flag[:-1], group_separator_flag[1:])
        duplicate_consecutive_separators = np.append(duplicate_consecutive_separators, False)

        result_dataframe = result_dataframe[np.logical_not(duplicate_consecutive_separators)]

        return result_dataframe



import os
import numpy as np
import pandas as pd


class ResultFolderLoader(object):
    """ResultFolderLoader"""

    # Default list of algorithms to be loaded
    # Each None will represent a horizontal line in the latex table
    _DEFAULT_BASE_ALGORITHM_LIST = [
        Random,
        TopPop,
        None,
        UserKNNCFRecommender,
        ItemKNNCFRecommender,
        P3alphaRecommender,
        RP3betaRecommender,
        None,
        EASE_R_Recommender,
        SLIM_BPR_Cython,
        SLIMElasticNetRecommender,
        # MatrixFactorization_AsySVD_Cython,
        MatrixFactorization_BPR_Cython,
        MatrixFactorization_FunkSVD_Cython,
        PureSVDRecommender,
        NMFRecommender,
        IALSRecommender,
        None,
        ItemKNNCBFRecommender,
        UserKNNCBFRecommender,
        None,
        ItemKNN_CFCBF_Hybrid_Recommender,
        UserKNN_CFCBF_Hybrid_Recommender,
    ]


    # Dictionary used to translate the metric name into Latex column header
    # Metrics whose name is not in this dictionary will be displayed with str(metric_key)
    _METRIC_NAME_TO_LATEX_LABEL_DICT = {
        "ROC_AUC":   "AUC",
        "PRECISION": "PREC",
        "PRECISION_RECALL_MIN_DEN":  "\\begin{tabular}{@{}c@{}}PREC \\\\ REC\\end{tabular}",
        "RECALL":   "REC",
        "MAP":      "MAP",
        "MRR":      "MRR",
        "NDCG":     "NDCG",
        "F1":       "F1",
        "HIT_RATE": "HR",
        "ARHR":     "ARHR",
        "NOVELTY":  "Novelty",
        "DIVERSITY_SIMILARITY":      "\\begin{tabular}{@{}c@{}}Div. \\\\ Similarity\\end{tabular}",
        "DIVERSITY_MEAN_INTER_LIST": "\\begin{tabular}{@{}c@{}}Div. \\\\ MIL\\end{tabular}",
        "DIVERSITY_HERFINDAHL":      "\\begin{tabular}{@{}c@{}}Div. \\\\ HHI\\end{tabular}",
        "COVERAGE_ITEM":             "\\begin{tabular}{@{}c@{}}Cov. \\\\ Item\\end{tabular}",
        "COVERAGE_USER":             "\\begin{tabular}{@{}c@{}}Cov. \\\\ User\\end{tabular}",
        "DIVERSITY_GINI":            "\\begin{tabular}{@{}c@{}}Div. \\\\ Gini\\end{tabular}",
        "SHANNON_ENTROPY":           "\\begin{tabular}{@{}c@{}}Div. \\\\ Shannon\\end{tabular}",
        }




    def __init__(self, folder_path,
                 base_algorithm_list = None,
                 other_algorithm_list = None,
                 KNN_similarity_list = None,
                 ICM_names_list = None,
                 UCM_names_list = None):

        super(ResultFolderLoader, self).__init__()

        assert os.path.isdir(folder_path), "ResultFolderLoader: folder_path does not exist '{}'".format(folder_path)

        self._folder_path = folder_path
        self._algorithm_list = base_algorithm_list.copy() if base_algorithm_list is not None else self._DEFAULT_BASE_ALGORITHM_LIST.copy()

        self._ICM_names_list = ICM_names_list
        self._UCM_names_list = UCM_names_list

        self._KNN_similarity_list = KNN_similarity_list if KNN_similarity_list is not None else ["cosine"]
        self._other_algorithm_list = other_algorithm_list.copy() if other_algorithm_list is not None else []

        if other_algorithm_list is not None:
            self._algorithm_list.extend([None, *self._other_algorithm_list])


        self._metadata_list = _get_algorithm_metadata_to_print_list(self._folder_path,
                                          algorithm_list = self._algorithm_list,
                                          KNN_similarity_list = self._KNN_similarity_list,
                                          ICM_names_list = self._ICM_names_list,
                                          UCM_names_list = self._UCM_names_list
                                          )


    def _get_column_name(self, metric_name, cutoff):
        metric_label = self._METRIC_NAME_TO_LATEX_LABEL_DICT[metric_name] if metric_name in self._METRIC_NAME_TO_LATEX_LABEL_DICT else metric_name
        return "{}@{}".format(metric_label, cutoff)

    def get_metadata(self):
        return self._metadata_list.copy()



    def get_results_dataframe(self,
                              metrics_list,
                              cutoffs_list,
                              ):
        """
        Loads the metadata in a dataframe
        :param metrics_list:
        :param cutoffs_list:
        :return:
        """

        algorithm_label_index = [row_dict["algorithm_row_label"] if row_dict is not None else "algorithm_group_{}".format(row_index) for row_index, row_dict in enumerate(self._metadata_list) ]
        cutoffs_list = [str(cutoff) for cutoff in cutoffs_list]

        # The dataframe will have the algorithm label as index and the carthesian product of cutoff and metric as columns
        cutoff_metric_multiindex = pd.MultiIndex.from_product([cutoffs_list, metrics_list])#, names=['cutoff', 'metric'])
        result_dataframe = pd.DataFrame(None, index=algorithm_label_index, columns = cutoff_metric_multiindex)
        # result_dataframe.rename_axis("Algorithm", inplace=True)

        # Remove duplicate group separator
        result_dataframe = _remove_duplicate_group_separator(result_dataframe)


        for row_index, row_dict in enumerate(self._metadata_list):

            if row_dict is None:
                continue

            algorithm_row_label = row_dict["algorithm_row_label"]
            search_metadata = row_dict["search_metadata"]

            for cutoff in cutoffs_list:

                for metric_name in metrics_list:
                    if search_metadata is not None:
                        result_on_last = search_metadata["result_on_last"]

                        if result_on_last is not None and cutoff in result_on_last and metric_name in result_on_last[cutoff]:
                            value = result_on_last[cutoff][metric_name]
                            result_dataframe.loc[algorithm_row_label, (cutoff, metric_name)] = value

        return result_dataframe




    def get_hyperparameters_dataframe(self):

        column_labels = ["algorithm_row_label", "hyperparameter_name", "hyperparameter_value"]

        result_dataframe = pd.DataFrame(columns=column_labels)


        for row_index, row_dict in enumerate(self._metadata_list):

            if row_dict is None:
                # Add None row to preserve the separation between different groups of algorithms
                # I don't like this but is simple enough and it works
                result_dataframe = result_dataframe.append({
                        "algorithm_row_label": "algorithm_group_{}".format(row_index),
                        "hyperparameter_name": None,
                        "hyperparameter_value": None,
                        }, ignore_index = True)

                continue


            algorithm_row_label = row_dict["algorithm_row_label"]
            search_metadata = row_dict["search_metadata"]

            # If search failed or was not done, add placeholder
            if search_metadata is None or search_metadata["hyperparameters_best"] is None:
                hyperparameters_best = {None:None}

            else:
                hyperparameters_best = search_metadata["hyperparameters_best"]

                # If it doesn't have hyperparameters don't add in dataframe
                if len(hyperparameters_best) == 0:
                    continue


            for hyperparameter_name, hyperparameter_value in hyperparameters_best.items():

                result_dataframe = result_dataframe.append({
                                        "algorithm_row_label": algorithm_row_label,
                                        "hyperparameter_name": hyperparameter_name,
                                        "hyperparameter_value": hyperparameter_value,
                                        }, ignore_index = True)

        result_dataframe.set_index(["algorithm_row_label", "hyperparameter_name"], inplace=True)

        return result_dataframe





    def generate_latex_results(self, output_file_path,
                               metrics_list,
                               cutoffs_list,
                               n_decimals = 4,
                               table_title = None,
                               highlight_best = True,
                               collapse_multicolumn_if_needed = True):

        result_dataframe = self.get_results_dataframe(metrics_list = metrics_list, cutoffs_list = cutoffs_list)
        output_file = open(output_file_path, "w")

        # If there is only a single cutoff, the multilevel columns can be collapsed
        if collapse_multicolumn_if_needed and (len(metrics_list) == 1):
            result_dataframe.columns = ['@'.join([self._METRIC_NAME_TO_LATEX_LABEL_DICT[col[1]], col[0]]).strip() for col in result_dataframe.columns.values]

        else:
            # Rename columns in such a way that they are nicely printable in latex
            result_dataframe.rename(columns= {col_name: "@ {}".format(col_name) for col_name in result_dataframe.columns.levels[0]},
                                    level = 0, inplace = True)

            result_dataframe.rename(columns= {col_name:self._METRIC_NAME_TO_LATEX_LABEL_DICT[col_name] for col_name in result_dataframe.columns.levels[1]},
                                    level = 1, inplace = True)



        if highlight_best:

            dataframe_baselines = result_dataframe.iloc[:-len(self._other_algorithm_list), :]
            dataframe_other_algs = result_dataframe.iloc[-len(self._other_algorithm_list):, :]

            dataframe_best_baseline_value = dataframe_baselines.max(axis=0)
            dataframe_best_other_alg_value = dataframe_other_algs.max(axis=0)


            def _format_result_row_values(dataframe_row_series, dataframe_threshold_value, n_decimals):

                for dataset_column in dataframe_row_series.index.tolist():

                    result_value = dataframe_row_series[dataset_column]

                    if dataset_column == "Algorithm" or not np.isfinite(result_value):
                            continue

                    if result_value > dataframe_threshold_value[dataset_column]:
                        result_string = "\\textbf{{{:.{n_decimals}f}}}".format(result_value, n_decimals = n_decimals)
                    else:
                        result_string = "{:.{n_decimals}f}".format(result_value, n_decimals = n_decimals)

                    dataframe_row_series[dataset_column] = result_string

                return dataframe_row_series

            dataframe_baselines = dataframe_baselines.apply(partial(_format_result_row_values,
                                           dataframe_threshold_value = dataframe_best_other_alg_value,
                                           n_decimals = n_decimals,
                                           ), axis=1, result_type = "broadcast")

            dataframe_other_algs = dataframe_other_algs.apply(partial(_format_result_row_values,
                                           dataframe_threshold_value = dataframe_best_baseline_value,
                                           n_decimals = n_decimals,
                                           ), axis=1, result_type = "broadcast")

            result_dataframe = pd.concat([dataframe_baselines, dataframe_other_algs], ignore_index=False)



        result_dataframe.fillna("-", inplace = True)

        n_metrics_cutoffs = len(result_dataframe.columns)
        latex_code = result_dataframe.to_latex(index = True,
                                               escape = False, #do not automatically escape special characters
                                               multicolumn = True,
                                               multicolumn_format = "c",
                                               column_format = "l|" + "{}|".format("c"*len(metrics_list))*len(cutoffs_list),
                                               float_format = "{:.4f}".format,
                                               )

        # Replace group separator with \midrule
        separator_old = "(\\\\\\\\\n)\s*algorithm_group_.+\\\\\\\\(\n)"
        separator_midrule = r"\1\\midrule\2"

        latex_code = re.sub(separator_old, separator_midrule, latex_code)

        # Prints table title
        if table_title is not None:
            header_old = "(\\\\toprule\n)"
            header_custom_title = r"\1\t&\t\\multicolumn{{{}}}{{c}}{{{}}} \\\\\n".format(n_metrics_cutoffs, table_title)

            latex_code = re.sub(header_old, header_custom_title, latex_code)


        # ad vline at the end of the line whith multicolumn
        separator_old = "(\\\multicolumn{[^}]*}{[^}]*}{[^}]*}\s*)(\\\\\\\\\n)"
        separator_midrule = r"\1\\vline\2"

        latex_code = re.sub(separator_old, separator_midrule, latex_code)
        latex_code = re.sub("_", " ", latex_code)

        output_file.write(latex_code)
        output_file.flush()

        output_file.close()





    def get_time_statistics_dataframe(self,
                                     n_decimals = 2,
                                     n_evaluation_users = None,
                                     ):


        # COLUMN HEADERS
        column_labels = ["Train Time",
                         "Recommendation Time",
                         "Recommendation Throughput"]

        algorithm_label_index = [row_dict["algorithm_row_label"] if row_dict is not None else "algorithm_group_{}".format(row_index) for row_index, row_dict in enumerate(self._metadata_list) ]
        result_dataframe = pd.DataFrame(None, index=algorithm_label_index, columns=column_labels)

        # Remove duplicate group separator
        result_dataframe = _remove_duplicate_group_separator(result_dataframe)

        for row_index, row_dict in enumerate(self._metadata_list):

            if row_dict is None:
                continue

            algorithm_row_label = row_dict["algorithm_row_label"]
            search_metadata = row_dict["search_metadata"]

            if search_metadata is not None:

                # Print mean and stdv of train time
                value_string = _time_string_builder(search_metadata["time_on_train_list"],
                                                          n_decimals=n_decimals)

                result_dataframe.loc[algorithm_row_label, "Train Time"] = value_string

                # Print mean and stdv of evaluation time
                value_string = _time_string_builder(search_metadata["time_on_test_list"],
                                                    n_decimals=n_decimals)

                result_dataframe.loc[algorithm_row_label, "Recommendation Time"] = value_string


                # Print n of users evaluated per second for the last model
                optimal_hyperparameters_index = search_metadata["hyperparameters_best_index"]

                if optimal_hyperparameters_index is None:
                    optimal_hyperparameters_test_time = None
                else:
                    optimal_hyperparameters_test_time = search_metadata["time_on_test_list"][optimal_hyperparameters_index]

                if n_evaluation_users is not None and optimal_hyperparameters_test_time is not None:
                    value_string = "{:.0f}".format(n_evaluation_users/optimal_hyperparameters_test_time)
                    result_dataframe.loc[algorithm_row_label, "Recommendation Throughput"] = value_string


        return result_dataframe



    def generate_latex_time_statistics(self, output_file_path,
                                       n_decimals = 2,
                                       n_evaluation_users = None,
                                       table_title = None):

        result_dataframe = self.get_time_statistics_dataframe(n_decimals = n_decimals, n_evaluation_users = n_evaluation_users)
        output_file = open(output_file_path, "w")

        result_dataframe.rename(columns= {"Recommendation Time":"\\begin{tabular}{@{}c@{}}Recommendation\\\\Time\\end{tabular}",
                                          "Recommendation Throughput": "\\begin{tabular}{@{}c@{}}Recommendation\\\\Throughput\\end{tabular}"},
                                inplace = True)


        result_dataframe.fillna("-", inplace = True)

        n_columns = len(result_dataframe.columns)
        latex_code = result_dataframe.to_latex(index = True,
                                               escape = False, #do not automatically escape special characters
                                               column_format = "l|" + "r"*n_columns + "|",
                                               float_format = "{:.4f}".format,
                                               )

        # Replace group separator with \midrule
        separator_old = "(\\\\\\\\\n)\s*algorithm_group_.+\\\\\\\\(\n)"
        separator_midrule = r"\1\\midrule\2"

        latex_code = re.sub(separator_old, separator_midrule, latex_code)


        # Prints table title
        if table_title is not None:
            header_old = "(\\\\toprule\n)"
            header_custom_title = r"\1\t&\t\\multicolumn{{{}}}{{c}}{{{}}}  \\\\\n".format(n_columns, table_title)

            latex_code = re.sub(header_old, header_custom_title, latex_code)

        # ad vline at the end of the line whith multicolumn
        separator_old = "(\\\multicolumn{[^}]*}{[^}]*}{[^}]*}\s*)(\\\\\\\\\n)"
        separator_midrule = r"\1\\vline\2"

        latex_code = re.sub(separator_old, separator_midrule, latex_code)
        latex_code = re.sub("_", " ", latex_code)

        output_file.write(latex_code)
        output_file.flush()

        output_file.close()


