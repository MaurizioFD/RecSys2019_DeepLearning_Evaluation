#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17

@author: Maurizio Ferrari Dacrema
"""

from Recommender_import_list import *

import re
from Utils.seconds_to_biggest_unit import seconds_to_biggest_unit




import numpy as np

import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)


def print_results_latex_table(result_folder_path,
                              results_file_prefix_name,
                              dataset_name,
                              metrics_to_report_list,
                              cutoffs_to_report_list,
                              other_algorithm_list,
                              highlight_best = True,
                              KNN_similarity_to_report_list = list(["cosine"]),
                              ICM_names_to_report_list = list(),
                              ):

    import pickle

    results_file_prefix_name = "{}_{}".format(results_file_prefix_name, dataset_name)

    results_file = open(result_folder_path + "..//" + results_file_prefix_name + "_latex_results.txt", "w")


    metric_name_to_printable_lable_dict = {
        "ROC_AUC": "AUC",
        "PRECISION": "PREC",
        "PRECISION_TEST_LEN": "\\begin{tabular}{@{}c@{}}PREC \\\\ cutoff\\end{tabular}",
        "RECALL": "REC",
        "RECALL_TEST_LEN": "\\begin{tabular}{@{}c@{}}REC \\\\ cutoff\\end{tabular}",
        "MAP": "MAP",
        "MRR": "MRR",
        "NDCG": "NDCG",
        "F1": "F1",
        "HIT_RATE": "HR",
        "ARHR": "ARHR",
        "NOVELTY": "Novelty",
        "DIVERSITY_SIMILARITY": "\\begin{tabular}{@{}c@{}}Div. \\\\ Similarity\\end{tabular}",
        "DIVERSITY_MEAN_INTER_LIST": "\\begin{tabular}{@{}c@{}}Div. \\\\ MIL\\end{tabular}",
        "DIVERSITY_HERFINDAHL": "\\begin{tabular}{@{}c@{}}Div. \\\\ HHI\\end{tabular}",
        "COVERAGE_ITEM": "\\begin{tabular}{@{}c@{}}Cov. \\\\ Item\\end{tabular}",
        "COVERAGE_USER": "\\begin{tabular}{@{}c@{}}Cov. \\\\ User\\end{tabular}",
        "DIVERSITY_GINI": "\\begin{tabular}{@{}c@{}}Div. \\\\ Gini\\end{tabular}",
        "SHANNON_ENTROPY": "\\begin{tabular}{@{}c@{}}Div. \\\\ Shannon\\end{tabular}",
    }


    # Write columns
    columns_datasets_list = "\t&"
    columns_metrics_list = "\t\t&"

    colum_width = len(cutoffs_to_report_list) * len(metrics_to_report_list)

    for experiment_subfolder in [dataset_name]:

        columns_datasets_list += " \\multicolumn{{{}}}{{c}}{{{}}}  \t".format(colum_width, experiment_subfolder)

        if experiment_subfolder != [dataset_name][-1]:
            columns_datasets_list += "&"


        for cutoff in cutoffs_to_report_list:
            for metric in metrics_to_report_list:

                columns_metrics_list += " {}@{} \t".format(metric_name_to_printable_lable_dict[metric], cutoff)

                if not (metric == metrics_to_report_list[-1] and cutoff == cutoffs_to_report_list[-1]):
                    columns_metrics_list += "&"


    columns_datasets_list += "\\\\ \n"
    columns_metrics_list += "\\\\ \n"
    results_file.write(columns_datasets_list)
    results_file.write(columns_metrics_list)
    results_file.flush()





    algorithm_data_to_print_list = get_algorithm_data_to_print_list(
                                    KNN_similarity_to_report_list = KNN_similarity_to_report_list,
                                    ICM_names_to_report_list = ICM_names_to_report_list,
                                    other_algorithm_list = other_algorithm_list)



    data_to_print = np.zeros((len(algorithm_data_to_print_list), colum_width))


    for row_index in range(len(algorithm_data_to_print_list)):

        algorithm = algorithm_data_to_print_list[row_index]["algorithm"]
        algorithm_row_label = algorithm_data_to_print_list[row_index]["algorithm_row_label"]
        algorithm_file_name = algorithm_data_to_print_list[row_index]["algorithm_file_name"]

        column_index = 0

        for _ in [dataset_name]:

            try:
                result_dict = pickle.load(open(result_folder_path +
                                          algorithm_file_name + "_metadata", "rb"))

                result_dict = result_dict["best_result_test"]

            except:
                result_dict = None


            for cutoff in cutoffs_to_report_list:
                for metric in metrics_to_report_list:

                    if result_dict is not None and cutoff in result_dict and metric in result_dict[cutoff]:
                        value = result_dict[cutoff][metric]

                        data_to_print[row_index, column_index] = value
                        #result_row_string += "{:.4f}\t".format(value)
                    else:
                        data_to_print[row_index, column_index] = np.nan
                        #result_row_string += " - \t"

                    if not (cutoff == cutoffs_to_report_list[-1] and metric == metrics_to_report_list[-1]):
                        pass
                        #result_row_string += "&"

                    column_index += 1





    column_threshold_value = data_to_print[-1,:]
    column_better_than_threshold_found = np.zeros_like(column_threshold_value, dtype=np.bool)


    for row_index in range(len(algorithm_data_to_print_list)):

        algorithm = algorithm_data_to_print_list[row_index]["algorithm"]
        algorithm_row_label = algorithm_data_to_print_list[row_index]["algorithm_row_label"]
        algorithm_file_name = algorithm_data_to_print_list[row_index]["algorithm_file_name"]

        result_row_string = algorithm_row_label + "\t&"

        for column_index in range(data_to_print.shape[1]):

            result_to_print = data_to_print[row_index, column_index]

            if np.isnan(result_to_print):
                result_row_string += " - \t"
            else:

                if highlight_best and (
                    result_to_print > column_threshold_value[column_index] or
                    (row_index == len(algorithm_data_to_print_list)-1 and not column_better_than_threshold_found[column_index])):

                    column_better_than_threshold_found[column_index] = True

                    result_row_string += "\\textbf{{{:.4f}}}\t".format(result_to_print)


                else:
                    result_row_string += "{:.4f}\t".format(result_to_print)


            if column_index < data_to_print.shape[1]-1:
                result_row_string += "&"


        result_row_string += "\\\\ \n"
        results_file.write(result_row_string)
        results_file.flush()



    results_file.close()







def print_parameters_latex_table(result_folder_path,
                              results_file_prefix_name,
                              experiment_subfolder_list,
                              other_algorithm_list,
                              KNN_similarity_to_report_list = list(["cosine"]),
                              ICM_names_to_report_list = list(),
                              ):

    import pickle

    parameters_file = open(result_folder_path + results_file_prefix_name + "_latex_parameters.txt", "w")



    import numbers
    columns_datasets_list = "Algorithm\t& Parameter\t& "

    for experiment_subfolder in experiment_subfolder_list:

        columns_datasets_list += " {}\t".format(experiment_subfolder)

        if experiment_subfolder != experiment_subfolder_list[-1]:
            columns_datasets_list += "&"

    columns_datasets_list += "\\\\ \n"
    parameters_file.write(columns_datasets_list)
    parameters_file.flush()




    def get_parameter_values_for_algorithm(algorithm_file_name):

        experiment_subfolder_to_parameters_dict = {}
        parameters_list = None

        for experiment_subfolder in experiment_subfolder_list:

            try:
                parameters_dict = pickle.load(open(result_folder_path + results_file_prefix_name + "_"
                                                   + experiment_subfolder + "/" +
                                                    algorithm_file_name + "_metadata", "rb"))

                parameters_dict = parameters_dict["best_parameters"]

                if parameters_list is None:
                    parameters_list = list(parameters_dict.keys())
                else:
                    assert parameters_list == list(parameters_dict.keys()), "n_parameters {}, len(parameters_dict) {}".format(parameters_list, list(parameters_dict.keys()))

            except:
                parameters_dict = None

            experiment_subfolder_to_parameters_dict[experiment_subfolder] = parameters_dict


        return experiment_subfolder_to_parameters_dict, parameters_list





    algorithm_data_to_print_list = get_algorithm_data_to_print_list(
                                    KNN_similarity_to_report_list = KNN_similarity_to_report_list,
                                    ICM_names_to_report_list = ICM_names_to_report_list,
                                    other_algorithm_list = other_algorithm_list)



    for row_index in range(len(algorithm_data_to_print_list)):

        algorithm = algorithm_data_to_print_list[row_index]["algorithm"]
        algorithm_row_label = algorithm_data_to_print_list[row_index]["algorithm_row_label"]
        algorithm_file_name = algorithm_data_to_print_list[row_index]["algorithm_file_name"]



        experiment_subfolder_to_parameters_dict, parameters_list = get_parameter_values_for_algorithm(algorithm_file_name)

        if parameters_list is not None:

            parameters_file.write("\\midrule\n")

            for current_parameter in parameters_list:

                algorithm_parameter_new_line = ""

                # If first line print parameter name
                if current_parameter == parameters_list[0]:
                    algorithm_parameter_new_line = "\\multirow{{{}}}{{*}}{{{}}}  \t\n".format(len(parameters_list), algorithm_row_label)

                algorithm_parameter_new_line += "\t\t\t\t&"

                # Then print parameter column
                current_parameter_string = current_parameter.replace("_", " ")
                algorithm_parameter_new_line += current_parameter_string + "\t&"

                # and parameter value for each experiment
                for experiment_subfolder in experiment_subfolder_list:

                    if experiment_subfolder_to_parameters_dict[experiment_subfolder] is None:
                        parameter_value = None
                    else:
                        parameter_value = experiment_subfolder_to_parameters_dict[experiment_subfolder][current_parameter]

                    if parameter_value is None:
                        algorithm_parameter_new_line += "- \t"

                    elif not isinstance(parameter_value, list) and any(substring in current_parameter for substring in ["penalty", "rate", "reg", "l1_ratio", "lambda"]):
                        algorithm_parameter_new_line += "{:.2E} \t".format(parameter_value)

                    elif isinstance(parameter_value, bool) or \
                            isinstance(parameter_value, str) or \
                            isinstance(parameter_value, numbers.Integral):
                        algorithm_parameter_new_line += "{} \t".format(parameter_value)

                    elif isinstance(parameter_value, numbers.Real):
                        algorithm_parameter_new_line += "{:.4f} \t".format(parameter_value)
                    else:
                        algorithm_parameter_new_line += "{} \t".format(parameter_value)
                        #assert False, "parameter value not recognized: '{}'".format(parameter_value)


                    if experiment_subfolder != experiment_subfolder_list[-1]:
                        algorithm_parameter_new_line += "&"

                algorithm_parameter_new_line += "\\\\ \n"

                parameters_file.write(algorithm_parameter_new_line)


    parameters_file.close()




def get_algorithm_data_to_print_list(KNN_similarity_to_report_list = list(["cosine"]),
                                    ICM_names_to_report_list = list(),
                                    other_algorithm_list = list(),
                                    ):


    algorithm_list = [
        Random,
        TopPop,
        UserKNNCFRecommender,
        ItemKNNCFRecommender,
        P3alphaRecommender,
        RP3betaRecommender,
        SLIMElasticNetRecommender,
        PureSVDRecommender,
        ItemKNNCBFRecommender,
        ItemKNN_CFCBF_Hybrid_Recommender
    ]

    algorithm_list.extend(other_algorithm_list)



    algorithm_data_to_print_list = []


    for algorithm in algorithm_list:

        algorithm_row_label = algorithm.RECOMMENDER_NAME
        algorithm_row_label = algorithm_row_label.replace("Recommender", "")
        algorithm_row_label = algorithm_row_label.replace("_", " ")
        algorithm_row_label = re.sub("CF$", " CF", algorithm_row_label)
        algorithm_row_label = re.sub("CBF$", " CBF", algorithm_row_label)

        algorithm_file_name = algorithm.RECOMMENDER_NAME

        if algorithm in [ItemKNNCFRecommender,
                         UserKNNCFRecommender]:

            for similarity in KNN_similarity_to_report_list:

                if len(KNN_similarity_to_report_list) >1:
                    algorithm_row_label += " " + similarity

                algorithm_data_to_print_list.append({
                    "algorithm": algorithm,
                    "algorithm_row_label": algorithm_row_label,
                    "algorithm_file_name": algorithm_file_name + "_" + similarity,
                })

        elif algorithm in [ItemKNNCBFRecommender,
                           ItemKNN_CFCBF_Hybrid_Recommender]:

            for ICM_name in ICM_names_to_report_list:
                for similarity in KNN_similarity_to_report_list:

                    if len(ICM_names_to_report_list) >1:
                        algorithm_row_label += " " + ICM_name

                    if len(KNN_similarity_to_report_list) >1:
                        algorithm_row_label += " " + similarity


                    algorithm_data_to_print_list.append({
                        "algorithm": algorithm,
                        "algorithm_row_label": algorithm_row_label,
                        "algorithm_file_name": algorithm_file_name + "_" + ICM_name  + "_" + similarity,
                    })

        else:

            algorithm_data_to_print_list.append({
                "algorithm": algorithm,
                "algorithm_row_label": algorithm_row_label,
                "algorithm_file_name": algorithm_file_name,
            })


    return algorithm_data_to_print_list




def print_time_statistics_latex_table(result_folder_path, dataset_name, results_file_prefix_name,
                                      other_algorithm_list = list(),
                                      n_validation_users = None,
                                      n_test_users = None,
                                      KNN_similarity_to_report_list = list(["cosine"]),
                                      ICM_names_to_report_list = list(),
                                      n_decimals = 4):

    import pickle
    import numpy as np

    results_file_root_name = "{}_{}".format(results_file_prefix_name, dataset_name)

    results_file = open(result_folder_path + "..//" + results_file_root_name + "_latex_time.txt", "w")



    def mean_and_stdd_of_list(data_list):

        data_list = np.array(data_list)

        mean = np.mean(data_list)

        if len(data_list) == 1:
            stddev = 0.0
        else:
            stddev = np.std(data_list, ddof=1)

        return mean, stddev


    # Write columns
    columns_datasets_list = "\t&"
    columns_metrics_list = "\t\t&"

    column_name_list = ["\\begin{tabular}{@{}c@{}}Train time\\end{tabular}",
                        #"validation_time",
                        #"\\begin{tabular}{@{}c@{}}validation \\ usr/sec\end{tabular}",
                        "\\begin{tabular}{@{}c@{}}Recommendation\\end{tabular}",
                        "\\begin{tabular}{@{}c@{}}Recommendation \\\\ {[usr/s]}\\end{tabular}"]


    for dataset_name in [dataset_name]:

        colum_width = len(column_name_list)

        columns_datasets_list += " \\multicolumn{{{}}}{{c}}{{{}}}  \t".format(colum_width, dataset_name)

        if dataset_name != [dataset_name][-1]:
            columns_datasets_list += "&"

        for column_name in column_name_list:

            columns_metrics_list += " {} \t".format(column_name)

            if column_name != column_name_list[-1]:
                columns_metrics_list += "\n\t&"


    columns_datasets_list += "\\\\ \n"
    columns_metrics_list += "\\\\ \n"
    results_file.write(columns_datasets_list)
    results_file.write(columns_metrics_list)
    results_file.flush()



    algorithm_data_to_print_list = get_algorithm_data_to_print_list(
                                    KNN_similarity_to_report_list = KNN_similarity_to_report_list,
                                    ICM_names_to_report_list = ICM_names_to_report_list,
                                    other_algorithm_list = other_algorithm_list)



    for row_index in range(len(algorithm_data_to_print_list)):

        algorithm = algorithm_data_to_print_list[row_index]["algorithm"]
        algorithm_row_label = algorithm_data_to_print_list[row_index]["algorithm_row_label"]
        algorithm_file_name = algorithm_data_to_print_list[row_index]["algorithm_file_name"]

        result_row_string = algorithm_row_label + "\t&"

        for experiment_subfolder in [result_folder_path]:

            try:
                result_dict = pickle.load(open(experiment_subfolder + algorithm_file_name + "_metadata", "rb"))
            except:
                result_dict = None


            if result_dict is not None:

                data_list = result_dict["train_time_list"]
                data_list = np.array(data_list)

                data_list_not_none_mask = np.array([val is not None for val in data_list])
                data_list = data_list[data_list_not_none_mask]

                mean, stddev = mean_and_stdd_of_list(data_list)

                if len(result_dict["train_time_list"]) > 1:
                    result_row_string+= "{:.{n_decimals}f} $\\pm$ {:.{n_decimals}f} [s]\t&".format(mean, stddev, n_decimals=n_decimals)
                else:
                    new_time_value, new_time_unit = seconds_to_biggest_unit(mean)

                    if new_time_unit == "s":
                        result_row_string+= "{:.{n_decimals}f} [{}] \t&".format(mean, "s", n_decimals=n_decimals)
                    else:
                        result_row_string+= "{:.{n_decimals}f} [{}] / {:.{n_decimals}f} [{}]\t&".format(mean, "s", new_time_value, new_time_unit, n_decimals=n_decimals)

                # mean, stddev = mean_and_stdd_of_list(result_dict["evaluation_time_list"])
                # result_row_string+= "{:.{n_decimals}f} $\pm$ {:.{n_decimals}f}\t&".format(mean, stddev, n_decimals=n_decimals)
                #
                # if n_validation_users is not None:
                #     evaluation_time = result_dict["evaluation_time_list"][-1]
                #     result_row_string+= "{:.0f}\t&".format(n_validation_users/evaluation_time)
                # else:
                #     result_row_string+=" - \t&"



                non_nan_test_time = []
                for value in result_dict["evaluation_test_time_list"]:
                    if value is not None:
                        non_nan_test_time.append(value)

                mean, stddev = mean_and_stdd_of_list(non_nan_test_time)

                if len(non_nan_test_time) > 1:
                    result_row_string+= "{:.{n_decimals}f} $\\pm$ {:.{n_decimals}f}  [s]\t&".format(mean, stddev, n_decimals=n_decimals)
                else:
                    new_time_value, new_time_unit = seconds_to_biggest_unit(mean)

                    # result_row_string+= "{:.{n_decimals}f} \t&".format(mean, n_decimals=n_decimals)
                    if new_time_unit == "s":
                        result_row_string+= "{:.{n_decimals}f} [{}] \t&".format(mean, "s", n_decimals=n_decimals)
                    else:
                        result_row_string+= "{:.{n_decimals}f} [{}] / {:.{n_decimals}f} [{}]\t&".format(mean, "s", new_time_value, new_time_unit, n_decimals=n_decimals)


                if n_test_users is not None:
                    evaluation_time = non_nan_test_time[-1]
                    result_row_string+= "{:.0f}\t".format(n_test_users/evaluation_time)
                else:
                    result_row_string+=" - \t"


            else:
                result_row_string+=" - \t& - \t& - "


        result_row_string += "\\\\ \n"
        results_file.write(result_row_string)
        results_file.flush()


    results_file.close()


